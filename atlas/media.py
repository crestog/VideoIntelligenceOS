"""
Playback.

*"The videos should be instantly playable also as I click them... lightning fast
speed, I don't care how you achieve this logic (how cunning, smart, pre-loading,
or very optimised, better architecture)."*

The honest problem: the video is a 5–40 MB file sitting in a Telegram channel,
and fetching it takes seconds. Nothing makes that transfer instant. So the trick
is to have already done it before the click happens.

Four mechanisms, in the order they pay off:

**Local first.** If Atlas is running on the machine that harvested the reel, the
file is already on disk and `video_index.local_path` points at it. Zero network.
This is the common case on Kaggle, and it is checked before anything else.

**Speculative prefetch.** Every search response kicks off downloads for the top
few results before the person has clicked anything. By the time the eye has
travelled to the first card, the file behind it is usually resident. This is the
single biggest win, and it costs bandwidth for videos nobody opens — which is
the right trade when the alternative is a spinner on every click.

**Hover intent.** The card asks for its video on `pointerenter` and on keyboard
focus. Between deciding to click and clicking, a person spends 200–400 ms;
that is not the whole download, but it is a head start and it is free.

**Range serving.** Playback goes through a hand-written 206 responder rather
than FileResponse, because seeking in a `<video>` element requires byte ranges
and Starlette only grew range support recently. Doing it here works on every
version, and lets a partially-downloaded file serve the bytes it already has.

Eviction is LRU by access time against `VIDEO_CACHE_GB`. The cache lives on
Kaggle's scratch disk, which is wiped between sessions anyway, so nothing here
is precious — every file can be fetched again from the channel.
"""

import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time

from . import config
from .tgchannel import log

_LOCK = threading.RLock()
_STATE = {}              # video_key → {status, got, total, note, at}
_INFLIGHT = {}           # video_key → threading.Event
_LAST_EVICT = [0.0]

# Two at a time. Telegram throttles hard on parallel downloads from one bot,
# and a wide pool turns into a wall of FloodWait — slower overall than a
# narrow one that never trips it.
_SLOTS = threading.Semaphore(2)


def _now() -> float:
    return time.time()


def _cache_path(video_key: str) -> str:
    return os.path.join(config.VIDEO_CACHE, f"{video_key}.mp4")


def _set(key: str, **kw) -> None:
    with _LOCK:
        slot = _STATE.setdefault(key, {"status": "unknown", "got": 0,
                                       "total": 0, "note": ""})
        slot.update(kw)
        slot["at"] = _now()


def resident(local_path: str, video_key: str) -> bool:
    """True when this video can be played without touching the network.

    The database records the path the harvester wrote, but on Kaggle that path
    lives on the scratch disk and is wiped between sessions. So a stored path
    is a hint, not a fact — the disk is asked. Without this, every card claims
    to be resident after a restart while `resolve()` correctly says `remote`.
    """
    for p in (local_path, _cache_path(str(video_key))):
        if not p:
            continue
        try:
            if os.path.getsize(p) > 0:
                return True
        except OSError:
            continue
    return False


_RESIDENT = {"at": 0.0, "keys": frozenset()}
_RESIDENT_TTL = 15.0


def resident_keys(conn: sqlite3.Connection, force: bool = False) -> frozenset:
    """Every video key playable right now, as one cached set.

    Asking the disk per row would be correct but wasteful: the status poll and
    the library filter both want this, several times a minute, for the whole
    corpus. One pass every 15 s is accurate enough for a badge — a download
    finishing early only means the badge appears a few seconds late, and the
    player never trusts this set anyway. `resolve()` re-checks the disk on the
    click that matters.
    """
    if not force and _now() - _RESIDENT["at"] < _RESIDENT_TTL:
        return _RESIDENT["keys"]

    keys = set()
    try:
        for name in os.listdir(config.VIDEO_CACHE):
            stem, ext = os.path.splitext(name)
            # Video keys are the digits of a Telegram message id, so anything
            # else in here is not a video. Zero-byte files are a download that
            # died between create and write — claiming those are playable puts
            # a spinner on a card that promised none.
            if ext != ".mp4" or not stem.isdigit():
                continue
            try:
                if os.path.getsize(os.path.join(config.VIDEO_CACHE, name)) > 0:
                    keys.add(stem)
            except OSError:
                continue
    except OSError:
        pass

    try:
        for key, path in conn.execute(
                "SELECT video_key, local_path FROM video_index "
                "WHERE local_path IS NOT NULL AND local_path <> ''"):
            if key in keys or not path:
                continue
            try:
                if os.path.getsize(path) > 0:
                    keys.add(key)
            except OSError:
                continue
    except sqlite3.Error:
        pass

    out = frozenset(keys)
    _RESIDENT.update(at=_now(), keys=out)
    return out


def invalidate_resident() -> None:
    """Forget the residency set — call after a download or a cache wipe."""
    _RESIDENT["at"] = 0.0


def state(video_key: str) -> dict:
    """What the UI polls while a download is in flight."""
    key = str(video_key)
    path = _cache_path(key)
    with _LOCK:
        slot = dict(_STATE.get(key) or {})
    if os.path.exists(path):
        size = os.path.getsize(path)
        if not slot.get("status") or slot.get("status") in ("ready", "unknown"):
            return {"status": "ready", "got": size, "total": size,
                    "source": "cache"}
        slot["got"] = max(slot.get("got", 0), size)
    if not slot:
        return {"status": "absent", "got": 0, "total": 0}
    pct = 0
    if slot.get("total"):
        pct = round(100.0 * slot.get("got", 0) / slot["total"], 1)
    slot["percent"] = pct
    return slot


# ══════════════════════════════════════════════════════════════════════════
# RESOLUTION
# ══════════════════════════════════════════════════════════════════════════
def _row(conn: sqlite3.Connection, video_key: str) -> dict:
    try:
        cur = conn.execute(
            "SELECT video_key, msg_id, local_path, duration, title, poster "
            "FROM video_index WHERE video_key = ?", (str(video_key),))
    except sqlite3.Error:
        return {}
    row = cur.fetchone()
    if not row:
        return {}
    return dict(zip([d[0] for d in cur.description], row))


def resolve(conn: sqlite3.Connection, video_key: str) -> dict:
    """Where this video can be played from, right now.

    `local` means the harvester's own copy is still on disk — the fastest
    possible answer and the reason this check comes first. `cache` means Atlas
    downloaded it earlier. `remote` means it exists in the channel but not here
    yet, and `missing` means there is no message id to fetch it with.
    """
    key = str(video_key)
    info = _row(conn, key)

    local = info.get("local_path")
    if local and os.path.exists(local) and os.path.getsize(local) > 0:
        return {"key": key, "where": "local", "path": local,
                "size": os.path.getsize(local),
                "duration": info.get("duration"), "msg_id": info.get("msg_id")}

    cached = _cache_path(key)
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        return {"key": key, "where": "cache", "path": cached,
                "size": os.path.getsize(cached),
                "duration": info.get("duration"), "msg_id": info.get("msg_id")}

    # The video key is the digits of the Telegram message id, so even a video
    # with no metadata row is still fetchable.
    msg_id = info.get("msg_id")
    if not msg_id:
        try:
            msg_id = int(key)
        except (TypeError, ValueError):
            msg_id = None
    if msg_id:
        return {"key": key, "where": "remote", "path": None, "size": 0,
                "duration": info.get("duration"), "msg_id": int(msg_id)}
    return {"key": key, "where": "missing", "path": None, "size": 0,
            "duration": info.get("duration"), "msg_id": None}


# ══════════════════════════════════════════════════════════════════════════
# FETCHING
# ══════════════════════════════════════════════════════════════════════════
def _download(video_key: str, msg_id: int) -> None:
    """Pull one video into the cache. Runs on a worker thread."""
    key = str(video_key)
    dest = _cache_path(key)
    tmp = dest + ".part"
    _set(key, status="queued", note="")

    with _SLOTS:
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            _set(key, status="ready")
            return
        _set(key, status="downloading", got=0)

        def progress(current, total):
            _set(key, got=int(current), total=int(total or 0))

        ok = False
        try:
            from . import tgchannel
            ok = tgchannel.download_by_id(msg_id, tmp, progress=progress)
        except Exception as e:
            _set(key, status="error", note=f"{type(e).__name__}: {e}")
            log(f"video {key} download failed — {type(e).__name__}: {e}")

        if ok and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            try:
                os.replace(tmp, dest)
                _set(key, status="ready", got=os.path.getsize(dest),
                     total=os.path.getsize(dest))
            except OSError as e:
                _set(key, status="error", note=str(e))
        else:
            _set(key, status="error",
                 note=_STATE.get(key, {}).get("note") or "download returned nothing")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    with _LOCK:
        ev = _INFLIGHT.pop(key, None)
    if ev:
        ev.set()
    invalidate_resident()
    _maybe_evict()


def ensure(conn: sqlite3.Connection, video_key: str, wait: float = 0.0) -> dict:
    """Make sure the video is here, starting a download if not.

    `wait=0` returns immediately and is what prefetch and hover use — the point
    is to start the transfer, not to block on it. A positive wait is for the
    click path, where the caller would rather hold the connection for a second
    than hand back a 404 for a file that is 90% there.
    """
    key = str(video_key)
    found = resolve(conn, key)
    if found["where"] in ("local", "cache"):
        _touch(found["path"])
        return {"ok": True, "where": found["where"], "state": "ready"}
    if found["where"] == "missing":
        return {"ok": False, "where": "missing",
                "note": "no Telegram message id for this video"}

    with _LOCK:
        ev = _INFLIGHT.get(key)
        fresh = ev is None
        if fresh:
            ev = threading.Event()
            _INFLIGHT[key] = ev
    if fresh:
        threading.Thread(target=_download, args=(key, found["msg_id"]),
                         name=f"atlas-fetch-{key}", daemon=True).start()
    if wait > 0:
        ev.wait(timeout=wait)
    return {"ok": True, "where": "remote", "state": state(key)["status"]}


def prefetch(conn: sqlite3.Connection, keys: list, limit: int = None) -> int:
    """Warm the cache for a page of results. Returns how many were started.

    Called from the search handler with the top results, before anybody has
    clicked. Files already present are skipped for free, so a repeated search
    costs nothing.
    """
    limit = config.PREFETCH_TOP_N if limit is None else limit
    started = 0
    for key in list(keys)[:limit]:
        try:
            found = resolve(conn, key)
        except sqlite3.Error:
            continue
        if found["where"] in ("local", "cache", "missing"):
            continue
        with _LOCK:
            if key in _INFLIGHT:
                continue
        ensure(conn, key, wait=0)
        started += 1
    return started


def prefetch_async(db_path: str, keys: list, limit: int = None) -> None:
    """Prefetch on a thread with its own connection.

    A search response must not wait on resolution queries, and the request's
    own connection cannot cross a thread boundary safely.
    """
    if not keys:
        return

    def run():
        try:
            conn = sqlite3.connect(db_path, timeout=30.0,
                                   check_same_thread=False)
        except sqlite3.Error:
            return
        try:
            prefetch(conn, keys, limit)
        except Exception:
            pass
        finally:
            conn.close()

    threading.Thread(target=run, name="atlas-prefetch", daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════
# CACHE HYGIENE
# ══════════════════════════════════════════════════════════════════════════
def _touch(path: str) -> None:
    """Mark a file as recently used, so eviction takes something else."""
    if not path or not path.startswith(config.VIDEO_CACHE):
        return
    try:
        os.utime(path, None)
    except OSError:
        pass


def cache_stats() -> dict:
    total = 0
    files = 0
    try:
        for name in os.listdir(config.VIDEO_CACHE):
            p = os.path.join(config.VIDEO_CACHE, name)
            if os.path.isfile(p):
                total += os.path.getsize(p)
                files += 1
    except OSError:
        pass
    return {"files": files, "bytes": total,
            "gb": round(total / 1073741824, 2),
            "limit_gb": config.VIDEO_CACHE_GB}


def _maybe_evict() -> None:
    """Drop least-recently-used videos when the cache outgrows its budget.

    Rate-limited to once every 30 s because it stats the whole directory and
    it runs after every download.
    """
    if _now() - _LAST_EVICT[0] < 30:
        return
    _LAST_EVICT[0] = _now()
    limit = config.VIDEO_CACHE_GB * 1073741824
    try:
        entries = []
        total = 0
        for name in os.listdir(config.VIDEO_CACHE):
            p = os.path.join(config.VIDEO_CACHE, name)
            if not os.path.isfile(p):
                continue
            st = os.stat(p)
            entries.append((st.st_atime, st.st_size, p))
            total += st.st_size
        if total <= limit:
            return
        entries.sort()
        freed = 0
        for _atime, size, path in entries:
            if total - freed <= limit * 0.85:
                break
            try:
                os.remove(path)
                freed += size
            except OSError:
                continue
        if freed:
            invalidate_resident()
            log(f"video cache trimmed — freed {freed / 1048576:.0f} MB")
    except OSError:
        pass


def clear_cache() -> dict:
    """Empty the video and poster caches. Everything here is re-fetchable."""
    freed = 0
    for folder in (config.VIDEO_CACHE, config.POSTER_CACHE):
        try:
            for name in os.listdir(folder):
                p = os.path.join(folder, name)
                try:
                    if os.path.isfile(p):
                        freed += os.path.getsize(p)
                        os.remove(p)
                    else:
                        shutil.rmtree(p, ignore_errors=True)
                except OSError:
                    continue
        except OSError:
            continue
    with _LOCK:
        _STATE.clear()
    invalidate_resident()
    return {"ok": True, "freed_mb": round(freed / 1048576, 1)}


# ══════════════════════════════════════════════════════════════════════════
# POSTERS
# ══════════════════════════════════════════════════════════════════════════
_FFMPEG = shutil.which("ffmpeg")


def poster(conn: sqlite3.Connection, video_key: str, at: float = None) -> str:
    """A still frame, cached on disk. Returns a path or "".

    `at` matters more than it looks. A result card can show the frame at the
    moment that actually matched instead of the first frame of the reel, which
    turns a grid of near-identical intro shots into a grid of answers. Frames
    are cached per (video, second), so scrubbing a ribbon does not re-run
    ffmpeg for a position already seen.
    """
    key = str(video_key)
    found = resolve(conn, key)
    if found["where"] not in ("local", "cache"):
        return ""
    if not _FFMPEG:
        return ""

    pos = 0.0 if at is None else max(0.0, float(at))
    stamp = f"{pos:.0f}"
    dest = os.path.join(config.POSTER_CACHE, f"{key}_{stamp}.jpg")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    # -ss before -i seeks by keyframe without decoding the file up to that
    # point: milliseconds instead of seconds on a long clip. The frame may land
    # slightly early, which does not matter for a thumbnail.
    cmd = [_FFMPEG, "-nostdin", "-loglevel", "error", "-ss", f"{pos:.2f}",
           "-i", found["path"], "-frames:v", "1", "-vf", "scale=480:-2",
           "-q:v", "5", "-y", dest]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=25)
        if r.returncode == 0 and os.path.exists(dest) \
                and os.path.getsize(dest) > 0:
            return dest
    except (subprocess.SubprocessError, OSError):
        pass
    return ""


# ══════════════════════════════════════════════════════════════════════════
# RANGE SERVING
# ══════════════════════════════════════════════════════════════════════════
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK = 512 * 1024


def range_plan(path: str, range_header: str) -> dict:
    """Work out what bytes to send for a `<video>` request.

    Browsers open a video with `Range: bytes=0-` and then seek with explicit
    ranges. Answering 200-with-everything makes the first frame wait for the
    whole file and disables seeking entirely, so a 206 with the right headers is
    not a nicety — it is what makes playback start immediately.
    """
    size = os.path.getsize(path)
    ctype = mimetypes.guess_type(path)[0] or "video/mp4"
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": ctype,
        # The channel is immutable and the key is the message id, so a cached
        # response is never stale. This is what makes a re-open instant.
        "Cache-Control": "public, max-age=604800, immutable",
    }
    m = _RANGE.match(range_header or "")
    if not m or size == 0:
        headers["Content-Length"] = str(size)
        return {"status": 200, "start": 0, "end": size - 1, "size": size,
                "headers": headers}

    raw_start, raw_end = m.group(1), m.group(2)
    if raw_start == "":
        # A suffix range — "the last N bytes". Rare, but mp4 players use it to
        # find a moov atom stored at the end of the file.
        length = int(raw_end or 0)
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    start = max(0, min(start, size - 1))
    end = max(start, min(end, size - 1))

    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return {"status": 206, "start": start, "end": end, "size": size,
            "headers": headers}


def stream(path: str, start: int, end: int):
    """Yield one byte range. Closes the handle even if the client disconnects
    mid-play, which happens constantly — people scrub."""
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            block = f.read(min(_CHUNK, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block
