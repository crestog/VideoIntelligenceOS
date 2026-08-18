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


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe(video_key: str) -> str:
    """A video key as a filename component.

    Keys are Instagram shortcodes or `up_<msg_id>`, both of which are already
    safe — but a key that reached here from a restored database written by an
    older revision is not guaranteed to be, and a `/` in one would silently
    write outside the cache directory.
    """
    return _UNSAFE.sub("_", str(video_key))[:120] or "unknown"


# The suffix that separates a proxy's cache identity from its video's. A dot,
# because `_safe` permits one and `os.path.splitext` therefore reads
# `1234.proxy.mp4`'s stem as `1234.proxy` — which is exactly the string
# `proxy_key` returns, so `resident_keys` can recognise a cached proxy by
# splitting on it rather than by guessing.
_PROXY_SUFFIX = ".proxy"


def proxy_key(video_key: str) -> str:
    """The cache identity of this video's 480p proxy."""
    return f"{_safe(video_key)}{_PROXY_SUFFIX}"


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

    A cached 480p proxy counts, because `resolve()` will play it. Checked last:
    it is the least likely of the three to be present and the only one that
    needs a string built for it.
    """
    for p in (local_path, _cache_path(str(video_key)),
              _cache_path(proxy_key(video_key))):
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
            if ext != ".mp4":
                continue
            # `1234.proxy.mp4` is this video's 480p proxy, and playing it needs
            # no network either — so it counts as resident, under the video's
            # own key. Without this line a proxy-only video wears a "fetching"
            # badge and then plays instantly, which reads as a bug in the badge.
            if stem.endswith(_PROXY_SUFFIX):
                stem = stem[:-len(_PROXY_SUFFIX)]
            # Video keys are the digits of a Telegram message id, so anything
            # else in here is not a video. Zero-byte files are a download that
            # died between create and write — claiming those are playable puts
            # a spinner on a card that promised none.
            if not stem.isdigit():
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


def artifact(conn: sqlite3.Connection, video_key: str, kind: str) -> dict:
    """The best row of the `artifact` table for this kind, or `{}`.

    The processing plane's `artifacts` pass renders a 480p `+faststart` proxy, a
    poster, a sprite sheet and a waveform per video, and now uploads them and
    records the message id. The table lands in Atlas generically — it is not in
    `reflect._ATLAS_OWN`, so `import_shard`'s inferred-table loop creates it from
    the shard's own columns.

    Which is why every failure here is `{}` rather than an exception. Three
    entirely normal databases have no usable row: one whose shards predate the
    upload (the `msg_id` column was all-NULL, and `_sql_type` drops an all-None
    column, so the table exists without it), one that has imported no shard
    carrying artifacts at all, and one whose `artifacts` pass failed for this
    video. In all three the caller must fall back to the original, so a missing
    table and a missing row have to be the same answer.

    Why "best" and not "the" row
    ────────────────────────────
    In the processing plane `(video_key, kind)` is the primary key, so there is
    exactly one row. In Atlas it is not: `ingest._dedup_columns` infers the
    unique key from a single shard's rows, and for this table more than one
    candidate is unique within a batch. So a re-rendered proxy uploaded to a new
    message can land as a second row rather than replacing the first, and taking
    whichever sqlite returns first would eventually point playback at a
    superseded message.

    Resolved here rather than by constraining the table, because a constraint
    inferred per shard cannot be relied on retroactively and this cannot be
    wrong: prefer a row that carries a message id, and among those the newest.
    Ordering happens in Python because `created_at` is not guaranteed to be a
    column — an all-None column is dropped at ingest, so `ORDER BY` on one that
    was dropped would take the whole lookup down.
    """
    try:
        cur = conn.execute(
            "SELECT * FROM artifact WHERE video_key = ? AND kind = ?",
            (str(video_key), str(kind)))
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
    except sqlite3.Error:
        return {}
    if not rows:
        return {}

    def clean(got: dict) -> dict:
        try:
            got["msg_id"] = int(got.get("msg_id") or 0) or None
        except (TypeError, ValueError):
            got["msg_id"] = None
        try:
            got["bytes"] = int(got.get("bytes") or 0)
        except (TypeError, ValueError):
            got["bytes"] = 0
        return got

    def rank(got: dict) -> tuple:
        try:
            when = float(got.get("created_at") or 0.0)
        except (TypeError, ValueError):
            when = 0.0
        return (1 if got.get("msg_id") else 0, when, got.get("msg_id") or 0)

    return max((clean(r) for r in rows), key=rank)


def resolve(conn: sqlite3.Connection, video_key: str) -> dict:
    """Where this video can be played from, right now.

    `local` means the harvester's own copy is still on disk — the fastest
    possible answer and the reason this check comes first. `cache` means Atlas
    downloaded it earlier. `remote` means it exists in the channel but not here
    yet, and `missing` means there is no message id to fetch it with.

    Two identities, not one
    ───────────────────────
    Every answer carries a **`cache_key`** as well as a `key`. They differ when
    the bytes on offer are not the original reel: the processing plane's
    `artifacts` pass renders a 480p `+faststart` proxy and now uploads it, and
    for playback that file is strictly better — two megabytes instead of forty,
    a moov atom at the front, and a seek that lands on the first request.

    So the proxy has to cache under its own name. Everything downstream —
    `_cache_path`, the `.sparse` file, `_sparse_index`, `fill`, `state` — is
    keyed by whatever string it is handed, and handing them the plain video key
    for proxy bytes would let `sparse_hit` return chunks of the *original* to a
    range computed against the *proxy's* length. That splice would play as
    corruption, so the two files get two keys and can never mix.

    Callers that only want to know "can this play, and from where" can keep
    ignoring the field; callers that touch the cache must pass `cache_key` on.

    Order of preference, cheapest first: an original already on disk beats
    everything, because it costs no network at all and is better quality. After
    that the proxy wins — a cached proxy over a channel fetch, and a channel
    fetch of two megabytes over a channel fetch of forty.
    """
    key = str(video_key)
    info = _row(conn, key)

    local = info.get("local_path")
    if local and os.path.exists(local) and os.path.getsize(local) > 0:
        return {"key": key, "cache_key": key, "via": "original",
                "where": "local", "path": local,
                "size": os.path.getsize(local),
                "duration": info.get("duration"), "msg_id": info.get("msg_id")}

    cached = _cache_path(key)
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        return {"key": key, "cache_key": key, "via": "original",
                "where": "cache", "path": cached,
                "size": os.path.getsize(cached),
                "duration": info.get("duration"), "msg_id": info.get("msg_id")}

    proxy = artifact(conn, key, "proxy")
    if proxy.get("msg_id"):
        pkey = proxy_key(key)
        pcached = _cache_path(pkey)
        if os.path.exists(pcached) and os.path.getsize(pcached) > 0:
            return {"key": key, "cache_key": pkey, "via": "proxy",
                    "where": "cache", "path": pcached,
                    "size": os.path.getsize(pcached),
                    "duration": info.get("duration"),
                    "msg_id": int(proxy["msg_id"])}
        return {"key": key, "cache_key": pkey, "via": "proxy",
                "where": "remote", "path": None,
                "size": int(proxy.get("bytes") or 0),
                "duration": info.get("duration"),
                "msg_id": int(proxy["msg_id"])}

    # The video key is the digits of the Telegram message id, so even a video
    # with no metadata row is still fetchable.
    msg_id = info.get("msg_id")
    if not msg_id:
        try:
            msg_id = int(key)
        except (TypeError, ValueError):
            msg_id = None
    if not msg_id:
        # A video keyed by shortcode, known only to the new capture plane, has
        # no digits to fall back on — but if it has an asset set, its manifest
        # recorded the video's own message id when the clips were published.
        # Refusing to play a video whose clips are already in hand would be a
        # strange way to fail.
        try:
            from . import index as _index
            msg_id = int((_index.part_of(conn, key, "video") or {})
                         .get("msg_id") or 0) or None
        except Exception:                                   # noqa: BLE001
            msg_id = None
    if msg_id:
        return {"key": key, "cache_key": key, "via": "original",
                "where": "remote", "path": None, "size": 0,
                "duration": info.get("duration"), "msg_id": int(msg_id)}
    return {"key": key, "cache_key": key, "via": "original",
            "where": "missing", "path": None, "size": 0,
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

    Everything past `resolve` uses `cache_key`, not `video_key`. When the answer
    is the 480p proxy that is a different string, and it has to be: the download
    lands at `<cache_key>.mp4`, so keying the in-flight event or the destination
    by the video would write proxy bytes over the path the original claims.
    """
    key = str(video_key)
    found = resolve(conn, key)
    ck = found.get("cache_key") or key
    if found["where"] in ("local", "cache"):
        _touch(found["path"])
        return {"ok": True, "where": found["where"], "state": "ready",
                "via": found.get("via", "original")}
    if found["where"] == "missing":
        return {"ok": False, "where": "missing",
                "note": "no Telegram message id for this video"}

    with _LOCK:
        ev = _INFLIGHT.get(ck)
        fresh = ev is None
        if fresh:
            ev = threading.Event()
            _INFLIGHT[ck] = ev
    if fresh:
        threading.Thread(target=_download, args=(ck, found["msg_id"]),
                         name=f"atlas-fetch-{ck}", daemon=True).start()
    if wait > 0:
        ev.wait(timeout=wait)
    return {"ok": True, "where": "remote", "state": state(ck)["status"],
            "via": found.get("via", "original")}


def prefetch(conn: sqlite3.Connection, keys: list, limit: int = None) -> int:
    """Warm a page of results before anybody clicks. Returns how many started.

    What "warm" means changed when playback started streaming. Pulling whole
    files ahead of a click was the old way to make playback instant, and it
    cost 30 MB of bandwidth per video for a guess. Now the first click only
    needs two 1 MiB chunks — the head, where playback begins, and the tail,
    where an mp4 written by a phone usually keeps its moov atom. Fetch those
    into the sparse file and the video starts from disk with no round trip at
    all, for a fifteenth of the traffic.

    Without MTProto there is no chunk access, so this falls back to the whole
    file over the Bot API, which is the only thing that transport can do.

    Where a proxy exists this warms the proxy, because that is what a click will
    play — and at 480p the head and tail chunks are a larger fraction of the
    file, so the same two chunks buy more of it.
    """
    limit = config.PREFETCH_TOP_N if limit is None else limit
    from . import tgchannel
    stream_ok = tgchannel.mtproto_ready()
    started = 0
    for key in list(keys)[:limit]:
        try:
            found = resolve(conn, key)
        except sqlite3.Error:
            continue
        if found["where"] in ("local", "cache", "missing"):
            continue
        ck = found.get("cache_key") or key
        with _LOCK:
            if ck in _INFLIGHT:
                continue
        if stream_ok:
            # Already warmed is a success, not a reason to fall back to
            # pulling the entire file.
            if warm(ck, found["msg_id"]):
                started += 1
            continue
        ensure(conn, key, wait=0)
        started += 1
    return started


_WARMED = set()
_WARM_SLOTS = threading.Semaphore(3)


def warm(video_key: str, msg_id: int) -> bool:
    """Fetch the head and tail chunks on a thread. True if it was dispatched.

    Kept off the download semaphore deliberately: warming must never queue
    behind a full download, because the whole point is that it finishes in the
    time it takes to move the mouse.
    """
    key = str(video_key)
    if not msg_id:
        return False
    with _LOCK:
        if key in _WARMED:
            return False
        _WARMED.add(key)

    def run():
        from . import tgchannel
        if not tgchannel.mtproto_ready():
            with _LOCK:
                _WARMED.discard(key)
            return
        with _WARM_SLOTS:
            try:
                message, facts = _message_for(key, msg_id)
                if message is None:
                    return
                size = facts["size"]
                last = max(0, (size - 1) // _TG_CHUNK)
                part = _cache_path(key) + ".sparse"
                index = _sparse_index(key)
                for chunk_no in ({0, last} if last else {0}):
                    with _SPARSE_LOCK:
                        if chunk_no in index:
                            continue
                    for piece in tgchannel.stream_chunks(
                            message, first_chunk=chunk_no, chunk_limit=1):
                        _remember_chunk(part, index, chunk_no, piece, size)
                        break
                _maybe_promote(key, part, index, size)
            except Exception as exc:                       # noqa: BLE001
                log(f"warm {key} skipped — {type(exc).__name__}: {exc}", "WARN")

    threading.Thread(target=run, name=f"atlas-warm-{key}", daemon=True).start()
    return True


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


# The play handler is the one caller outside this module that has a legitimate
# reason to say "this file was just watched" — eviction should never take the
# video somebody is looking at.
touch = _touch


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
            entries.append((st.st_atime, st.st_size, p, name))
            total += st.st_size
        if total <= limit:
            return
        entries.sort()
        freed = 0
        for _atime, size, path, name in entries:
            if total - freed <= limit * 0.85:
                break
            try:
                os.remove(path)
                freed += size
            except OSError:
                continue
            # A sparse file's chunk index is the only record of which of its
            # bytes are real. Removing the file without forgetting the index
            # would let a later request serve a hole — a run of zeros — as if
            # it were video. Forget it in the same breath.
            if name.endswith(".sparse"):
                stem = name[:-len(".sparse")]
                key = os.path.splitext(stem)[0]
                with _SPARSE_LOCK:
                    _SPARSE.pop(key, None)
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
        _WARMED.clear()
    with _SPARSE_LOCK:
        _SPARSE.clear()
    _MSG_CACHE.clear()
    invalidate_resident()
    return {"ok": True, "freed_mb": round(freed / 1048576, 1)}


# ══════════════════════════════════════════════════════════════════════════
# POSTERS
# ══════════════════════════════════════════════════════════════════════════
_FFMPEG = shutil.which("ffmpeg")

_ARTIFACT_LOCK = threading.RLock()
_ARTIFACT_INFLIGHT: dict = {}


def artifact_file(conn: sqlite3.Connection, video_key: str, kind: str,
                  ext: str = ".jpg", wait: float = 20.0) -> str:
    """Fetch one small artifact out of the channel onto disk. Path or "".

    Only for the small ones — the poster, the sprite sheet, the waveform. The
    proxy is a video and goes through the ordinary cache and range machinery, so
    it deliberately does not come through here; downloading a whole proxy to
    answer a thumbnail request would defeat the point of having one.

    Concurrency is the same bargain `clip_fetch` makes, for the same reason: a
    grid of twenty cards asks for twenty posters at once and several ask twice.
    One in-flight fetch per artifact, everyone else waits on it.
    """
    row = artifact(conn, video_key, kind)
    if not row.get("msg_id") and not (row.get("file_id") or ""):
        return ""

    dest = os.path.join(config.POSTER_CACHE,
                        f"{_safe(video_key)}_{_safe(kind)}{ext}")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    ident = f"{video_key}#{kind}"
    with _ARTIFACT_LOCK:
        ev = _ARTIFACT_INFLIGHT.get(ident)
        mine = ev is None
        if mine:
            ev = threading.Event()
            _ARTIFACT_INFLIGHT[ident] = ev

    if not mine:
        ev.wait(max(0.5, float(wait)))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest
        return ""

    try:
        tmp = dest + ".part"
        info = {"file_id": row.get("file_id") or "",
                "message_id": row.get("msg_id"),
                "file_size": int(row.get("bytes") or 0),
                "file_name": f"{_safe(video_key)}-{_safe(kind)}{ext}"}
        ok = False
        try:
            from . import tgchannel          # noqa: PLC0415 (cycle at import)
            ok = bool(tgchannel.fetch_document(info, tmp))
        except Exception as e:                              # noqa: BLE001
            log(f"artifact {ident} fetch failed — {type(e).__name__}: {e}")
        if ok and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            try:
                os.replace(tmp, dest)
                return dest
            except OSError:
                pass
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return ""
    finally:
        with _ARTIFACT_LOCK:
            _ARTIFACT_INFLIGHT.pop(ident, None)
        ev.set()


def poster(conn: sqlite3.Connection, video_key: str, at: float = None) -> str:
    """A still frame, cached on disk. Returns a path or "".

    `at` matters more than it looks. A result card can show the frame at the
    moment that actually matched instead of the first frame of the reel, which
    turns a grid of near-identical intro shots into a grid of answers. Frames
    are cached per (video, second), so scrubbing a ribbon does not re-run
    ffmpeg for a position already seen.

    A video that is not on disk used to return "" here, which is why a fresh
    Atlas showed a grid of grey rectangles until something had been watched.
    The clip index removes that: a two-second segment covering the moment is a
    small standalone mp4, so the frame can be cut from it without the reel
    ever being downloaded.

    Three sources, cheapest first
    ────────────────────────────
    For the cover frame — `at` unset or zero — the processing plane has already
    rendered exactly this image and now uploads it, so the first thing tried is
    a ~30 KB download of a finished JPEG. That beats both alternatives outright:
    no ffmpeg, no video bytes, and it is the frame the plane itself chose.

    A positional request skips it, because a poster artifact is one frame and it
    is not the one being asked for. Those fall through to cutting the frame from
    whatever file is here, and then to `clip_poster`.
    """
    key = str(video_key)
    pos = 0.0 if at is None else max(0.0, float(at))

    if pos <= 0.0:
        got = artifact_file(conn, key, "poster")
        if got:
            return got

    if not _FFMPEG:
        return ""
    found = resolve(conn, key)
    if found["where"] not in ("local", "cache"):
        return clip_poster(conn, key, pos)

    stamp = f"{pos:.0f}"
    dest = os.path.join(config.POSTER_CACHE, f"{_safe(key)}_{stamp}.jpg")
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
# CLIPS — the two-second segments that make playback instant
# ══════════════════════════════════════════════════════════════════════════
# A clip is a complete, standalone, keyframe-aligned mp4 of about two seconds,
# cut at capture time and stored as its own channel message. Two things follow,
# and both are the point of the whole mechanism:
#
#   1. Playing a *moment* costs one small download, not a media session over
#      a whole reel. A 2 s clip is ~200-600 KB, which is inside the Bot API's
#      20 MB `getFile` ceiling — so it comes down over plain HTTPS with no
#      MTProto session, no sparse file, and no window arithmetic.
#   2. A frame can be extracted from a video the server has never held. The
#      clip is a real mp4, so `ffmpeg` reads it directly and `poster()` can
#      answer "the frame at 47 s" for a reel that was never downloaded.
#
# The clip is not a substitute for the full file — seeking freely and watching
# to the end still wants the whole reel, which the sparse path fetches in the
# background. Clips are what remove the wait *before* the first frame.
_CLIP_LOCK = threading.RLock()
_CLIP_INFLIGHT: dict = {}


def clip_dir() -> str:
    d = os.path.join(config.VIDEO_CACHE, "_clips")
    os.makedirs(d, exist_ok=True)
    return d


def clip_path(video_key: str, seq: int) -> str:
    return os.path.join(clip_dir(), f"{_safe(video_key)}_{int(seq):04d}.mp4")


def clip_fetch(conn: sqlite3.Connection, video_key: str, t: float,
               wait: float = 20.0) -> dict:
    """Get the clip covering `t` onto disk. Returns {path, t0, t1, seq} or {}.

    Concurrency matters here in a way it does not for the full file: a grid of
    twenty search results all hovering fires twenty of these, and several will
    ask for the same clip. One in-flight download per clip, everyone else waits
    on it — the same bargain `ensure()` makes for whole videos.
    """
    from . import index as index_mod
    row = index_mod.clip_at(conn, video_key, t)
    if not row or not row.get("msg_id"):
        return {}

    seq = int(row.get("seq") or 0)
    dest = clip_path(video_key, seq)
    out = {"path": dest, "t0": float(row.get("t_start") or 0.0),
           "t1": float(row.get("t_end") or 0.0), "seq": seq,
           "msg_id": row.get("msg_id")}
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return out

    ident = f"{video_key}#{seq}"
    with _CLIP_LOCK:
        ev = _CLIP_INFLIGHT.get(ident)
        mine = ev is None
        if mine:
            ev = threading.Event()
            _CLIP_INFLIGHT[ident] = ev

    if not mine:
        ev.wait(max(0.5, float(wait)))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return out
        return {}

    try:
        tmp = dest + ".part"
        info = {"file_id": row.get("file_id") or "",
                "message_id": row.get("msg_id"),
                "file_size": int(row.get("bytes") or 0),
                "file_name": row.get("name") or ""}
        ok = False
        try:
            from . import tgchannel          # noqa: PLC0415 (cycle at import)
            ok = bool(tgchannel.fetch_document(info, tmp))
        except Exception as e:
            log(f"clip {ident} fetch failed — {type(e).__name__}: {e}")
        if ok and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, dest)
            return out
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return {}
    finally:
        with _CLIP_LOCK:
            _CLIP_INFLIGHT.pop(ident, None)
        ev.set()


def clip_poster(conn: sqlite3.Connection, video_key: str, at: float) -> str:
    """A still at `at` taken from the clip covering it. Path or "".

    This is what makes a thumbnail of the *matched moment* available for a reel
    the server has never downloaded, which is the difference between a search
    grid of answers and a search grid of intro frames.
    """
    if not _FFMPEG:
        return ""
    pos = max(0.0, float(at))
    dest = os.path.join(config.POSTER_CACHE,
                        f"{_safe(video_key)}_{pos:.0f}.jpg")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    got = clip_fetch(conn, video_key, pos)
    if not got:
        return ""
    # Offset *within* the clip. The clip starts at t0, so asking ffmpeg for
    # `pos` would seek past the end of a two-second file and produce nothing.
    inner = max(0.0, pos - float(got.get("t0") or 0.0))
    cmd = [_FFMPEG, "-nostdin", "-loglevel", "error", "-ss", f"{inner:.2f}",
           "-i", got["path"], "-frames:v", "1", "-vf", "scale=480:-2",
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


def _is_disconnect(exc: BaseException) -> bool:
    """Is this exception just the player having gone away?

    A scrub, a tab close, a player deciding it has buffered enough — all of them
    tear the socket down mid-range, and every layer below reports it with a
    different type. anyio raises `BrokenResourceError`, starlette raises
    `ClientDisconnect`, the OS raises `BrokenPipeError`/`ConnectionResetError`.
    None of them mean anything is wrong, and none of them are worth a traceback
    in a log that has real errors in it.

    Matched by name rather than by import so this stays true across the
    starlette/anyio versions Kaggle's image happens to pin, and so a missing
    package cannot turn the guard itself into the failure.
    """
    name = type(exc).__name__
    if name in ("ClientDisconnect", "BrokenResourceError", "ClosedResourceError",
                "EndOfStream", "BrokenPipeError", "ConnectionResetError"):
        return True
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    # anyio wraps concurrent failures in an ExceptionGroup; a disconnect inside
    # one is still a disconnect.
    inner = getattr(exc, "exceptions", None)
    if inner:
        return all(_is_disconnect(e) for e in inner)
    return False


def stream(path: str, start: int, end: int):
    """Yield one byte range. Closes the handle even if the client disconnects
    mid-play, which happens constantly — people scrub.

    The disconnect guard is deliberate. A `<video>` element abandons a range on
    every seek, and with the response half-written that used to surface as a
    traceback per scrub. The handle is closed by the `with` either way; this
    only stops a normal event from being reported as a fault.
    """
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        try:
            while remaining > 0:
                block = f.read(min(_CHUNK, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block
        except GeneratorExit:
            return
        except BaseException as exc:
            if _is_disconnect(exc):
                return
            raise


# ══════════════════════════════════════════════════════════════════════════
# STREAM-THROUGH — play now, cache on the way past
# ══════════════════════════════════════════════════════════════════════════
_MSG_CACHE = {}                  # video_key → (message, facts, at)
_MSG_TTL = 1800.0
_TG_CHUNK = 1024 * 1024          # pyrogram's stream granularity
# How much one streamed-from-Telegram response will serve. Eight chunks is
# several seconds of a reel — comfortably more than the player needs to keep
# buffered — and it bounds how long a single media session stays open.
REMOTE_WINDOW = 8 * _TG_CHUNK
_SPARSE = {}                     # video_key → set of chunk numbers held
_SPARSE_LOCK = threading.Lock()


def _sparse_index(video_key: str) -> set:
    with _SPARSE_LOCK:
        return _SPARSE.setdefault(str(video_key), set())


def sweep_sparse() -> int:
    """Delete half-built sparse files left by a previous run.

    Which chunks a sparse file holds is knowledge that lives in memory, and a
    hole reads back as zeros rather than as an error — so an index the process
    no longer has makes the file unsafe to trust and impossible to complete.
    Deleting them on boot costs a re-fetch of bytes nobody is watching yet.
    """
    gone = 0
    try:
        for name in os.listdir(config.VIDEO_CACHE):
            if not name.endswith(".sparse"):
                continue
            try:
                os.remove(os.path.join(config.VIDEO_CACHE, name))
                gone += 1
            except OSError:
                continue
    except OSError:
        pass
    with _SPARSE_LOCK:
        _SPARSE.clear()
    return gone


def _remember_chunk(part: str, index: set, chunk_no: int, piece: bytes,
                    size: int) -> None:
    """Write one 1 MiB chunk into the sparse file at its true offset.

    Seeking past the end of a file and writing produces a hole, which every
    filesystem Atlas runs on stores as nothing until filled. So watching the
    middle of a video costs the middle of a file, and the pieces converge on a
    complete copy in whatever order they are watched.

    Opened without truncation on purpose: two viewers on the same video would
    otherwise race, and the one that creates the file second would erase chunks
    the first has already recorded as held.
    """
    with _SPARSE_LOCK:
        if chunk_no in index:
            return
    fd = None
    try:
        fd = os.open(part, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0))
        os.lseek(fd, chunk_no * _TG_CHUNK, os.SEEK_SET)
        written = 0
        while written < len(piece):
            written += os.write(fd, piece[written:])
        with _SPARSE_LOCK:
            index.add(chunk_no)
    except OSError:
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _maybe_promote(video_key: str, part: str, index: set, size: int) -> None:
    """Turn a fully-populated sparse file into a real cache entry."""
    if size <= 0:
        return
    needed = (size + _TG_CHUNK - 1) // _TG_CHUNK
    with _SPARSE_LOCK:
        have = len(index)
    if have < needed:
        return
    try:
        if os.path.getsize(part) < size:
            with open(part, "r+b") as f:
                f.truncate(size)
        os.replace(part, _cache_path(str(video_key)))
        with _SPARSE_LOCK:
            _SPARSE.pop(str(video_key), None)
        invalidate_resident()
        log(f"video {video_key} fully cached from streaming")
        _maybe_evict()
    except OSError:
        pass


def sparse_hit(video_key: str, start: int, end: int) -> str:
    """The sparse file, if it already holds every chunk this range needs.

    Re-watching the first ten seconds of a video should not re-fetch them, and
    scrubbing backwards is the most common thing a person does in a player.

    The file's length is checked as well as the index, because a chunk recorded
    as held is only useful if the bytes are still there — an eviction between
    the two checks would otherwise serve a hole, which decodes as silence and a
    grey frame rather than as an error anybody could see.
    """
    part = _cache_path(str(video_key)) + ".sparse"
    index = _sparse_index(str(video_key))
    with _SPARSE_LOCK:
        held = set(index)
    if not held:
        return ""
    for c in range(start // _TG_CHUNK, end // _TG_CHUNK + 1):
        if c not in held:
            return ""
    try:
        if os.path.getsize(part) <= end:
            return ""
    except OSError:
        return ""
    return part


def _message_for(video_key: str, msg_id: int):
    """A message object and its media facts, memoised.

    Every range request the browser makes would otherwise cost a round trip to
    resolve the same message — and a seek storm makes a dozen of them. The
    channel is immutable, so caching this for half an hour is free.
    """
    key = str(video_key)
    hit = _MSG_CACHE.get(key)
    if hit and _now() - hit[2] < _MSG_TTL:
        return hit[0], hit[1]

    from . import tgchannel
    message = tgchannel.message_by_id(int(msg_id))
    if message is None:
        return None, {}
    facts = tgchannel.media_facts(message)
    if not facts.get("size"):
        return None, {}
    _MSG_CACHE[key] = (message, facts, _now())
    if len(_MSG_CACHE) > 512:
        for k in sorted(_MSG_CACHE, key=lambda k: _MSG_CACHE[k][2])[:128]:
            _MSG_CACHE.pop(k, None)
    return message, facts


def remote_plan(video_key: str, msg_id: int, range_header: str) -> dict:
    """A range plan for a video that is still in the channel.

    Same contract as `range_plan`, but the size comes off the Telegram message
    instead of a file on disk, so the browser can be told the real length and
    start playing without anything having been downloaded yet.

    One difference: a remote range is capped to `REMOTE_WINDOW`. Browsers open a
    video with `bytes=0-`, meaning "everything from here", and answering that
    literally holds one Telegram media session open for the entire file — each
    of which costs an auth handshake to build, and there are only a handful of
    permits. Returning a smaller range than asked for is explicitly allowed by
    the range spec and is what CDNs do; the player simply asks for the next
    window, by which time the background fill has usually put it on disk.
    """
    message, facts = _message_for(video_key, msg_id)
    if message is None:
        return {}
    size = facts["size"]
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": facts.get("mime") or "video/mp4",
        "Cache-Control": "public, max-age=604800, immutable",
    }
    m = _RANGE.match(range_header or "")
    if not m:
        # No range header at all: a 200 must carry the whole body, so this one
        # cannot be windowed. Only non-browser clients land here.
        headers["Content-Length"] = str(size)
        return {"status": 200, "start": 0, "end": size - 1, "size": size,
                "headers": headers, "message": message}

    raw_start, raw_end = m.group(1), m.group(2)
    if raw_start == "":
        length = int(raw_end or 0)
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    start = max(0, min(start, size - 1))
    end = max(start, min(end, size - 1))
    end = min(end, start + REMOTE_WINDOW - 1)

    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return {"status": 206, "start": start, "end": end, "size": size,
            "headers": headers, "message": message}


def stream_remote(video_key: str, message, start: int, end: int,
                  size: int):
    """Serve a byte range straight out of Telegram, caching whole chunks.

    Telegram only addresses media in 1 MiB chunks, so a request for byte
    1_500_000 starts at chunk 1 and the first 476 KiB of that chunk are
    trimmed. That is the whole trick behind seeking into a video nobody has
    downloaded: skip to the chunk that holds the byte, discard the prefix, and
    the browser has its frame in one round trip instead of thirty.

    Chunks that go past are written into a sparse side-file, so a video watched
    once is progressively assembled on disk. When every chunk has landed the
    file is promoted to the ordinary cache and later views never touch the
    network — the second watch is a local read.
    """
    from . import tgchannel

    key = str(video_key)
    first_chunk = start // _TG_CHUNK
    skip = start - first_chunk * _TG_CHUNK
    want = end - start + 1

    part = _cache_path(key) + ".sparse"
    index = _sparse_index(key)

    sent = 0
    chunk_no = first_chunk
    # The UI polls for progress, and "absent" would make it show a download bar
    # over a video that is already playing. Streaming is its own state.
    _set(key, status="streaming", got=start, total=size, note="")
    try:
        for piece in tgchannel.stream_chunks(message, first_chunk=first_chunk):
            _remember_chunk(part, index, chunk_no, piece, size)
            chunk_no += 1

            if skip:
                piece = piece[skip:]
                skip = 0
            if not piece:
                continue
            if sent + len(piece) >= want:
                yield piece[:want - sent]
                sent = want
                break
            yield piece
            sent += len(piece)
            _set(key, got=min(size, start + sent))
    except GeneratorExit:
        # The player seeked or closed. Everything already pulled from Telegram
        # is in the sparse file, so this is progress, not loss — fall through to
        # the `finally` and record the partial state quietly.
        pass
    except BaseException as exc:
        # Same for the socket-level forms of "gone away". Anything else is a
        # real failure and still raises: a Telegram error must not be filed as
        # a disconnect, or a video that can never be fetched would look like one
        # the user simply stopped watching.
        if not _is_disconnect(exc):
            raise
    finally:
        if sent >= want:
            _maybe_promote(key, part, index, size)
        with _LOCK:
            slot = _STATE.get(key)
            if slot and slot.get("status") == "streaming":
                slot["status"] = "ready" if sent >= want else "partial"


def stream_progress(video_key: str) -> dict:
    """How much of this video is already on disk in the sparse file.

    Reported so the interface can show a real "cached" fraction for a video
    being watched before it has finished arriving, rather than either nothing
    or a misleading download bar.
    """
    key = str(video_key)
    with _SPARSE_LOCK:
        held = len(_SPARSE.get(key) or ())
    if not held:
        return {}
    return {"chunks": held, "bytes": held * _TG_CHUNK}


# ══════════════════════════════════════════════════════════════════════════
# BACKGROUND FILL — one session per video instead of one per seek
# ══════════════════════════════════════════════════════════════════════════
_FILLING = set()
_FILL_SLOTS = threading.Semaphore(2)


def fill(video_key: str, msg_id: int, size: int) -> bool:
    """Quietly finish a video that is being watched. True if it was dispatched.

    Streaming a range on demand makes the first frame appear immediately, but
    every range costs a Telegram media session — pyrogram builds one per
    `get_file` call, complete with an auth handshake — and a person scrubbing a
    timeline generates a dozen ranges in as many seconds. Paying that per seek
    is what makes an otherwise-working player feel broken.

    So the first remote range also starts this: one sequential pass that writes
    every chunk into the same sparse file the player reads from. Within a few
    seconds the whole video is local, seeks stop touching the network entirely,
    and the file promotes itself into the ordinary cache. The watcher sees the
    video get faster while they watch it.

    Chunks the player already fetched are skipped on the write side, so the two
    never fight over the file — they cooperate on it.
    """
    key = str(video_key)
    if not msg_id or size <= 0:
        return False
    if os.path.exists(_cache_path(key)):
        return False
    with _LOCK:
        if key in _FILLING:
            return False
        _FILLING.add(key)

    def run():
        from . import tgchannel
        try:
            with _FILL_SLOTS:
                if os.path.exists(_cache_path(key)):
                    return
                message, facts = _message_for(key, msg_id)
                if message is None:
                    return
                total = facts.get("size") or size
                part = _cache_path(key) + ".sparse"
                index = _sparse_index(key)
                needed = (total + _TG_CHUNK - 1) // _TG_CHUNK
                with _SPARSE_LOCK:
                    start_at = min(
                        (c for c in range(needed) if c not in index),
                        default=needed)
                if start_at >= needed:
                    _maybe_promote(key, part, index, total)
                    return
                chunk_no = start_at
                for piece in tgchannel.stream_chunks(
                        message, first_chunk=start_at, queue_size=2):
                    _remember_chunk(part, index, chunk_no, piece, total)
                    chunk_no += 1
                _maybe_promote(key, part, index, total)
        except Exception as exc:                           # noqa: BLE001
            log(f"fill {key} stopped — {type(exc).__name__}: "
                f"{str(exc)[:120]}", "WARN")
        finally:
            with _LOCK:
                _FILLING.discard(key)

    threading.Thread(target=run, name=f"atlas-fill-{key}", daemon=True).start()
    return True
