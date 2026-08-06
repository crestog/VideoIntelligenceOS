"""
vios.process.intake — where the videos come from, and how they get here.

The processing plane owns no bytes. Every original mp4 lives in the Telegram
channel, put there by the capture plane, and the evidence store holds a
`video_key` and a message id. So before any pass can run, three things have to
happen, and this module is all three:

  sync      the capture ledger's finished rows become video rows in the
            evidence store — the list of what there is to process
  Source    one video's mp4 and its capture record land in a working directory
  restore   a fresh Kaggle session pulls back every evidence shard the channel
            already holds, so processing resumes instead of restarting

Two transports, and the choice between them is not stylistic. The Bot API can
download a file by `file_id` and is the simpler path, but it refuses anything
over 20 MB and it cannot fetch a message by id at all — which rules it out for
the record documents, whose file ids were never recorded for the 552 reels the
old Colab script uploaded. MTProto can do both. So MTProto is the primary path
and the Bot API is the fallback, which is the reverse of the capture plane's
arrangement and correct for the same underlying reason: capture writes small
files and knows their ids, processing reads arbitrary files and knows only
where they sit in the channel.

The MTProto session is opened once and held for the length of a sweep. Five
thousand videos through a per-download handshake would spend more time
connecting than transferring, and would look far more like a script than a
client.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

# 20 MB — the Bot API's download ceiling. Reels are 2–8 MB, so this path
# carries most of the archive; the ones it refuses are the three-minute ones.
BOT_DOWNLOAD_LIMIT = 20 * 1024 * 1024

SHARD_PREFIX = "vios-evidence-"
SHARD_SUFFIX = ".jsonl.gz"


class SourceError(RuntimeError):
    """The bytes could not be obtained. Not the video's fault, not a pass's."""


# ══════════════════════════════════════════════════════════════════════════
# The capture ledger → the evidence store
# ══════════════════════════════════════════════════════════════════════════

def sync(store, ledger_path: str, limit: int = 0) -> dict:
    """Register every captured reel as a video row. Idempotent.

    Opened read-only, over URI, because the capture engine may be running in
    this same process a week into its own sweep. A read-only connection cannot
    take the write lock and therefore cannot stall it.

    What travels into `video.meta` is a *head*, not the capture record: the
    pointers needed to fetch the real record later plus the denormalised
    engagement figures the interface sorts by. The record itself is tens of
    kilobytes of JSON per reel and belongs in the working directory, not in a
    database that gets uploaded.
    """
    out = {"seen": 0, "added": 0, "ledger": ledger_path}
    if not os.path.exists(ledger_path):
        out["reason"] = "no capture ledger yet"
        return out

    try:
        conn = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True,
                               timeout=15.0)
    except sqlite3.Error as exc:
        out["reason"] = f"cannot open the capture ledger: {exc}"
        return out
    conn.row_factory = sqlite3.Row

    try:
        collections: dict = {}
        for r in conn.execute("SELECT key, collection FROM membership"):
            collections.setdefault(r["key"], []).append(r["collection"])

        sql = ("SELECT * FROM item WHERE state='uploaded' AND msg_id IS NOT NULL"
               " ORDER BY done_at")
        if limit:
            sql += f" LIMIT {int(limit)}"
        known = set(store.video_keys())
        for r in conn.execute(sql):
            out["seen"] += 1
            key = r["key"]
            head = {
                "msg_id": r["msg_id"],
                "record_msg_id": r["record_msg_id"],
                "file_id": r["file_id"],
                "collections": sorted(collections.get(key, [])),
                "title": r["title"],
                "views": r["views"],
                "likes": r["likes"],
                "comment_count": r["comment_count"],
                "comments_got": r["comments_got"],
                "captured_at": r["done_at"],
            }
            store.add_video(
                key, url=r["url"], uploader=r["uploader"],
                duration=r["duration"], width=r["width"], height=r["height"],
                bytes=r["file_size"], sha256=r["sha256"], msg_id=r["msg_id"],
                taken_at=r["taken_at"], meta={"capture": head})
            if key not in known:
                out["added"] += 1
    except sqlite3.Error as exc:
        out["reason"] = f"capture ledger unreadable: {exc}"
    finally:
        conn.close()
    return out


def capture_head(video: dict) -> dict:
    """The pointers `sync` stored, back out of the video row."""
    meta = video.get("meta")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except json.JSONDecodeError:
            return {}
    if not isinstance(meta, dict):
        return {}
    head = meta.get("capture")
    return head if isinstance(head, dict) else {}


# ══════════════════════════════════════════════════════════════════════════
# A live MTProto session
# ══════════════════════════════════════════════════════════════════════════

class Channel:
    """One pyrogram client, running on its own event loop, in its own thread.

    The processing worker is a plain thread with no loop of its own, and
    `asyncio.run` per call — which is what the capture plane does for its
    handful of oversize uploads — would reconnect for every download. Here the
    loop is started once and coroutines are posted to it from the worker with
    `run_coroutine_threadsafe`, so the session survives the whole sweep.

    Every method returns a falsy value rather than raising when the transport
    is simply absent. A session without pyrogram installed, or without an API
    id, is a session that uses the Bot API — not a session that fails.
    """

    def __init__(self, tg, log=None):
        self.tg = tg
        self.log = log or (lambda m: None)
        self.ready = False
        self.reason = ""
        self.last_error = ""
        self._loop = None
        self._thread = None
        self._app = None
        self._lock = threading.RLock()

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> bool:
        with self._lock:
            if self.ready:
                return True
            if not (self.tg and self.tg.token and self.tg.api_id
                    and self.tg.api_hash):
                self.reason = ("no API id and hash — large files and capture "
                               "records need MTProto")
                return False
            try:
                import asyncio  # noqa: PLC0415
                from pyrogram import Client  # noqa: PLC0415
                from vios.tgcompat import patch as _tgpatch  # noqa: PLC0415
            except ImportError:
                self.reason = "pyrogram is not installed"
                return False

            # Must happen before the first call that names the channel.
            # Pyrogram rejects channel ids past 2**31 outright, and a channel
            # created recently has one — which surfaces here as every download
            # failing while the session itself reports healthy. See
            # vios/tgcompat.py for the whole story.
            note = _tgpatch()
            if note:
                self.log(note)

            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, name="vios-process-mtproto",
                daemon=True)
            self._thread.start()

            async def _boot():
                # Constructed inside the loop thread on purpose: some pyrogram
                # builds capture the running loop at __init__ time, and one
                # built on the worker's thread would post its callbacks
                # somewhere nothing is listening.
                app = Client(
                    "vios_process", api_id=int(self.tg.api_id),
                    api_hash=self.tg.api_hash, bot_token=self.tg.token,
                    in_memory=True, no_updates=True,
                    max_concurrent_transmissions=2)
                await app.start()
                return app

            try:
                self._app = self._submit(_boot(), timeout=180)
                self.ready = True
                self.reason = ""
                self.log("MTProto session open")
            except Exception as exc:
                self.reason = f"{type(exc).__name__}: {str(exc)[:160]}"
                self._shutdown_loop()
            return self.ready

    def stop(self) -> None:
        with self._lock:
            if self._app is not None:
                try:
                    self._submit(self._app.stop(), timeout=60)
                except Exception:
                    pass
                self._app = None
            self._shutdown_loop()
            self.ready = False

    def _shutdown_loop(self) -> None:
        loop, thread = self._loop, self._thread
        self._loop = self._thread = None
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        if thread and thread.is_alive():
            thread.join(timeout=10)
        try:
            loop.close()
        except Exception:
            pass

    def _submit(self, coro, timeout: float):
        import asyncio  # noqa: PLC0415
        if self._loop is None:
            raise SourceError("MTProto session is not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ── reading ──────────────────────────────────────────────────────────
    def messages(self, ids: list, timeout: float = 120.0) -> dict:
        """message id → message, for the ids that still exist."""
        if not self.ready or not ids:
            return {}
        want = [int(i) for i in ids if i]
        if not want:
            return {}

        async def _go():
            import asyncio  # noqa: PLC0415
            from pyrogram.errors import FloodWait  # noqa: PLC0415
            for attempt in range(4):
                try:
                    out = await self._app.get_messages(self.tg.channel, want)
                    if out is None:
                        return []
                    return out if isinstance(out, list) else [out]
                except FloodWait as e:
                    wait = int(getattr(e, "value", getattr(e, "x", 5))) + 1
                    self.log(f"Telegram asked for a {wait}s pause")
                    await asyncio.sleep(min(wait, 120))
            return []

        try:
            msgs = self._submit(_go(), timeout=timeout)
        except Exception as exc:
            # Kept, not just logged: `Source.ensure` raises the error the user
            # actually sees, and "could not download the original" with no
            # cause attached is what made this class of failure unreadable.
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            self.log(f"message fetch failed: {self.last_error}")
            return {}
        return {int(m.id): m for m in msgs
                if m is not None and not getattr(m, "empty", False)}

    def download(self, msg, dest: str, timeout: float = 900.0) -> bool:
        """Pull one message's media to an absolute path."""
        if not self.ready or msg is None:
            return False
        dest = os.path.abspath(dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"

        async def _go():
            import asyncio  # noqa: PLC0415
            from pyrogram.errors import FloodWait  # noqa: PLC0415
            for attempt in range(3):
                try:
                    return await self._app.download_media(msg, file_name=tmp)
                except FloodWait as e:
                    wait = int(getattr(e, "value", getattr(e, "x", 5))) + 1
                    await asyncio.sleep(min(wait, 120))
            return None

        try:
            got = self._submit(_go(), timeout=timeout)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            self.log(f"download failed: {self.last_error}")
            got = None
        if not got or not os.path.exists(got):
            for stray in (tmp, dest + ".part"):
                if os.path.exists(stray):
                    try:
                        os.remove(stray)
                    except OSError:
                        pass
            return False
        try:
            os.replace(got, dest)
        except OSError:
            return False
        return True


# ══════════════════════════════════════════════════════════════════════════
# One video onto local disk
# ══════════════════════════════════════════════════════════════════════════

class Source:
    """Materialises a video's mp4 and capture record in a working directory.

    `ensure` is the only method the engine calls, and it is deliberately
    forgiving about the record: a reel whose metadata document has gone missing
    is still a reel worth processing, and the caption pass will say so itself.
    Missing *bytes* are fatal for that video and nothing else.
    """

    def __init__(self, tg=None, channel: Channel | None = None, log=None):
        self.tg = tg
        self.channel = channel
        self.log = log or (lambda m: None)
        self.downloaded = 0
        self.bytes = 0
        self.reused = 0

    def ensure(self, video: dict, workdir: str) -> str:
        os.makedirs(workdir, exist_ok=True)
        dest = os.path.join(workdir, "source.mp4")
        record = os.path.join(workdir, "record.json")
        head = capture_head(video)

        have_source = os.path.exists(dest) and os.path.getsize(dest) > 4096
        have_record = os.path.exists(record) and os.path.getsize(record) > 2
        if have_source and have_record:
            self.reused += 1
            return dest

        msg_id = video.get("msg_id") or head.get("msg_id")
        record_id = head.get("record_msg_id")
        file_id = head.get("file_id") or ""
        size = int(video.get("bytes") or 0)

        # MTProto first: it can fetch both messages in one round trip, and it
        # is the only transport that can reach the record document at all.
        if (not have_source or not have_record) and self.channel and self.channel.ready:
            wanted = [i for i in (msg_id if not have_source else None,
                                  record_id if not have_record else None) if i]
            msgs = self.channel.messages(wanted) if wanted else {}
            if not have_source and msg_id and int(msg_id) in msgs:
                if self.channel.download(msgs[int(msg_id)], dest):
                    have_source = True
            if not have_record and record_id and int(record_id) in msgs:
                if self.channel.download(msgs[int(record_id)], record):
                    have_record = True

        # Bot API fallback for the bytes. Nothing here can reach the record.
        if not have_source and self.tg and file_id:
            if size and size > BOT_DOWNLOAD_LIMIT:
                self.log(f"{video.get('video_key')}: {size / 1048576:.0f} MB is "
                         f"over the Bot API limit and MTProto is unavailable")
            else:
                try:
                    have_source = bool(self.tg.download(file_id, dest))
                except Exception as exc:
                    self.log(f"bot download failed: {type(exc).__name__}: "
                             f"{str(exc)[:120]}")

        if not have_source:
            raise SourceError(self._why(msg_id, file_id, size))

        got = os.path.getsize(dest)
        if got < 4096:
            try:
                os.remove(dest)
            except OSError:
                pass
            raise SourceError(f"downloaded file is {got} bytes — not a video")
        self.downloaded += 1
        self.bytes += got
        return dest

    def _why(self, msg_id, file_id, size: int = 0) -> str:
        """The reason this video's bytes could not be had, in the user's terms.

        Written as a full diagnosis rather than a status line because this
        string is the only thing the operator sees, and the previous version
        ("could not download the original from Telegram (message 38)") named
        the symptom while hiding all four things that could have caused it.
        """
        if not (msg_id or file_id):
            return ("this video has no Telegram message id and no file id — "
                    "the capture ledger row is incomplete, so there is nothing "
                    "to fetch. Re-run the capture tab's channel scan to repair "
                    "it.")

        err = getattr(self.channel, "last_error", "") if self.channel else ""
        if err:
            return (f"Telegram refused the fetch for message {msg_id or '?'}: "
                    f"{err}")

        if self.channel is None or not self.channel.ready:
            why = (getattr(self.channel, "reason", "") if self.channel
                   else "no MTProto session was opened")
            over = (size and size > BOT_DOWNLOAD_LIMIT)
            return (f"message {msg_id or '?'} needs MTProto and it is not "
                    f"available ({why or 'reason unknown'})."
                    + (f" The file is {size / 1048576:.0f} MB, over the Bot "
                       f"API's 20 MB download limit, so there is no fallback."
                       if over else
                       " Add the API id and API hash on the Setup page."))

        return (f"message {msg_id or '?'} is gone from the channel — the "
                f"session can read the channel, but that message id returned "
                f"nothing. It was deleted, or the channel id now points "
                f"somewhere else than it did at capture time.")


# ══════════════════════════════════════════════════════════════════════════
# Shards: publishing, and getting them back
# ══════════════════════════════════════════════════════════════════════════

def site_id(store) -> str:
    """A short id for this database, minted once and kept forever.

    Ten Kaggle accounts each produce shards into one channel. Shard ids have to
    be unique across all of them and stable across a restore, and claim id
    ranges are per-database — worker 3's rows 1..900 are not worker 7's rows
    1..900. Prefixing with a per-database id is what keeps `import_shard` from
    silently treating one worker's shard as another's.
    """
    sid = store.get_meta("site_id", "")
    if not sid:
        import uuid  # noqa: PLC0415
        sid = uuid.uuid4().hex[:8]
        store.set_meta("site_id", sid)
    return sid


def shard_name(shard_id: str) -> str:
    return f"{SHARD_PREFIX}{shard_id}{SHARD_SUFFIX}"


def shard_id_from_name(name: str) -> str:
    name = (name or "").strip()
    if not (name.startswith(SHARD_PREFIX) and name.endswith(SHARD_SUFFIX)):
        return ""
    return name[len(SHARD_PREFIX):-len(SHARD_SUFFIX)]


def restore_shards(store, tg, channel: Channel, on_progress=None,
                   should_stop=None, batch: int = 200) -> dict:
    """Replay every evidence shard in the channel that this database lacks.

    The first thing a fresh session should do. Ten accounts push shards; any
    one of them can pull all of them and end up with the union, because
    `import_shard` is idempotent by uid and order-independent. A shard already
    named in the local `shard` table is not downloaded again — which is what
    makes running this at every startup cheap rather than a full re-import.
    """
    out = {"found": 0, "imported": 0, "skipped": 0, "claims": 0, "vectors": 0,
           "errors": []}
    if not (channel and channel.ready):
        out["reason"] = channel.reason if channel else "no MTProto session"
        return out

    from ..capture.seed import head_message_id  # noqa: PLC0415
    try:
        head = head_message_id(tg)
    except Exception as exc:
        out["reason"] = f"could not read the channel head: {exc}"
        return out
    if head <= 0:
        out["reason"] = "could not read the channel head"
        return out

    have = {r["shard_id"] for r in store.shards()}
    tmpdir = os.path.join(os.path.dirname(os.path.abspath(store.path)),
                          "_shards")
    os.makedirs(tmpdir, exist_ok=True)

    for lo in range(1, head + 1, batch):
        if should_stop and should_stop():
            break
        ids = list(range(lo, min(lo + batch, head + 1)))
        for msg in channel.messages(ids).values():
            doc = getattr(msg, "document", None)
            sid = shard_id_from_name(getattr(doc, "file_name", "") if doc else "")
            if not sid:
                continue
            out["found"] += 1
            if sid in have:
                out["skipped"] += 1
                continue
            path = os.path.join(tmpdir, shard_name(sid))
            if not channel.download(msg, path):
                out["errors"].append(f"{sid}: download failed")
                continue
            try:
                counts = store.import_shard(path)
                out["imported"] += 1
                out["claims"] += counts.get("claim", 0)
                out["vectors"] += counts.get("vector", 0)
                store.note_shard(sid, "restored", int(msg.id),
                                 {"claims": counts.get("claim", 0),
                                  "vectors": counts.get("vector", 0),
                                  "bytes": os.path.getsize(path)})
                have.add(sid)
            except Exception as exc:
                out["errors"].append(f"{sid}: {type(exc).__name__}: "
                                     f"{str(exc)[:120]}")
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        if on_progress:
            try:
                on_progress(min(lo + batch - 1, head), head, out["imported"])
            except Exception:
                pass
    return out


def free_mb(path: str) -> int:
    import shutil  # noqa: PLC0415
    try:
        return shutil.disk_usage(path).free // (1024 ** 2)
    except OSError:
        return 0


def dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def evict(cache_dir: str, budget_mb: int, keep: set | None = None,
          floor_mb: int = 0) -> dict:
    """Delete the least recently touched working directories over budget.

    The cache is an optimisation with teeth: a video whose artifacts survive
    from one cohort to the next saves a download and about four seconds of
    ffmpeg. But 57 GB of Kaggle scratch divided by five thousand reels is not
    much, so this runs after every video and the eviction order is oldest
    first by mtime.

    Nothing here is load-bearing for correctness. Every file it deletes can be
    rebuilt from the original, and the engine rebuilds them on demand.
    """
    keep = keep or set()
    out = {"removed": 0, "freed_mb": 0}
    if not os.path.isdir(cache_dir):
        return out
    entries = []
    for name in os.listdir(cache_dir):
        p = os.path.join(cache_dir, name)
        if not os.path.isdir(p) or name in keep:
            continue
        try:
            entries.append((os.path.getmtime(p), p, dir_bytes(p)))
        except OSError:
            continue
    total_mb = sum(e[2] for e in entries) // (1024 ** 2)
    entries.sort()

    import shutil  # noqa: PLC0415
    for _mtime, path, nbytes in entries:
        over_budget = budget_mb and total_mb > budget_mb
        low_disk = floor_mb and free_mb(cache_dir) < floor_mb
        if not (over_budget or low_disk):
            break
        try:
            shutil.rmtree(path, ignore_errors=True)
            out["removed"] += 1
            out["freed_mb"] += nbytes // (1024 ** 2)
            total_mb -= nbytes // (1024 ** 2)
        except OSError:
            continue
    return out


def touch(path: str) -> None:
    """Mark a working directory as recently used, for the eviction order."""
    try:
        os.utime(path, (time.time(), time.time()))
    except OSError:
        pass
