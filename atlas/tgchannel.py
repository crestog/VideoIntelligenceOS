"""
Channel access.

Two transports, chosen per operation rather than per program:

  Bot API (HTTP)   getChat for the pinned manifest, and getFile for anything
                   under 20 MB. One request, no session file, no login. This is
                   the fast path Atlas uses on a cold start.
  MTProto (pyrogram)  everything the Bot API cannot do: reading an arbitrary
                   message by id, and downloading a file over 20 MB. Both are
                   required — the whole-channel scan is a walk over message ids,
                   and reels routinely exceed the HTTP download cap.

The one thing neither can do is list history. `messages.getHistory` is a
user-only method and returns BOT_METHOD_INVALID for a bot account, so there is
no "give me the last N messages" call available to us at all. The scan works
around it the way the harvester does: find the newest id by posting a message
and reading back its own id, then walk backwards asking for ids in batches.
Ids are dense enough in a channel that this is cheap, and it is the only method
a bot is actually permitted to use.

Pyrogram is async and Atlas's callers are not, so the client lives on its own
event loop in its own thread and every call out of here is a synchronous
wrapper with a deadline. That is deliberate: the bug that made the exporter
hang forever was an await with no timeout, and none of this code can repeat it.
"""

import asyncio
import json
import os
import re
import threading
import time

import requests

from . import config

_LOG = []
_LOG_LOCK = threading.Lock()


def log(msg: str, level: str = "INFO") -> None:
    line = f"{time.strftime('%H:%M:%S')} · {level} · {msg}"
    with _LOG_LOCK:
        _LOG.append(line)
        del _LOG[:-200]
    # Logging must never be able to fail a request. A Windows console runs
    # cp1252, so a print carrying the satellite glyph raises UnicodeEncodeError
    # — and this is called from inside the playback generator, where an
    # exception would abort a working video stream over a cosmetic character.
    try:
        print(f"📡 [ATLAS] {msg}", flush=True)
    except Exception:                                  # noqa: BLE001
        try:
            print(f"[ATLAS] {msg}".encode("ascii", "replace").decode("ascii"),
                  flush=True)
        except Exception:                              # noqa: BLE001
            pass


def recent_log(limit: int = 200) -> list:
    with _LOG_LOCK:
        return list(_LOG)[-int(limit):] if limit else list(_LOG)


# ══════════════════════════════════════════════════════════════════════════
# BOT API — the no-session fast path
# ══════════════════════════════════════════════════════════════════════════
_HTTP_TIMEOUT = (10, 60)          # (connect, read)


def _api(method: str) -> str:
    return f"https://api.telegram.org/bot{config.BOT_TOKEN}/{method}"


def _call(method: str, params: dict = None, timeout=_HTTP_TIMEOUT) -> dict:
    """One Bot API call. Raises on transport failure, returns the result dict.

    Telegram answers 200 with ok=false for logical errors, so the status code
    alone never tells you whether it worked.
    """
    r = requests.get(_api(method), params=params or {}, timeout=timeout)
    try:
        body = r.json()
    except ValueError:
        raise RuntimeError(f"{method}: non-JSON reply ({r.status_code})")
    if not body.get("ok"):
        raise RuntimeError(f"{method}: {body.get('description', 'failed')}")
    return body.get("result")


def probe() -> dict:
    """Can we reach the channel at all? Answered in one round trip.

    This is the first thing the UI calls, because every other failure in Atlas
    is downstream of it and a missing token otherwise shows up as an empty
    library that looks like an empty channel.
    """
    missing = config.missing_secrets()
    if not config.BOT_TOKEN:
        return {"ok": False, "missing": missing,
                "error": "No bot token. Atlas cannot read the channel."}
    try:
        me = _call("getMe", timeout=(5, 15))
        chat = _call("getChat", {"chat_id": config.CHANNEL_ID}, timeout=(5, 20))
        pinned = (chat or {}).get("pinned_message")
        return {
            "ok": True,
            "missing": missing,
            "bot": me.get("username"),
            "channel": chat.get("title") or str(config.CHANNEL_ID),
            "channel_id": config.CHANNEL_ID,
            "pinned_message_id": (pinned or {}).get("message_id"),
            "mtproto": bool(config.API_ID and config.API_HASH),
        }
    except Exception as exc:
        return {"ok": False, "missing": missing, "error": str(exc)[:240]}


def pinned_message() -> dict:
    """The channel's pinned message, or None. The exporter pins each manifest,
    so this is the newest bundle in a single call — no scan needed to start."""
    try:
        chat = _call("getChat", {"chat_id": config.CHANNEL_ID})
        return (chat or {}).get("pinned_message")
    except Exception as exc:
        log(f"getChat failed: {exc}", "WARN")
        return None


def http_download(file_id: str, dest: str) -> bool:
    """Download by file_id over HTTP. False when the file is too big for the
    Bot API's 20 MB getFile cap, which is the caller's cue to use MTProto."""
    try:
        info = _call("getFile", {"file_id": file_id})
    except Exception as exc:
        # Not a fault, and it must not read like one. Every caller treats False
        # as "use MTProto next", and for a video over the 20 MB ceiling that is
        # the expected path — it succeeded on every occurrence of the last run,
        # while the log said "getFile failed" 60 times and sent the reader
        # looking for a download problem that did not exist.
        cap = config.HTTP_DOWNLOAD_LIMIT // 1048576
        if "too big" in str(exc).lower():
            log(f"over the Bot API's {cap} MB cap — fetching it over MTProto "
                f"instead")
        else:
            log(f"getFile unavailable ({exc}) — falling back to MTProto", "WARN")
        return False
    size = info.get("file_size") or 0
    if size > config.HTTP_DOWNLOAD_LIMIT:
        return False
    url = (f"https://api.telegram.org/file/bot{config.BOT_TOKEN}/"
           f"{info['file_path']}")
    tmp = dest + ".part"
    try:
        with requests.get(url, stream=True, timeout=(10, 300)) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, dest)
        return True
    except Exception as exc:
        log(f"download failed: {exc}", "WARN")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


# ══════════════════════════════════════════════════════════════════════════
# MTPROTO — arbitrary message reads and large downloads
# ══════════════════════════════════════════════════════════════════════════
def _raw(client, name: str):
    """The genuine coroutine function behind a pyrogram method.

    Pyrogram ships `sync.py`, which at import time rewrites every async method
    on Client into a synchronous wrapper via `functools.wraps`. Whether that
    wrapper returns a coroutine, a finished result, or a future depends on
    which thread you are on and whether a loop is already running — so calling
    `client.get_messages(...)` from Atlas's worker threads returned a *result*
    where this module expected an awaitable, and handing that to
    `run_coroutine_threadsafe` raised "An asyncio.Future, a coroutine or an
    awaitable is required".

    `functools.wraps` leaves the original on `__wrapped__`, which is the async
    method, unbound. Calling it with the client as the first argument gives a
    real coroutine every time, on any thread, with no dependence on ambient
    loop state. Falls back to the attribute itself for a pyrogram build that
    does not apply the sync layer.
    """
    method = getattr(client, name)
    inner = getattr(method, "__wrapped__", None)
    if inner is None:
        return method
    return lambda *a, **kw: inner(client, *a, **kw)


class _Mtproto:
    """A pyrogram client pinned to a private event loop on a private thread.

    Callers are synchronous FastAPI handlers and background workers. Rather
    than dragging an async runtime through the whole program (or reaching for
    nest_asyncio, which makes reentrancy bugs look like hangs), the client owns
    one loop and every entry point here is `submit(coro, timeout)`.
    """

    def __init__(self):
        self._loop = None
        self._thread = None
        self._client = None
        self._ready = threading.Event()
        self._error = None
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────
    def available(self) -> bool:
        return bool(config.API_ID and config.API_HASH and config.BOT_TOKEN)

    def start(self) -> bool:
        """Idempotent. Returns False if MTProto cannot be used at all."""
        if not self.available():
            self._error = ("MTProto needs TELEGRAM_API_ID and "
                           "TELEGRAM_API_HASH as well as the bot token.")
            return False
        with self._lock:
            if self._ready.is_set():
                return self._client is not None
            if self._thread is None:
                self._thread = threading.Thread(target=self._run,
                                                name="atlas-mtproto",
                                                daemon=True)
                self._thread.start()
        # 120 s: a cold pyrogram start negotiates an auth key and can be slow
        # on a fresh container, but it must not be unbounded.
        self._ready.wait(timeout=120)
        return self._client is not None

    def _run(self):
        try:
            from pyrogram import Client
        except ImportError:
            self._error = "pyrogram is not installed"
            self._ready.set()
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _start():
            try:
                client = Client(
                    "atlas_reader",
                    api_id=config.API_ID, api_hash=config.API_HASH,
                    bot_token=config.BOT_TOKEN,
                    workdir=config.SESSION_DIR,
                    no_updates=True,
                    in_memory=False,
                    # Pyrogram defaults this to 1, which puts every file
                    # transfer behind a single semaphore. With streaming
                    # playback that means a prefetch warming somebody else's
                    # thumbnail stalls the video actually on screen. Four lets
                    # the player, a warm and a background download coexist
                    # without inviting the flood limits a wide pool trips.
                    max_concurrent_transmissions=4)
                await _raw(client, "start")()
                self._client = client
                log("MTProto session ready")
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {str(exc)[:200]}"
                log(f"MTProto unavailable — {self._error}", "WARN")
            finally:
                self._ready.set()

        self._loop.run_until_complete(_start())
        if self._client is not None:
            self._loop.run_forever()

    def submit(self, coro, timeout: float):
        """Run a coroutine on the client's loop and wait, with a deadline."""
        if not self.start():
            raise RuntimeError(self._error or "MTProto unavailable")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            fut.cancel()
            raise

    @property
    def client(self):
        return self._client

    @property
    def loop(self):
        return self._loop

    @property
    def error(self):
        return self._error


_mt = _Mtproto()


def mtproto_ready() -> bool:
    return _mt.start()


def mtproto_error() -> str:
    return _mt.error or ""


async def _with_floodwait(make_coro, tries: int = 4):
    """Retry a call across FloodWait, which the channel WILL throw during a
    full scan. Anything else propagates — a retry loop that swallows real
    errors is how a scan silently returns half a channel."""
    from pyrogram.errors import FloodWait
    for attempt in range(tries):
        try:
            return await make_coro()
        except FloodWait as e:
            wait = int(getattr(e, "value", getattr(e, "x", 5))) + 1
            if attempt == tries - 1:
                raise
            log(f"FloodWait {wait}s — pausing the scan", "WARN")
            await asyncio.sleep(wait)
    return None


def newest_message_id() -> int:
    """The highest message id in the channel.

    A bot cannot ask. It can, however, post — and the id it gets back is the
    current head. So: post a dot, keep the number, delete the dot. The channel
    sees a message that exists for well under a second, which is the price of
    an API that has no getHistory for bots.
    """
    try:
        sent = _call("sendMessage", {"chat_id": config.CHANNEL_ID,
                                     "text": ".",
                                     "disable_notification": "true"})
        mid = sent["message_id"]
        try:
            _call("deleteMessage", {"chat_id": config.CHANNEL_ID,
                                    "message_id": mid})
        except Exception:
            # A failed delete leaves a dot in the channel. Harmless, and much
            # less bad than failing the scan over it.
            pass
        return int(mid)
    except Exception as exc:
        log(f"could not determine newest message id: {exc}", "ERROR")
        return 0


def get_messages(ids: list, timeout: float = 120) -> list:
    """Read messages by id over MTProto. Missing ids come back as None and are
    dropped — deleted messages and service events leave gaps everywhere."""
    if not ids:
        return []
    client = None

    async def _go():
        nonlocal client
        client = _mt.client
        return await _with_floodwait(
            lambda: _raw(client, "get_messages")(config.CHANNEL_ID, ids))

    out = _mt.submit(_go(), timeout=timeout)
    if out is None:
        return []
    if not isinstance(out, list):
        out = [out]
    return [m for m in out if m is not None and not getattr(m, "empty", False)]


def download_message(message, dest: str, progress=None,
                     timeout: float = 900) -> bool:
    """Download a message's media over MTProto. Handles files of any size."""
    async def _go():
        return await _with_floodwait(
            lambda: _raw(_mt.client, "download_media")(
                message, file_name=dest,
                progress=(lambda c, t: progress(c, t)) if progress else None))

    try:
        got = _mt.submit(_go(), timeout=timeout)
        return bool(got) and os.path.exists(dest)
    except Exception as exc:
        log(f"MTProto download failed: {str(exc)[:160]}", "WARN")
        return False


def download_by_id(message_id: int, dest: str, progress=None) -> bool:
    """Fetch one message's media straight to `dest`, whatever its size."""
    msgs = get_messages([message_id])
    if not msgs:
        return False
    return download_message(msgs[0], dest, progress=progress)


# ══════════════════════════════════════════════════════════════════════════
# STREAMING — bytes out of Telegram without waiting for the whole file
# ══════════════════════════════════════════════════════════════════════════
CHUNK = 1024 * 1024          # pyrogram's fixed stream granularity
STREAM_STALL = 90.0          # give up on a transfer that has gone quiet


def stream_chunks(message, first_chunk: int = 0, chunk_limit: int = 0,
                  queue_size: int = 4):
    """Yield 1 MiB pieces of a message's media, starting at `first_chunk`.

    This is what makes a video playable before it has finished arriving.
    `download_media` has to complete before it returns anything, so a 30 MB
    reel means half a minute of nothing; `stream_media` hands back pieces as
    they land, and Telegram lets us skip straight to the chunk containing the
    byte the browser asked for — so seeking to the middle of a video does not
    fetch the beginning.

    Pyrogram is async and the caller is a synchronous response generator, so the
    async generator runs on the MTProto loop and pushes into a bounded queue.
    Bounded matters: without it a fast connection buffers the whole file into
    RAM, which is exactly what this is meant to avoid. The queue applies
    backpressure, so Atlas reads from Telegram at roughly the speed the browser
    drains it.

    Every hand-off — pieces, the end marker, an error — goes through `_offer`,
    which waits by awaiting rather than by blocking. A blocking `put` on the
    loop thread would freeze the entire MTProto client the moment a viewer
    paused a video with a full queue, and nothing could cancel it because a
    thread stuck in C code has no await point to interrupt.
    """
    import queue as _queue

    if not _mt.start():
        raise RuntimeError(_mt.error or "MTProto unavailable")

    q = _queue.Queue(maxsize=queue_size)
    DONE = object()

    async def _offer(item):
        while True:
            try:
                q.put_nowait(item)
                return
            except _queue.Full:
                await asyncio.sleep(0.02)

    async def _pump():
        agen = None
        try:
            agen = _raw(_mt.client, "stream_media")(
                message, offset=first_chunk, limit=chunk_limit)
            async for piece in agen:
                await _offer(piece)
            await _offer(DONE)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                       # noqa: BLE001
            await _offer(exc)
        finally:
            # Release Telegram's side of the transfer straight away rather than
            # leaving it to garbage collection, which on a paused video could be
            # a long time holding a media session open.
            if agen is not None:
                try:
                    await agen.aclose()
                except Exception:                      # noqa: BLE001
                    pass

    fut = asyncio.run_coroutine_threadsafe(_pump(), _mt.loop)
    try:
        while True:
            try:
                item = q.get(timeout=STREAM_STALL)
            except _queue.Empty:
                # Nothing for a long time and the pump is gone: the loop died or
                # the coroutine was cancelled out from under us. A plain
                # blocking get would hold this worker thread forever.
                if fut.done():
                    return
                raise RuntimeError("Telegram stopped sending — no data for "
                                   f"{STREAM_STALL:.0f}s")
            if item is DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # The browser closing the tab mid-video lands here. Cancelling stops
        # pulling bytes we are throwing away.
        fut.cancel()


def message_by_id(message_id: int):
    """One message object, or None. Used by the streaming player."""
    msgs = get_messages([message_id])
    return msgs[0] if msgs else None


def media_facts(message) -> dict:
    """Size and type of a message's media, without fetching any of it.

    A 206 response has to state the file's total length in `Content-Range`
    before a single byte is sent, so this is read off the message itself.
    """
    info = message_document(message)
    return {
        "size": int(info.get("file_size") or 0),
        "mime": info.get("mime") or "video/mp4",
        "name": info.get("file_name") or "",
        "duration": info.get("duration"),
    }


# ══════════════════════════════════════════════════════════════════════════
# MESSAGE SHAPES
# ══════════════════════════════════════════════════════════════════════════
def message_document(msg) -> dict:
    """Normalise a message down to what Atlas cares about.

    Handles both shapes, because both transports are in use and they disagree
    on everything: pyrogram hands back objects with attributes, `.id` and a
    `datetime`, while the Bot API hands back plain dicts with `message_id` and
    a Unix int. The pinned-manifest fast path comes from the Bot API and the
    channel walk comes from pyrogram, so a function that only understood one of
    them would make the fast path silently find no bundles at all.
    """
    if msg is None:
        return {}

    if isinstance(msg, dict):
        media = msg.get("document") or msg.get("video") or {}
        vid = msg.get("video") or {}
        return {
            "message_id": msg.get("message_id") or msg.get("id"),
            "date": msg.get("date"),
            "caption": msg.get("caption") or "",
            "file_name": media.get("file_name"),
            "file_id": media.get("file_id"),
            "file_size": media.get("file_size"),
            "mime": media.get("mime_type"),
            "is_video": bool(vid),
            "duration": vid.get("duration") or None,
            "width": vid.get("width") or None,
            "height": vid.get("height") or None,
        }

    doc = getattr(msg, "document", None)
    vid = getattr(msg, "video", None)
    media = doc or vid
    when = getattr(msg, "date", None)
    return {
        "message_id": getattr(msg, "id", None) or getattr(msg, "message_id",
                                                          None),
        "date": (when.timestamp() if hasattr(when, "timestamp") else when),
        "caption": getattr(msg, "caption", None) or "",
        "file_name": getattr(media, "file_name", None) if media else None,
        "file_id": getattr(media, "file_id", None) if media else None,
        "file_size": getattr(media, "file_size", None) if media else None,
        "mime": getattr(media, "mime_type", None) if media else None,
        "is_video": vid is not None,
        "duration": getattr(vid, "duration", None) if vid else None,
        "width": getattr(vid, "width", None) if vid else None,
        "height": getattr(vid, "height", None) if vid else None,
    }


def looks_like_manifest(info: dict) -> bool:
    """Is this message a bundle manifest?

    Two independent signals, because either one alone has a failure mode: the
    file name is what the exporter writes today, and the caption marker is what
    survives if a future exporter renames the file. Matching either keeps Atlas
    reading bundles it did not write.
    """
    name = (info.get("file_name") or "").lower()
    cap = info.get("caption") or ""
    if name.startswith("manifest-") and name.endswith(".json"):
        return True
    if "VIOS bundle" in cap and "manifest" in cap.lower():
        return True
    return "✅ VIOS bundle" in cap


SHARD_PREFIX = "vios-evidence-"
SHARD_SUFFIX = ".jsonl.gz"


def looks_like_shard(info: dict) -> bool:
    """Is this message an evidence shard from the GPU plane?

    Two independent lanes post to this channel and they share no format. The
    harvester posts *bundles*: a manifest naming the parts of a SQLite
    snapshot. The process engine posts *shards*: one gzipped JSONL file per
    batch of claims, complete on its own, with no manifest anywhere and nothing
    pinned. A reader that only knew manifests walked straight past every claim
    the GPU ever produced — which is exactly what Atlas did.

    Same two signals as a manifest, for the same reason: the file name is what
    the engine writes today, the caption marker is what survives a rename.
    """
    name = (info.get("file_name") or "").lower()
    if name.startswith(SHARD_PREFIX) and name.endswith(SHARD_SUFFIX):
        return True
    cap = (info.get("caption") or "").strip().lower()
    return cap.startswith("vios evidence")


def shard_seq(info: dict) -> str:
    """The shard's own id — `<site>-<seq>` — from its name, caption, or failing
    both, its message id.

    The site prefix is the load-bearing half. Ten Kaggle accounts each number
    their shards from one, so `0007` alone names ten different files; keyed on
    that, importing worker 3's seventh shard would mark worker 7's as held.
    """
    name = (info.get("file_name") or "").strip()
    if name.lower().startswith(SHARD_PREFIX) and \
            name.lower().endswith(SHARD_SUFFIX):
        got = name[len(SHARD_PREFIX):-len(SHARD_SUFFIX)].strip()
        if got:
            return got
    cap = (info.get("caption") or "").strip()
    m = re.match(r"vios\s+evidence\s*[·:•-]\s*(\S+)", cap, re.I)
    if m:
        return m.group(1)
    return f"msg{info.get('message_id')}"


def fetch_document(info: dict, dest: str) -> bool:
    """Pull a message's file to `dest`, cheap transport first.

    HTTP needs no session and no login, so it is tried whenever the message
    carried a `file_id`; it returns False rather than raising when the file is
    over the Bot API's 20 MB ceiling, and MTProto picks that up.
    """
    ok = False
    if info.get("file_id"):
        ok = http_download(info["file_id"], dest)
    if not ok and info.get("message_id"):
        ok = download_by_id(info["message_id"], dest)
    return bool(ok and os.path.exists(dest))


def read_manifest_document(info: dict, work_dir: str) -> dict:
    """Download a manifest message and parse it. Manifests are a few KB, so the
    HTTP path always fits and MTProto is only the fallback."""
    dest = os.path.join(work_dir, f"manifest-{info['message_id']}.json")
    if not fetch_document(info, dest):
        return None
    try:
        with open(dest, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        log(f"manifest {info['message_id']} is not readable JSON: {exc}", "WARN")
        return None
    if not isinstance(data, dict) or "parts" not in data:
        return None
    data["_message_id"] = info["message_id"]
    data["_date"] = info.get("date")
    return data
