"""
vios.capture.upload — put the bytes somewhere that outlives everything else.

Telegram is the permanent store. Not a cache, not a mirror: the place the
original mp4 and its metadata live for the rest of the project's life. Kaggle
sessions die, laptops get reformatted, R2's free tier is 10 GB. This channel
is none of those things, and it already holds 552 reels.

What goes into the channel, per reel
────────────────────────────────────
Two messages, deliberately:

  1. `sendVideo` with the original file and a human-readable caption. Sent as
     a *video*, not a document, for three reasons that all matter downstream:
     Telegram generates and stores a thumbnail with it (a few KB, fetchable
     over the plain Bot API with no MTProto session — this is what makes the
     grid of posters instant instead of 24 video downloads), it records
     duration/width/height in the message, and it plays inline in the app when
     the user is browsing the channel by hand.

  2. `sendDocument` with the `.vios.json` record, sent as a *reply* to the
     video message. Reply threading is what binds them: the processing plane
     can walk from either one to the other with no external index.

The caption keeps the old Colab script's `🔗 <b>Link:</b> …` line verbatim.
That is not nostalgia — it is the parse anchor `vios.capture.seed` uses to
rebuild the ledger from the channel, and it has to match across both eras or
the 552 existing reels cannot be adopted.

Transport
─────────
The HTTP Bot API, for the same reason `tg_transport.py` chose it: every call
has an explicit deadline and a bounded retry, so a call either finishes or
raises and never parks. MTProto is used for one thing only — files over the
Bot API's 50 MB upload cap — and it is constructed, used and torn down inside
that one call rather than kept alive.
"""

from __future__ import annotations

import json
import os
import time

API_ROOT = "https://api.telegram.org"

BOT_UPLOAD_LIMIT = 50 * 1024 * 1024
CAPTION_LIMIT = 1024

CONNECT_TIMEOUT = 20.0
READ_TIMEOUT = 180.0
WRITE_TIMEOUT = 900.0
ATTEMPTS = 4
_BACKOFF = (3, 9, 20)


class UploadError(RuntimeError):
    pass


def _fmt(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _esc(text: str) -> str:
    """HTML-escape for Telegram's `parse_mode=HTML`.

    Captions carry user text: a reel description containing `<3` or `&` breaks
    the whole message with a 400 if it is not escaped, and that failure looks
    like a rate limit at the call site.
    """
    return (str(text or "").replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def build_caption(record: dict, collections=()) -> str:
    """The channel caption. Human-readable first, machine-parseable always.

    Trimmed from the description end rather than the metadata end: if
    something has to go it should be the tail of a long caption, never the
    permalink, because the permalink is what makes the message re-identifiable
    if every database is lost.
    """
    post = record.get("post", {}) or {}
    eng = record.get("engagement", {}) or {}
    cols = ", ".join(sorted({c for c in collections if c})) or "uncategorised"

    tail = (
        f"\n🔗 <b>Link:</b> {_esc(record.get('url', ''))}\n"
        f"<code>vios:{_esc(record.get('key', ''))}</code>"
    )
    head = (
        f"📁 <b>Category:</b> {_esc(cols)[:300]}\n"
        f"👤 <b>Creator:</b> {_esc(post.get('uploader') or 'unknown')[:120]}\n"
        f"👁️ <b>Views:</b> {_fmt(eng.get('views'))} | "
        f"❤️ <b>Likes:</b> {_fmt(eng.get('likes'))} | "
        f"💬 <b>Comments:</b> {_fmt(eng.get('comments'))}\n"
    )

    # Telegram counts caption length in UTF-16 code units, so each emoji here
    # costs two where Python sees one. The reserve covers that, the `<i>` and
    # `<b>` wrappers, and leaves the link line untouchable.
    reserve = 64
    room = CAPTION_LIMIT - len(head) - len(tail) - reserve
    raw = (post.get("description") or "").strip().replace("\n", " ")
    body = ""
    if room > 40 and raw:
        # Trim the *unescaped* text and escape afterwards. Cutting the escaped
        # string is the subtle way to break this: a slice landing inside `&amp;`
        # leaves `&am`, Telegram answers 400, and `call()` treats a 400 as
        # non-retryable — so one ampersand in the wrong column would cost the
        # whole reel.
        if len(raw) > room:
            cut = raw[:room - 1]
            # Prefer a word boundary, but not at any price: a caption ending in
            # a long unbroken token — a URL, a run of joined hashtags — has its
            # last space near the start, and backing up to it would throw away
            # most of what fits.
            spaced = cut.rsplit(" ", 1)[0]
            raw = (spaced if len(spaced) > room * 0.6 else cut) + "…"
        desc = _esc(raw)
        while len(desc) > room and raw:
            raw = raw[:max(1, len(raw) - (len(desc) - room) - 1)]
            desc = _esc(raw) + "…"
        body = f"📝 <b>Caption:</b> <i>{desc}</i>\n"
    return head + body + tail


class Telegram:
    """A bot-token-scoped client. Credentials are passed in, never imported.

    The whole point of the admin tab is that the token is typed at runtime and
    exists only in memory, so this class must not reach for a module-level
    constant — that is exactly how the previous token ended up in a public
    repository.
    """

    def __init__(self, bot_token: str, channel_id, api_id: int = 0,
                 api_hash: str = ""):
        self.token = (bot_token or "").strip()
        self.channel = channel_id
        self.api_id = int(api_id or 0)
        self.api_hash = (api_hash or "").strip()
        if not self.token:
            raise UploadError("No bot token.")
        if not self.channel:
            raise UploadError("No channel id.")

    # ── plumbing ─────────────────────────────────────────────────────────
    def _url(self, method: str) -> str:
        return f"{API_ROOT}/bot{self.token}/{method}"

    def _client(self):
        import httpx
        return httpx.Client(
            timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT,
                                  write=WRITE_TIMEOUT, pool=CONNECT_TIMEOUT),
            follow_redirects=True)

    def call(self, method: str, data: dict | None = None,
             file_factory=None, attempts: int = ATTEMPTS) -> dict:
        import httpx
        last = None
        for attempt in range(attempts):
            handles = []
            try:
                files = None
                if file_factory is not None:
                    files, handles = file_factory()
                with self._client() as c:
                    r = c.post(self._url(method), data=data or {}, files=files)

                if r.status_code == 429:
                    wait = 5
                    try:
                        wait = int(r.json().get("parameters", {})
                                   .get("retry_after", 5))
                    except Exception:
                        pass
                    time.sleep(min(wait, 120) + 1)
                    last = UploadError(f"{method}: rate limited")
                    continue

                try:
                    body = r.json()
                except Exception:
                    raise UploadError(
                        f"{method}: HTTP {r.status_code}, non-JSON reply")

                if body.get("ok"):
                    return body.get("result", {})

                desc = body.get("description", "no description")
                if r.status_code >= 500:
                    last = UploadError(f"{method}: {desc}")
                else:
                    # 4xx is us being wrong — a missing admin right, a bad
                    # channel id, a malformed caption. Retrying cannot fix it
                    # and the description is the actionable part.
                    raise UploadError(f"{method}: {desc}")
            except UploadError:
                raise
            except (httpx.HTTPError, OSError) as e:
                last = UploadError(f"{method}: {type(e).__name__}: {str(e)[:200]}")
            finally:
                for h in handles:
                    try:
                        h.close()
                    except OSError:
                        pass
            if attempt < attempts - 1:
                time.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
        raise last or UploadError(f"{method}: failed after {attempts} attempts")

    # ── readiness ────────────────────────────────────────────────────────
    def probe(self) -> dict:
        """Can we talk to the API, and can we post to the channel?

        Run before a week-long unattended job, not discovered at reel one. The
        send/delete pair is the only way to prove *write* access — `getChat`
        succeeds for a bot that can read the channel but cannot post to it.
        """
        out = {"ok": False, "bot": "", "channel": "", "can_post": False,
               "error": ""}
        try:
            me = self.call("getMe", attempts=2)
            out["bot"] = me.get("username") or me.get("first_name") or ""
        except UploadError as e:
            out["error"] = str(e)
            return out
        try:
            chat = self.call("getChat", {"chat_id": self.channel}, attempts=2)
            out["channel"] = chat.get("title") or str(self.channel)
        except UploadError as e:
            out["error"] = (f"Bot @{out['bot']} cannot see channel "
                            f"{self.channel}: {str(e)[:200]}")
            return out
        try:
            probe = self.call("sendMessage",
                              {"chat_id": self.channel,
                               "text": "· vios capture check",
                               "disable_notification": True}, attempts=2)
            out["can_post"] = True
            try:
                self.call("deleteMessage",
                          {"chat_id": self.channel,
                           "message_id": probe["message_id"]}, attempts=1)
            except UploadError:
                pass  # a leftover dot is harmless; failing the run is not
            out["ok"] = True
        except UploadError as e:
            out["error"] = (f"Bot @{out['bot']} can see the channel but cannot "
                            f"post: {str(e)[:200]}. Make it an admin with "
                            f"'Post messages'.")
        return out

    # ── sending ──────────────────────────────────────────────────────────
    def send_video(self, path: str, caption: str, progress=None) -> dict:
        size = os.path.getsize(path)
        if size > BOT_UPLOAD_LIMIT:
            return self._send_video_mtproto(path, caption)

        def _factory():
            handle = _ProgressFile(path, progress)
            return ({"video": (os.path.basename(path), handle, "video/mp4")},
                    [handle])

        res = self.call("sendVideo", {
            "chat_id": self.channel,
            "caption": caption,
            "parse_mode": "HTML",
            # Telegram only builds the streaming index and the thumbnail when
            # it is told the file is streamable. Both are things this project
            # depends on later, so this flag is load-bearing.
            "supports_streaming": True,
            "disable_notification": True,
        }, file_factory=_factory)
        return _video_result(res)

    def send_document(self, path: str, caption: str = "",
                      reply_to: int | None = None,
                      file_name: str | None = None) -> dict:
        name = file_name or os.path.basename(path)

        def _factory():
            handle = open(path, "rb")
            return ({"document": (name, handle, "application/json")}, [handle])

        data = {"chat_id": self.channel, "caption": caption[:CAPTION_LIMIT],
                "disable_notification": True}
        if reply_to:
            data["reply_to_message_id"] = reply_to
            # If the video message was deleted by hand, the reply would fail
            # and take the record with it. The record is worth more than the
            # threading.
            data["allow_sending_without_reply"] = True
        res = self.call("sendDocument", data, file_factory=_factory)
        doc = res.get("document") or {}
        return {"message_id": int(res.get("message_id", 0)),
                "file_id": doc.get("file_id", "")}

    def send_message(self, text: str) -> int:
        res = self.call("sendMessage",
                        {"chat_id": self.channel, "text": text[:4096],
                         "parse_mode": "HTML", "disable_notification": True})
        return int(res.get("message_id", 0))

    def pin(self, message_id: int) -> bool:
        try:
            self.call("pinChatMessage",
                      {"chat_id": self.channel, "message_id": message_id,
                       "disable_notification": True}, attempts=2)
            return True
        except UploadError:
            return False

    def download(self, file_id: str, dest: str) -> bool:
        """getFile then GET. Used to pull the newest ledger snapshot back."""
        import httpx
        meta = self.call("getFile", {"file_id": file_id}, attempts=2)
        file_path = meta.get("file_path")
        if not file_path:
            return False
        url = f"{API_ROOT}/file/bot{self.token}/{file_path}"
        tmp = dest + ".part"
        try:
            with self._client() as c, c.stream("GET", url) as r:
                if r.status_code != 200:
                    return False
                with open(tmp, "wb") as f:
                    for block in r.iter_bytes(1024 * 1024):
                        f.write(block)
            os.replace(tmp, dest)
            return True
        except (httpx.HTTPError, OSError):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return False

    # ── the oversize path ────────────────────────────────────────────────
    def _send_video_mtproto(self, path: str, caption: str) -> dict:
        """For files over the Bot API's 50 MB cap.

        Rare — a vertical reel is 2–8 MB and even a three-minute one rarely
        clears 50 — so this is built to be correct rather than fast, and the
        client is created and torn down inside the call. Running it under
        `asyncio.run` is safe because the capture worker is a plain thread
        with no event loop of its own.
        """
        if not (self.api_id and self.api_hash):
            raise UploadError(
                f"{os.path.basename(path)} is "
                f"{os.path.getsize(path) / 1048576:.0f} MB, over the Bot API's "
                f"50 MB limit. Add the API id and API hash in the admin tab to "
                f"upload files this large.")
        try:
            import asyncio
            from pyrogram import Client
            from vios.tgcompat import patch as _tgpatch
        except ImportError:
            raise UploadError("pyrogram is not installed; cannot upload a file "
                              "over 50 MB.")
        _tgpatch()   # see vios/tgcompat.py — modern channel ids, old library

        async def _go():
            app = Client("vios_capture_big", api_id=self.api_id,
                         api_hash=self.api_hash, bot_token=self.token,
                         in_memory=True, no_updates=True)
            await app.start()
            try:
                msg = await app.send_video(
                    self.channel, path, caption=caption[:CAPTION_LIMIT],
                    parse_mode=__import__("pyrogram").enums.ParseMode.HTML,
                    supports_streaming=True, disable_notification=True)
                video = getattr(msg, "video", None)
                return {"message_id": msg.id,
                        "file_id": getattr(video, "file_id", "") or "",
                        "duration": getattr(video, "duration", None),
                        "width": getattr(video, "width", None),
                        "height": getattr(video, "height", None),
                        "thumb_file_id": ""}
            finally:
                await app.stop()

        try:
            return asyncio.run(_go())
        except Exception as e:
            raise UploadError(f"MTProto upload failed: {type(e).__name__}: "
                              f"{str(e)[:200]}")


def _video_result(res: dict) -> dict:
    video = res.get("video") or res.get("document") or {}
    thumb = video.get("thumbnail") or video.get("thumb") or {}
    return {
        "message_id": int(res.get("message_id", 0)),
        "file_id": video.get("file_id", ""),
        "file_unique_id": video.get("file_unique_id", ""),
        "file_size": video.get("file_size"),
        "duration": video.get("duration"),
        "width": video.get("width"),
        "height": video.get("height"),
        # A few KB, and the reason a grid of posters can be instant without
        # touching a single video file. `atlas.tgchannel` never read this.
        "thumb_file_id": thumb.get("file_id", ""),
    }


class _ProgressFile:
    """File wrapper reporting bytes read, so the UI shows a real transfer.

    httpx sizes a multipart field with seek/tell and then read()s in a loop,
    so hooking read() gives byte-accurate progress for free.
    """

    def __init__(self, path: str, cb=None):
        self._f = open(path, "rb")
        self._cb = cb
        self._sent = 0
        self.total = os.path.getsize(path)
        self.name = os.path.basename(path)

    def read(self, size=-1):
        block = self._f.read(size)
        if block:
            self._sent += len(block)
            if self._cb:
                try:
                    self._cb(self._sent, self.total)
                except Exception:
                    pass
        return block

    def seek(self, offset, whence=0):
        if offset == 0 and whence == 0:
            self._sent = 0
        return self._f.seek(offset, whence)

    def tell(self):
        return self._f.tell()

    def close(self):
        try:
            self._f.close()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# The one call the engine makes
# ═══════════════════════════════════════════════════════════════════════
def publish(tg: Telegram, result: dict, collections=(),
            progress=None) -> dict:
    """Upload one captured reel: video, then record, then extra images.

    Order matters. The video goes first because it is the irreplaceable
    artifact — if the process dies between the two messages, the bytes are
    safe and the record can be regenerated from the temp directory or, in the
    worst case, re-derived. The reverse order would leave metadata pointing at
    nothing.
    """
    record = result["record"]
    caption = build_caption(record, collections)

    if result.get("video"):
        sent = tg.send_video(result["video"], caption, progress=progress)
    else:
        # A photo-only post. Still worth keeping: caption and comments carry
        # signal, and a gap in the ledger is worse than a post with no video.
        first = result["images"][0] if result.get("images") else None
        if not first:
            raise UploadError("nothing to upload")
        sent = tg.send_document(first, caption)
        sent = {"message_id": sent["message_id"], "file_id": sent["file_id"],
                "thumb_file_id": "", "duration": None,
                "width": None, "height": None}

    rec = tg.send_document(result["record_path"],
                           caption=f"metadata · vios:{record.get('key','')}",
                           reply_to=sent["message_id"])

    return {
        "msg_id": sent["message_id"],
        "record_msg_id": rec["message_id"],
        "file_id": sent.get("file_id", ""),
        "thumb_file_id": sent.get("thumb_file_id", ""),
        "duration": sent.get("duration"),
        "width": sent.get("width"),
        "height": sent.get("height"),
    }


def upload_snapshot(tg: Telegram, path: str, note: str = "") -> int:
    """Push the ledger to the channel and pin it.

    This is what makes a Kaggle session disposable. The pin is how `restore`
    finds the newest one in a single API call — the Bot API has no way to read
    channel history, but `getChat` hands back the pinned message.
    """
    size = os.path.getsize(path) / 1048576
    caption = (f"📒 vios capture ledger · {time.strftime('%Y-%m-%d %H:%M')} · "
               f"{size:.1f} MB\n{note}")[:CAPTION_LIMIT]
    res = tg.send_document(path, caption=caption,
                           file_name="vios_capture_ledger.db")
    tg.pin(res["message_id"])
    return res["message_id"]


def restore_snapshot(tg: Telegram, dest: str) -> bool:
    """Pull the newest pinned ledger back, if there is one."""
    try:
        chat = tg.call("getChat", {"chat_id": tg.channel}, attempts=2)
    except UploadError:
        return False
    pinned = chat.get("pinned_message") or {}
    doc = pinned.get("document") or {}
    name = (doc.get("file_name") or "").lower()
    if "ledger" not in name or not doc.get("file_id"):
        return False
    return tg.download(doc["file_id"], dest)
