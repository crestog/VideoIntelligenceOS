"""
tg_transport.py — bounded, synchronous Telegram transport over the HTTP Bot API.

Why this exists
───────────────
The bundle uploader used to go through pyrogram (MTProto) and it hung on Kaggle:
the panel sat at "Uploading part 1/2 · 72%" indefinitely, with no error, no
progress and no way to stop it. Reading the old code explains every part of
that display:

  * 72% was not measured. `_upload_bundle` set `pct = 55 + int(35 * n / total)`
    once per part, so with two parts the first one *is* 72% — a constant
    written before the upload began, not a byte count. The bar could not have
    moved even if bytes were flowing.
  * The job ran in a plain thread and nothing in the path — not the pyrogram
    client, not `send_document`, not the surrounding `run_until_complete` —
    carried a timeout. A media upload that stops making progress therefore
    blocks that thread forever, and `_job` keeps reporting `state: "running"`.
  * There was no cancel path, so a wedged export could only be cleared by
    restarting the notebook.

So the observable defect is: an unbounded blocking call under a progress
indicator that was never wired to real progress. This module replaces the
transport with one where *every* call has an explicit deadline and a bounded
retry count — a call either finishes or raises, never parks — and reports true
byte counts so the bar reflects the transfer.

Choosing HTTPS over MTProto for this is also what the rest of the codebase
already does under duress: when the harvester's MTProto `send_message` fails to
resolve the channel, `ui_server.background_downloader` falls back to a plain
POST to api.telegram.org to get unstuck. The Bot API is the path that has been
demonstrated to work from these containers.

What the Bot API costs us, and why it is still the right default
───────────────────────────────────────────────────────────────
  * Upload cap: 50 MB per document (MTProto allows 2 GB).
  * Download cap: 20 MB per `getFile` (MTProto has none).
  * No way to fetch an arbitrary message by id — Bot API is push-oriented.

The first two are bought off by shipping smaller parts: PART_SIZE drops to
18 MB so every part is both uploadable *and* downloadable over plain HTTPS.
The third is bought off by recording each part's `file_id` in the manifest at
upload time, so restore never needs to look a message up — it goes straight
from manifest to `getFile`. The manifest itself is found via `getChat`, whose
`pinned_message` the export pins for exactly this purpose.

Every call here has an explicit connect/read/write timeout, a bounded retry
count, and honours Telegram's 429 `retry_after`. The guarantee this module
makes, and the one MTProto could not, is that **a call either finishes or
raises** — it never parks forever.

MTProto remains available in db_export/db_restore as a fallback for parts that
exceed the Bot API caps (older bundles built with 480 MB parts), but it is now
always wrapped in a hard timeout.
"""

import os
import time

# `import config`, not `from config import BOT_TOKEN`. The second form binds the
# value into this module at import time and Python never asks again, which made a
# credential that arrived late — after a throttled Phase 0, or typed into the
# Setup page — permanently invisible to every upload, download and restore in
# this process. `config.BOT_TOKEN` is a lookup; see the note in config.py.
import config
from logger import vios_log

API_ROOT = "https://api.telegram.org"

# Bot API hard limits. Not tunable — these are Telegram's, not ours.
UPLOAD_LIMIT = 50 * 1024 * 1024
DOWNLOAD_LIMIT = 20 * 1024 * 1024

# Per-request deadlines. Generous on read/write because an 18 MB part over a
# shared Kaggle uplink is genuinely slow; finite because the whole point of
# this module is that nothing waits forever.
CONNECT_TIMEOUT = 20.0
READ_TIMEOUT = 180.0
WRITE_TIMEOUT = 600.0

ATTEMPTS = 4               # per call, before giving up
_BACKOFF = (2, 6, 15)      # seconds between attempts
_CHUNK = 1024 * 1024


class TelegramError(RuntimeError):
    """A Bot API call failed in a way retrying did not fix."""


class Cancelled(RuntimeError):
    """The caller's progress callback asked to stop. Not an error condition —
    the export/restore job raises this on itself when the user hits Cancel."""


def available() -> bool:
    return bool(config.BOT_TOKEN)


def _url(method: str) -> str:
    return f"{API_ROOT}/bot{config.BOT_TOKEN}/{method}"


class _ProgressFile:
    """File wrapper that reports bytes read and can abort mid-upload.

    httpx streams a multipart file field by calling seek/tell to size it and
    then read() in a loop, so hooking read() gives byte-accurate progress for
    free. `cb` returning False cancels: raising from inside read() unwinds the
    request rather than letting a cancelled job keep pushing bytes.
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
            if self._cb and self._cb(self._sent, self.total) is False:
                raise Cancelled("upload cancelled")
        return block

    def seek(self, offset, whence=0):
        if offset == 0 and whence == 0:
            self._sent = 0          # httpx rewinds before streaming for real
        return self._f.seek(offset, whence)

    def tell(self):
        return self._f.tell()

    def close(self):
        try:
            self._f.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _client():
    import httpx
    return httpx.Client(
        timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT,
                              write=WRITE_TIMEOUT, pool=CONNECT_TIMEOUT),
        follow_redirects=True,
    )


def _post(method: str, data: dict | None = None, file_factory=None,
          attempts: int = ATTEMPTS) -> dict:
    """One Bot API call, retried on transport errors and 429s.

    Returns the `result` field. Raises TelegramError with Telegram's own
    description on a 4xx that is not a rate limit — those do not get better
    with retrying, and surfacing the description is the difference between
    "export failed" and "the bot is not an admin of the channel".

    Files arrive as a *factory* returning (files_dict, handles_to_close)
    rather than as an open handle: a progress-wrapped file is single-use, so a
    retry has to reopen it or httpx streams from a spent descriptor.
    """
    if not config.BOT_TOKEN:
        raise TelegramError("No bot token configured (VIOS_BOT_TOKEN).")

    import httpx
    last = None
    for attempt in range(attempts):
        handles = []
        try:
            payload = None
            if file_factory is not None:
                payload, handles = file_factory()
            with _client() as c:
                r = c.post(_url(method), data=data or {}, files=payload)

            if r.status_code == 429:
                wait = 5
                try:
                    wait = int(r.json().get("parameters", {})
                               .get("retry_after", 5))
                except Exception:
                    pass
                vios_log(f"Telegram rate limit on {method} — waiting {wait}s",
                         "TG", "WARN")
                time.sleep(min(wait, 60) + 1)
                last = TelegramError(f"{method}: rate limited")
                continue

            try:
                body = r.json()
            except Exception:
                raise TelegramError(
                    f"{method}: HTTP {r.status_code}, non-JSON reply "
                    f"({r.text[:120]})")

            if body.get("ok"):
                return body.get("result", {})

            desc = body.get("description", "no description")
            # 5xx is Telegram having a bad minute; 4xx is us being wrong.
            if r.status_code >= 500:
                last = TelegramError(f"{method}: {desc}")
            else:
                raise TelegramError(f"{method}: {desc}")

        except Cancelled:
            raise
        except TelegramError:
            raise
        except (httpx.HTTPError, OSError) as e:
            last = TelegramError(f"{method}: {type(e).__name__}: {str(e)[:160]}")
        finally:
            for h in handles:
                h.close()

        if attempt < attempts - 1:
            time.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])

    raise last or TelegramError(f"{method}: failed after {attempts} attempts")


# ═══════════════════════════════════════════════════════════
# MESSAGES
# ═══════════════════════════════════════════════════════════
def send_message(text: str, silent: bool = True) -> int:
    res = _post("sendMessage", {"chat_id": config.CHANNEL_ID, "text": text,
                                "disable_notification": silent})
    return int(res["message_id"])


def delete_message(message_id: int) -> bool:
    try:
        _post("deleteMessage", {"chat_id": config.CHANNEL_ID,
                                "message_id": message_id}, attempts=1)
        return True
    except TelegramError:
        return False


def send_document(path: str, caption: str = "", file_name: str | None = None,
                  progress=None) -> dict:
    """Upload one file. Returns {message_id, file_id, file_unique_id}.

    `file_id` is the reason this returns a dict rather than an id: it is what
    lets restore fetch the part later without the message-lookup the Bot API
    does not offer. It is stable for the lifetime of the bot token, and export
    and restore share one token.
    """
    size = os.path.getsize(path)
    if size > UPLOAD_LIMIT:
        raise TelegramError(
            f"{os.path.basename(path)} is {size / 1048576:.0f} MB; the Bot API "
            f"caps uploads at {UPLOAD_LIMIT // 1048576} MB.")

    name = file_name or os.path.basename(path)

    def _factory():
        handle = _ProgressFile(path, progress)
        return {"document": (name, handle, "application/octet-stream")}, [handle]

    # No parse_mode: part names carry underscores and dots, and Markdown would
    # reject or mangle them. A caption is a label, not a document.
    res = _post(
        "sendDocument",
        {"chat_id": config.CHANNEL_ID, "caption": caption[:1024],
         "disable_notification": True},
        file_factory=_factory,
    )
    doc = res.get("document") or {}
    return {"message_id": int(res["message_id"]),
            "file_id": doc.get("file_id", ""),
            "file_unique_id": doc.get("file_unique_id", "")}


def pin_message(message_id: int) -> bool:
    """Pin, so restore finds the newest manifest in one call. Needs admin
    rights; a False here is survivable (restore falls back to a history scan)
    so this reports rather than raises."""
    try:
        _post("pinChatMessage", {"chat_id": config.CHANNEL_ID,
                                 "message_id": message_id,
                                 "disable_notification": True}, attempts=2)
        return True
    except TelegramError as e:
        vios_log(f"pin failed: {str(e)[:140]}", "TG", "WARN")
        return False


def get_pinned() -> dict | None:
    """The channel's pinned message, or None. This is how restore finds the
    manifest without MTProto — getChat is one of the few Bot API methods that
    hands back a message the bot did not just receive."""
    try:
        chat = _post("getChat", {"chat_id": config.CHANNEL_ID}, attempts=2)
    except TelegramError as e:
        vios_log(f"getChat failed: {str(e)[:140]}", "TG", "WARN")
        return None
    return chat.get("pinned_message")


# ═══════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════
def download_file(file_id: str, dest: str, progress=None) -> None:
    """getFile → stream the result to `dest`.

    Two calls by design: getFile resolves a short-lived path under /file/bot…,
    and the download itself is an ordinary GET. Written to `.part` and renamed
    so an interrupted transfer never leaves a file that looks complete.
    """
    import httpx

    meta = _post("getFile", {"file_id": file_id})
    file_path = meta.get("file_path")
    if not file_path:
        raise TelegramError("getFile returned no file_path")
    size = int(meta.get("file_size") or 0)

    url = f"{API_ROOT}/file/bot{config.BOT_TOKEN}/{file_path}"
    tmp = dest + ".part"
    last = None
    for attempt in range(ATTEMPTS):
        got = 0
        try:
            with _client() as c, c.stream("GET", url) as r:
                if r.status_code != 200:
                    raise TelegramError(f"download: HTTP {r.status_code}")
                with open(tmp, "wb") as f:
                    for block in r.iter_bytes(_CHUNK):
                        f.write(block)
                        got += len(block)
                        if progress and progress(got, size or got) is False:
                            raise Cancelled("download cancelled")
            os.replace(tmp, dest)
            return
        except Cancelled:
            raise
        except (httpx.HTTPError, OSError, TelegramError) as e:
            last = TelegramError(f"download: {type(e).__name__}: {str(e)[:160]}")
            if attempt < ATTEMPTS - 1:
                time.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
        finally:
            if os.path.exists(tmp) and not os.path.exists(dest):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    raise last or TelegramError("download failed")


def probe() -> dict:
    """Cheap reachability check for the admin panel: can we talk to the Bot API
    at all, and can we see the channel? Run before an export so a missing admin
    right is a sentence in the UI rather than a failure 40 MB in."""
    out = {"ok": False, "bot": None, "channel": None, "error": None}
    if not config.BOT_TOKEN:
        out["error"] = "No bot token configured (VIOS_BOT_TOKEN)."
        return out
    try:
        me = _post("getMe", attempts=2)
        out["bot"] = me.get("username") or me.get("first_name")
    except TelegramError as e:
        out["error"] = str(e)
        return out
    try:
        chat = _post("getChat", {"chat_id": config.CHANNEL_ID}, attempts=2)
        out["channel"] = chat.get("title") or str(config.CHANNEL_ID)
        out["ok"] = True
    except TelegramError as e:
        out["error"] = (f"Bot @{out['bot']} cannot see channel {config.CHANNEL_ID}: "
                        f"{str(e)[:140]}. Add it to the channel as an admin.")
    return out
