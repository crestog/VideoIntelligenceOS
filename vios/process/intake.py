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

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time

# The identity law. Imported, not restated — see the note over the same import
# in `store.py`. What this module needs from it is the shape test and the two
# key-minting helpers, so that a file arriving from a folder is named the way
# every reader already expects to find it.
from atlas import identity

# 20 MB — the Bot API's download ceiling. Reels are 2–8 MB, so this path
# carries most of the archive; the ones it refuses are the three-minute ones.
BOT_DOWNLOAD_LIMIT = 20 * 1024 * 1024

SHARD_PREFIX = "vios-evidence-"
SHARD_SUFFIX = ".jsonl.gz"

# What counts as a video file when scanning a folder the operator pointed at.
_VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


def _safe(key: str) -> str:
    """A video key as a filename stem. Keys are shortcodes, but never trust a
    string that arrived in a URL with a path join."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in key)[:64]


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

    Videos uploaded straight into the channel come through here too, under
    `up_<msg_id>` keys with no url. They are marked `uploader="user"` and given
    the collection `user uploaded videos`, which is what makes them filterable
    in the interface — and, via the claim written below, findable by search
    alongside everything else rather than being a second-class category that
    only the library tab knows about.
    """
    out = {"seen": 0, "added": 0, "uploads": 0, "refused": 0,
           "ledger": ledger_path}
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
        fresh_uploads: list = []
        for r in conn.execute(sql):
            out["seen"] += 1
            key = r["key"]
            if not identity.looks_canonical(key):
                # The ledger mints shortcodes and `up_<msg_id>`, so this is
                # unreachable for a ledger this build wrote. It is here because
                # the ledger is also rebuilt by scanning the channel, and a
                # caption that lost its permalink is one bad parse away from
                # offering a bare number as a key. Counted, not adopted.
                out["refused"] += 1
                continue
            upload = _is_upload_key(key)
            cols = sorted(collections.get(key, []))
            if upload and UPLOAD_COLLECTION not in cols:
                # The seeder tags these, but a row adopted before that branch
                # existed — or by `adopt_upload` from the bot — would arrive
                # bare. The collection is how the interface groups them, so it
                # is asserted here rather than assumed.
                cols.append(UPLOAD_COLLECTION)
            head = {
                "msg_id": r["msg_id"],
                "record_msg_id": r["record_msg_id"],
                "file_id": r["file_id"],
                "collections": cols,
                "title": r["title"],
                "views": r["views"],
                "likes": r["likes"],
                "comment_count": r["comment_count"],
                "comments_got": r["comments_got"],
                "captured_at": r["done_at"],
                **({"origin": "telegram-upload"} if upload else {}),
            }
            store.add_video(
                key, url=r["url"], uploader=(r["uploader"] or
                                             ("user" if upload else "")),
                duration=r["duration"], width=r["width"], height=r["height"],
                bytes=r["file_size"], sha256=r["sha256"], msg_id=r["msg_id"],
                taken_at=r["taken_at"], meta={"capture": head})
            if key not in known:
                out["added"] += 1
                if upload:
                    out["uploads"] += 1
                    fresh_uploads.append((key, r["title"] or ""))
        if fresh_uploads:
            out["claimed"] = _claim_uploads(store, fresh_uploads)
    except sqlite3.Error as exc:
        out["reason"] = f"capture ledger unreadable: {exc}"
    finally:
        conn.close()
    return out


UPLOAD_COLLECTION = "user uploaded videos"

# The observer for the origin claim. Not a model, so `revision` carries the
# whole meaning: this is a fact about where the file came from, recorded once,
# and it must read as evidence with attribution like everything else rather
# than as a special column only one query knows to look at.
_ORIGIN_OBSERVER = ("intake", "capture-ledger", "1")


def _is_upload_key(key: str) -> bool:
    """`up_<digits>`, without importing the capture plane at module scope.

    The processing plane runs on hosts where `vios.capture` is present but
    pyrogram is not, and the import chain there is worth keeping short. The
    definition lives in `vios/capture/ledger.py`; this mirrors it and the test
    below asserts the two agree.
    """
    key = str(key or "")
    return key.startswith("up_") and key[3:].isdigit()


def _claim_uploads(store, uploads) -> int:
    """Write the origin of each hand-uploaded video as a claim.

    Being able to *filter* to uploads is not the same as being able to *find*
    them: the collections list lives in `video.meta`, which no search backend
    reads. A claim goes through the same FTS index as a transcript line, so
    "user uploaded videos" answers in the search box, and a caption the person
    typed when they sent it becomes searchable text about the video instead of
    a string sitting unused in a ledger column.
    """
    oid = store.observer(*_ORIGIN_OBSERVER)
    n = 0
    for key, title in uploads:
        claims = [{"channel": "concept", "kind": "origin",
                   "value": UPLOAD_COLLECTION, "confidence": 1.0}]
        if title.strip():
            claims.append({"channel": "caption", "kind": "upload_note",
                           "value": title.strip()[:2000], "confidence": 1.0})
        try:
            n += store.add_claims(key, oid, claims)
        except Exception:      # noqa: BLE001
            # A video whose shots pass has not run yet cannot take a claim that
            # references one — these do not, but a future edit might, and one
            # unwritable origin note must not abort the whole sync.
            continue
    return n


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


def _sha256(path: str) -> str:
    """The file's digest, read in chunks. Mirrors `capture/fetch._sha256`."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def resolve_stem(store, stem: str, path: str, ledger: dict = None,
                 known=None) -> tuple:
    """What video is this file? Returns (video_key, how, evidence, digest).

    `digest` is the file's sha256 when the answer needed one and "" when it did
    not — a name that resolved by itself is not worth reading five megabytes to
    confirm, and the one that did resolve by bytes should not be hashed twice.

    A filename is not an identity, and this is the function that stops one from
    becoming one. Five answers, strongest first:

      key         the stem already *is* an identity — `up_4471`, a `file_` key,
                  or a key the store or the ledger already knows. A folder of
                  capture-engine output merges with the rows it came from
                  instead of duplicating them, which is why this is first.
      shortcode   the stem is shaped like an Instagram shortcode: 10–12
                  characters of base64 with at least one letter. Narrower than
                  `looks_canonical` on purpose — `IMG_20240513_112233` passes
                  that and is a camera's filename, not a permalink.
      msg_id      `10.mp4`, `tg38.mp4`, `frames_8` — a *location*, not a name.
                  The store knows what lives at message 10 and so does the
                  ledger; ask, and use the answer.
      content     the bytes are already in the archive under some other name.
      minted      none of the above: `file_<digest16>`, derived from the bytes,
                  so the same file adopted twice under two names is one video.

    The last two are the point. The old code took the stem, sanitised it, and
    adopted it — so a folder of files named after Telegram message ids minted
    thirty-two numeric identities beside the thirty real shortcodes they were
    already filed under. There is no stem this function returns a bare number
    for, and no filename it invents a second identity from.
    """
    stem = str(stem or "").strip()
    safe = _safe(stem)
    ledger = ledger or {}
    if identity.is_upload(safe) or identity.is_local(safe):
        return safe, "key", f"filename {stem} is a key", ""
    if safe and (safe in (known or ()) or ledger.get(("key", safe))):
        return safe, "key", f"filename {stem} is a video already here", ""

    # A message id, however it was spelled: `10`, `tg38`, `msg_40`, `frames_8`.
    # `atlas.identity` owns the list of spellings, because it is the same list it
    # refuses as identities — read here as what they actually are.
    n = identity.msg_id_in(safe)
    if n:
        hit = store.video_key_for_msg(n)
        if hit:
            return hit, "msg_id", f"filename {stem} is message {n}", ""
        hit = str(ledger.get(("msg", n)) or "")
        if hit:
            return hit, "msg_id", f"capture ledger: message {n}", ""
    elif _SHORTCODE_STEM.match(safe):
        return safe, "shortcode", f"filename {stem} is a shortcode", ""

    digest = _sha256(path)
    if not digest:
        return "", "unreadable", f"could not read {path}", ""
    hit = store.video_key_for_sha(digest)
    if hit:
        return hit, "content", f"same bytes as {hit}", digest
    hit = str(ledger.get(("sha", digest)) or "")
    if hit:
        return hit, "content", f"capture ledger: same bytes as {hit}", digest
    return identity.local_key(digest), "minted", f"sha256 of {stem}", digest


# 10–12 characters of Instagram's alphabet with at least one letter. A run of
# digits that long is a numeric id, not a shortcode, and is read as one above.
_SHORTCODE_STEM = re.compile(
    r"^(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{10,12}$")


def _ledger_index(ledger_path: str) -> dict:
    """`{("key", k): k, ("msg", id): k, ("sha", digest): k}` from the ledger.

    Read-only and optional, exactly like `sync`'s connection: a machine that
    adopted a folder may have no ledger at all, and the store's own `msg_id` and
    `sha256` columns answer most of the question anyway. Ambiguity is dropped
    rather than guessed — a message id or a digest claimed by two keys resolves
    to nothing, and the file falls through to its own `file_` identity.
    """
    if not ledger_path or not os.path.exists(ledger_path):
        return {}
    try:
        conn = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True,
                               timeout=15.0)
    except sqlite3.Error:
        return {}
    seen: dict = {}
    doubled: set = set()
    try:
        for row in conn.execute("SELECT key, msg_id, record_msg_id, sha256 "
                                "FROM item WHERE key IS NOT NULL"):
            key = str(row[0] or "").strip()
            if not identity.looks_canonical(key):
                continue
            marks = [("key", key)]
            for value in (row[1], row[2]):
                try:
                    n = int(value)
                except (TypeError, ValueError):
                    continue
                if n > 0:
                    marks.append(("msg", n))
            if row[3]:
                marks.append(("sha", str(row[3]).strip().lower()))
            for mark in marks:
                if seen.setdefault(mark, key) != key:
                    doubled.add(mark)
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    for mark in doubled:
        seen.pop(mark, None)
    return seen


def _remember_path(store, video_key: str, path: str) -> bool:
    """Record where a local copy of an already-known video is sitting.

    `add_video` fills blanks only, which is right for two sources that each know
    part of the truth and wrong here: a row synced from the ledger already has a
    `meta`, so a blanks-only write would drop the path and `Source.ensure` would
    keep going to the network for a file that is on the disk. Read, merge, write
    — and nothing else in `meta` is touched.
    """
    row = store.video(video_key) or {}
    try:
        meta = json.loads(row.get("meta") or "{}")
    except (TypeError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    if meta.get("local_path") == path:
        return False
    meta["local_path"] = path
    return bool(store.update_video(video_key, meta=meta))


def adopt_folder(store, folder: str, limit: int = 0,
                 ledger_path: str = "") -> dict:
    """Register every video file in a folder as a processable row.

    The capture ledger is one way videos arrive; a folder is the other. A
    Kaggle dataset of reels, a rescued scratch directory, a hand-assembled test
    set — none of them have a Telegram message id, and requiring one meant the
    processing engine could not touch a video that was already sitting on the
    disk it was running on.

    The row records its own path, so `Source.ensure` finds it with no network
    at all. Idempotent by key, so re-running after adding files is safe.

    Every file's identity goes through `resolve_stem`, which is why pointing
    this at the same archive twice — once as shortcodes, once as message-id
    filenames — cannot produce two videos. `by` in the report says which
    evidence answered, per file, so an operator can see what was merged instead
    of being handed a number.
    """
    out = {"seen": 0, "added": 0, "folder": folder,
           "by": {}, "merged": 0, "refused": 0}
    if not os.path.isdir(folder):
        out["reason"] = f"no folder at {folder}"
        return out

    known = set(store.video_keys())
    found: list = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if fn.lower().endswith(_VIDEO_EXT):
                found.append(os.path.join(root, fn))
        if limit and len(found) >= limit:
            break
    found.sort()
    if limit:
        found = found[:limit]

    ledger = _ledger_index(ledger_path)
    for path in found:
        out["seen"] += 1
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size <= 4096:
            # `Source.ensure` treats anything this small as not-a-video, so
            # adopting it would mint a row that can never be sourced and
            # fails on every sweep forever.
            out["too_small"] = out.get("too_small", 0) + 1
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        key, how, evidence, digest = resolve_stem(store, stem, path, ledger,
                                                  known)
        if not key:
            out["refused"] += 1
            continue
        out["by"][how] = out["by"].get(how, 0) + 1
        if key in known:
            if how in ("msg_id", "content"):
                out["merged"] += 1
            _remember_path(store, key, path)
            continue
        if not store.add_video(
                video_key=key, url="", uploader="", duration=None,
                width=None, height=None, bytes=size, taken_at=None,
                # The digest goes in when the identity was derived from it,
                # which is what makes the *next* file with these bytes resolve
                # by content instead of re-deriving the same key from scratch —
                # and what lets the twin sweep see two permalinks over one file.
                sha256=(digest or None),
                meta={"capture": {}, "local_path": path, "origin": "folder",
                      "named_by": how, "filename": stem,
                      "evidence": evidence}):
            # Unreachable by construction — `resolve_stem` returns an identity
            # or nothing — and counted rather than asserted, because the store
            # refusing a key it was handed is a fact worth seeing if the two
            # ever disagree.
            out["refused"] += 1
            continue
        known.add(key)
        out["added"] += 1
    return out


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
        # Set at `start()` when the channel resolved but reading it did not, and
        # read by `_why()` so an oversize video says which of the two transports
        # actually declined it. Distinct from `reason`, which is why there is no
        # session at all.
        self.peer_error = ""
        # fault string → how many times it has happened, so a permanent refusal
        # is logged once with a running count rather than once per attempt.
        self._seen_errors: dict = {}
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

                # One `get_chat` before anything asks for a message, and it is
                # not a health check — it is the fix for a whole class of
                # failure. `in_memory=True` means the session's peer cache is
                # empty at every start, and `no_updates=True` means nothing will
                # ever arrive to fill it. Pyrogram cannot address a channel it
                # has not resolved, so `get_messages` answers
                # `[400 PEER_ID_INVALID]` while the session itself reports
                # perfectly healthy — which is exactly what the log showed:
                # "MTProto session open" followed by every fetch refused. The
                # call writes the channel into the cache and each later
                # `get_messages` addresses it by the id it now knows.
                #
                # Failure here is recorded, not raised. A bot that cannot read
                # the channel still uploads shards perfectly well, and the
                # session is worth keeping for that.
                try:
                    chat = await app.get_chat(self.tg.channel)
                    return app, getattr(chat, "title", "") or str(
                        self.tg.channel), ""
                except Exception as exc:            # noqa: BLE001
                    return app, "", f"{type(exc).__name__}: {str(exc)[:160]}"

            try:
                self._app, title, peer_err = self._submit(_boot(), timeout=180)
                self.ready = True
                self.reason = ""
                self.peer_error = peer_err
                if peer_err:
                    # Said once, at start, in the one place that can explain it
                    # — rather than once per video per rotation with no cause
                    # attached, which is how this arrived before.
                    self.log(f"MTProto session open, but the channel could not "
                             f"be resolved — {peer_err}. Every message fetch "
                             f"will be refused; check the bot is a member of "
                             f"the channel and that VIOS_TG_CHANNEL names it "
                             f"correctly.")
                else:
                    self.log(f"MTProto session open — channel resolved"
                             + (f" ({title})" if title else ""))
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
            # Said once per distinct fault, not once per attempt. This is the
            # line that made a session log unreadable: `revive_failed` offers a
            # failed video again every rotation, seventy-six rotations happened,
            # and a permanent refusal like `PEER_ID_INVALID` printed the same
            # sentence several times a second for the length of the run. The
            # count is carried so the repetition is still visible as a number,
            # which is the part worth knowing.
            self._seen_errors[self.last_error] = n = \
                self._seen_errors.get(self.last_error, 0) + 1
            if n == 1:
                self.log(f"message fetch failed: {self.last_error}")
            elif n in (10, 100, 1000) or n % 5000 == 0:
                self.log(f"message fetch failed ×{n} (same fault): "
                         f"{self.last_error}")
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

    def __init__(self, tg=None, channel: Channel | None = None, log=None,
                 local_dirs=None):
        self.tg = tg
        self.channel = channel
        self.log = log or (lambda m: None)
        self.local_dirs = [d for d in (local_dirs or []) if d]
        self.downloaded = 0
        self.bytes = 0
        self.reused = 0
        self.from_disk = 0

    @staticmethod
    def _local_path(video: dict) -> str:
        """The path `adopt_folder` recorded, if the file is still there."""
        meta = video.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta or "{}")
            except json.JSONDecodeError:
                return ""
        if not isinstance(meta, dict):
            return ""
        path = meta.get("local_path") or ""
        if path and os.path.exists(path) and os.path.getsize(path) > 4096:
            return path
        return ""

    def find_local(self, key: str) -> str:
        """The reel's mp4 somewhere on this machine, or "".

        Telegram is the archive, but it is not the only place the bytes are.
        The capture engine leaves its work directory behind, a Kaggle dataset
        can be mounted read-only, and an operator with a folder of downloads
        should not have to upload them just to get them back. Checked before
        any network call, because a local file is both free and instant.
        """
        stem = _safe(key)
        for root in self.local_dirs:
            if not os.path.isdir(root):
                continue
            for cand in (os.path.join(root, f"{stem}.mp4"),
                         os.path.join(root, stem, "source.mp4"),
                         os.path.join(root, stem, f"{stem}.mp4")):
                if os.path.exists(cand) and os.path.getsize(cand) > 4096:
                    return cand
            # A shallow walk, not a full one: an unbounded recursive scan of a
            # Kaggle input mount runs for minutes per video.
            try:
                for entry in os.scandir(root):
                    if (entry.is_file() and stem in entry.name
                            and entry.name.lower().endswith(_VIDEO_EXT)
                            and entry.stat().st_size > 4096):
                        return entry.path
            except OSError:
                continue
        return ""

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

        # On disk already? Then nothing else needs to happen for the bytes.
        if not have_source:
            found = self._local_path(video) or self.find_local(
                str(video.get("video_key") or ""))
            if found:
                try:
                    if os.path.abspath(found) != os.path.abspath(dest):
                        shutil.copyfile(found, dest)
                    have_source = True
                    self.from_disk += 1
                except OSError as exc:
                    self.log(f"found {found} but could not use it: {exc}")

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
                # Which transport declined, truthfully. The previous wording said
                # "MTProto is unavailable" whatever had happened, so a session
                # that was open and had simply been refused the channel read as a
                # session that was never there — and the log carried both
                # sentences a few lines apart, each contradicting the other.
                if self.channel is None or not self.channel.ready:
                    why = "MTProto is unavailable"
                elif getattr(self.channel, "peer_error", ""):
                    why = ("MTProto is open but the channel never resolved "
                           f"({self.channel.peer_error})")
                else:
                    why = ("MTProto is open but did not return the message "
                           + (f"({self.channel.last_error})"
                              if self.channel.last_error else
                              "— it may have been deleted"))
                self.log(f"{video.get('video_key')}: {size / 1048576:.0f} MB is "
                         f"over the Bot API limit and {why}")
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
                    "it, or put the file on disk where the engine can find it.")

        err = getattr(self.channel, "last_error", "") if self.channel else ""
        if err:
            return (f"Telegram refused the fetch for message {msg_id or '?'}: "
                    f"{err}")

        if self.channel is None or not self.channel.ready:
            why = (getattr(self.channel, "reason", "") if self.channel
                   else "no MTProto session was opened")
            over = (size and size > BOT_DOWNLOAD_LIMIT)
            where = (f" Nothing matching was found on disk either (looked in "
                     f"{', '.join(self.local_dirs)})." if self.local_dirs else "")
            return (f"message {msg_id or '?'} needs MTProto and it is not "
                    f"available ({why or 'reason unknown'})."
                    + (f" The file is {size / 1048576:.0f} MB, over the Bot "
                       f"API's 20 MB download limit, so there is no fallback."
                       if over else
                       " Add the API id and API hash on the Setup page.")
                    + where)

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
                   should_stop=None, batch: int = 200,
                   ledger_path: str = "") -> dict:
    """Replay every evidence shard in the channel that this database lacks.

    The first thing a fresh session should do. Ten accounts push shards; any
    one of them can pull all of them and end up with the union, because
    `import_shard` is idempotent by uid and order-independent. A shard already
    named in the local `shard` table is not downloaded again — which is what
    makes running this at every startup cheap rather than a full re-import.

    `ledger_path` is optional and read once, not per shard. It answers "which
    reel is message 38?" for a shard an older build wrote under message ids
    before the `video` table has anything to answer with — which on a fresh
    restore is every shard, since the `video` rows arrive inside the shards
    themselves.
    """
    out = {"found": 0, "imported": 0, "skipped": 0, "claims": 0, "vectors": 0,
           "coverage": 0, "refused": 0, "rehomed": 0, "errors": []}
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
    # Flattened once: `{msg_id: video_key}` is all `import_shard` needs, and
    # opening the ledger per shard would be 76 connections for one dictionary.
    msg_aliases = {mark[1]: key for mark, key in _ledger_index(ledger_path).items()
                   if mark[0] == "msg"}
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
                counts = store.import_shard(path, msg_aliases)
                out["imported"] += 1
                out["claims"] += counts.get("claim", 0)
                out["vectors"] += counts.get("vector", 0)
                # Worth its own number in the report: it is the difference
                # between a restored session that knows what it has already done
                # and one that re-earns three hours of it.
                out["coverage"] += counts.get("coverage", 0)
                # Two numbers about identity, and the pair is the story. A shard
                # an older build wrote under a Telegram message id gets its
                # evidence rehomed onto the shortcode that message carried;
                # `refused` is what not even the ledger could place. Both belong
                # in the report rather than in a log nobody opens: this is the
                # only place a restore can say "this archive used to name videos
                # two different ways, and here is what became of it".
                out["rehomed"] += counts.get("rehomed", 0)
                out["refused"] += counts.get("refused", 0)
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

    `names` carries the directory names actually removed. The engine needs them:
    with several lanes running it keeps a set of videos whose bytes are staged,
    and a stage that says "ready" about a directory this function has just
    deleted sends a worker to a video with no `source.mp4`.
    """
    keep = keep or set()
    out = {"removed": 0, "freed_mb": 0, "names": []}
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
            out["names"].append(os.path.basename(path))
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
