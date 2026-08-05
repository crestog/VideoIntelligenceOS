"""
vios.capture.seed — rebuild the ledger from the channel itself.

552 reels are already in Telegram, uploaded by the old Colab script over
months. Nothing in this repository knows about them: the ledger is new and
empty, and an empty ledger means "nothing captured", which would send the
capture engine off to re-download every one of them. That is a week of
Instagram requests spent to obtain files we already have, and a week of
requests is exactly the thing the pacer exists to avoid spending.

So before any run, the channel is read and the ledger is taught what is in it.

The channel is the ground truth
───────────────────────────────
Every uploaded message carries its permalink in the caption. That was true of
the old Colab captions and it is true of `upload.build_caption`, deliberately —
the `🔗 Link:` line is a load-bearing format, not decoration. As long as the
channel exists, the list of captured reels is recoverable with no database at
all. This module is the recovery.

Reading a channel as a bot
──────────────────────────
The Bot API has no `getHistory`; a bot cannot ask what is in a channel. Two
tricks get around it, both borrowed from `atlas.tgchannel` where they are
already proven against this exact channel:

  * The head id: post a message, keep the id it returns, delete it. That id is
    the current top of the channel, so the id space to scan is 1..head.
  * The body: MTProto's `get_messages` *can* read by id, even for a bot. Ids
    are requested in batches of 200; gaps (deleted messages, service events)
    come back as None and are dropped.

Scanning ~600 ids is three or four API calls, so this runs in seconds and is
cheap enough to do at the start of every session as a consistency check rather
than as a one-off migration.

Credentials are passed in, never imported. The scan needs an api id and hash
because MTProto does; if the operator has not supplied them, `seed_from_urls`
is the fallback — paste the list of already-uploaded links and the ledger
adopts them without reading anything.
"""

from __future__ import annotations

import re
import time

from .ledger import Ledger, canonical

BATCH = 200

# The caption fields the two eras share. Written as tolerant regexes rather
# than a strict template because the old Colab captions were composed by hand
# and drifted: some have the emoji, some do not, and one build used "Author"
# where the rest use "Creator".
_KEY_TAG = re.compile(r"vios:([A-Za-z0-9_-]{5,})")
_CREATOR = re.compile(r"(?:Creator|Author|Uploader)\s*:\s*([^\n|]{1,80})",
                      re.IGNORECASE)
_CATEGORY = re.compile(r"(?:Category|Collection)\s*:\s*([^\n|]{1,120})",
                       re.IGNORECASE)
_VIEWS = re.compile(r"Views\s*:\s*([\d,\.]+)", re.IGNORECASE)
_LIKES = re.compile(r"Likes\s*:\s*([\d,\.]+)", re.IGNORECASE)
_COMMENTS = re.compile(r"Comments\s*:\s*([\d,\.]+)", re.IGNORECASE)


def _num(match) -> int | None:
    if not match:
        return None
    try:
        return int(re.sub(r"[^\d]", "", match.group(1)) or 0)
    except (ValueError, TypeError):
        return None


def parse_caption(text: str) -> dict:
    """Pull everything recoverable out of one channel caption.

    Returns {} when there is no permalink — that message is a record document,
    a snapshot, a status note or a stray, and none of those are reels.
    """
    if not text:
        return {}
    can = canonical(text)
    if not can:
        return {}
    key, url, kind = can
    tagged = _KEY_TAG.search(text)
    if tagged:
        # A caption written by this system states its own key. Trust it over
        # the URL parse: if the two ever disagree the tag is the one that was
        # written deliberately.
        key = tagged.group(1)

    cats = []
    cat = _CATEGORY.search(text)
    if cat:
        cats = [c.strip() for c in cat.group(1).split(",") if c.strip()]

    creator = _CREATOR.search(text)
    return {
        "key": key,
        "url": url,
        "kind": kind,
        "collections": cats,
        "uploader": (creator.group(1).strip() if creator else "") or "",
        "views": _num(_VIEWS.search(text)),
        "likes": _num(_LIKES.search(text)),
        "comment_count": _num(_COMMENTS.search(text)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Reading the channel
# ═══════════════════════════════════════════════════════════════════════
def head_message_id(tg) -> int:
    """The current top of the channel: post, note the id, delete.

    The dot exists for a fraction of a second. If the delete fails the dot
    stays, which is untidy and completely harmless — failing the whole scan
    over a stray character would not be.
    """
    mid = tg.send_message("·")
    try:
        tg.call("deleteMessage", {"chat_id": tg.channel, "message_id": mid},
                attempts=1)
    except Exception:
        pass
    return int(mid or 0)


def _facts(msg) -> dict:
    """Media facts off a pyrogram message, whatever kind of media it is."""
    media = (getattr(msg, "video", None) or getattr(msg, "document", None)
             or getattr(msg, "animation", None) or getattr(msg, "photo", None))
    if media is None:
        return {}
    thumb = ""
    thumbs = getattr(media, "thumbs", None) or []
    if thumbs:
        thumb = getattr(thumbs[-1], "file_id", "") or ""
    return {
        "file_id": getattr(media, "file_id", "") or "",
        "file_size": getattr(media, "file_size", None),
        "duration": getattr(media, "duration", None),
        "width": getattr(media, "width", None),
        "height": getattr(media, "height", None),
        "ext": (getattr(media, "file_name", "") or "").rsplit(".", 1)[-1][:8],
        "thumb_file_id": thumb,
    }


def scan_channel(tg, api_id: int, api_hash: str, head: int = 0,
                 start: int = 1, on_progress=None, should_stop=None) -> list:
    """Walk the channel and return one dict per reel-bearing message.

    Runs the whole scan inside a single MTProto session — connecting is the
    expensive part, and the alternative (a session per batch) turns four calls
    into four handshakes.
    """
    try:
        import asyncio
        from pyrogram import Client
    except ImportError:
        raise RuntimeError(
            "pyrogram is not installed, so the channel cannot be scanned. "
            "Seed from a URL list instead, or install pyrogram.")
    if not (api_id and api_hash):
        raise RuntimeError(
            "Reading channel history needs an API id and API hash. Add them "
            "in the admin tab, or seed from a URL list instead.")

    head = int(head or head_message_id(tg))
    if head <= 0:
        raise RuntimeError("Could not determine the newest message id. Check "
                           "the bot can post to the channel.")

    async def _go():
        app = Client("vios_capture_scan", api_id=int(api_id),
                     api_hash=api_hash, bot_token=tg.token,
                     in_memory=True, no_updates=True,
                     max_concurrent_transmissions=2)
        await app.start()
        found, reply_index = [], {}
        try:
            for lo in range(start, head + 1, BATCH):
                if should_stop and should_stop():
                    break
                ids = list(range(lo, min(lo + BATCH, head + 1)))
                msgs = await _batch(app, tg.channel, ids)
                for msg in msgs:
                    text = getattr(msg, "caption", None) or getattr(msg, "text", "")
                    parsed = parse_caption(text or "")
                    if not parsed:
                        # A document replying to a video is that video's
                        # metadata record. Remember the link so the ledger row
                        # can point at it.
                        reply = getattr(msg, "reply_to_message_id", None)
                        if reply and getattr(msg, "document", None):
                            reply_index[int(reply)] = int(msg.id)
                        continue
                    parsed["msg_id"] = int(msg.id)
                    parsed["at"] = getattr(msg, "date", None)
                    parsed.update(_facts(msg))
                    found.append(parsed)
                if on_progress:
                    try:
                        on_progress(min(lo + BATCH - 1, head), head, len(found))
                    except Exception:
                        pass
        finally:
            await app.stop()
        for item in found:
            rec = reply_index.get(item["msg_id"])
            if rec:
                item["record_msg_id"] = rec
        return found

    return asyncio.run(_go())


async def _batch(app, channel, ids):
    """One `get_messages` call, patient about FloodWait.

    A full-channel scan reliably trips FloodWait once or twice; that is the
    server asking for a pause, and honouring it costs seconds where ignoring
    it costs the session.
    """
    import asyncio
    from pyrogram.errors import FloodWait
    for attempt in range(4):
        try:
            out = await app.get_messages(channel, ids)
            if out is None:
                return []
            if not isinstance(out, list):
                out = [out]
            return [m for m in out
                    if m is not None and not getattr(m, "empty", False)]
        except FloodWait as e:
            wait = int(getattr(e, "value", getattr(e, "x", 5))) + 1
            if attempt == 3:
                raise
            await asyncio.sleep(wait)
    return []


# ═══════════════════════════════════════════════════════════════════════
# Writing what was found into the ledger
# ═══════════════════════════════════════════════════════════════════════
def adopt_all(ledger: Ledger, found, source: str = "channel-scan") -> dict:
    """Mark everything the scan found as already uploaded.

    Existing rows are updated, not skipped: a reel that is currently `queued`
    but demonstrably sitting in the channel must move to `uploaded` or the
    engine will fetch it again tonight. That correction is the entire value of
    running the scan on every boot instead of once.
    """
    adopted = refreshed = 0
    for item in found:
        key = item.get("key")
        url = item.get("url")
        if not (key and url):
            continue
        existing = ledger.conn.execute(
            "SELECT state FROM item WHERE key=?", (key,)).fetchone()
        fields = {k: item.get(k) for k in
                  ("record_msg_id", "file_id", "file_size", "ext", "duration",
                   "width", "height", "uploader", "views", "likes",
                   "comment_count")
                  if item.get(k) is not None}
        ledger.adopt(key, url, int(item.get("msg_id") or 0), **fields)
        for col in item.get("collections") or []:
            ledger.conn.execute(
                "INSERT OR IGNORE INTO membership(key,collection) VALUES(?,?)",
                (key, col[:120]))
        if existing is None:
            adopted += 1
        elif existing["state"] != "uploaded":
            refreshed += 1
    ledger.conn.commit()
    ledger.set_meta("last_channel_scan", str(time.time()))
    ledger.set_meta("channel_known", str(len(found)))
    ledger.conn.commit()
    ledger.log("seed", f"{source}: {len(found)} in channel, {adopted} new to "
                       f"the ledger, {refreshed} corrected")
    return {"in_channel": len(found), "adopted": adopted,
            "corrected": refreshed}


def seed_from_channel(ledger: Ledger, tg, api_id: int, api_hash: str,
                      on_progress=None, should_stop=None) -> dict:
    """Scan, then adopt. The one call the engine and the admin tab make.

    Resumes from the highest message id already seen, so the second run scans
    only what arrived since the first. A full rescan is available by clearing
    `scan_high_water`, which is what the admin tab's "rescan from scratch"
    does.
    """
    high = int(ledger.get_meta("scan_high_water", 0) or 0)
    head = head_message_id(tg)
    if head <= high:
        return {"in_channel": 0, "adopted": 0, "corrected": 0,
                "scanned_to": high, "skipped": True}
    found = scan_channel(tg, api_id, api_hash, head=head, start=max(1, high + 1),
                         on_progress=on_progress, should_stop=should_stop)
    out = adopt_all(ledger, found)
    if not (should_stop and should_stop()):
        ledger.set_meta("scan_high_water", str(head))
        ledger.conn.commit()
    out["scanned_to"] = head
    out["skipped"] = False
    return out


def seed_from_urls(ledger: Ledger, text: str,
                   source: str = "manual-seed") -> dict:
    """Adopt a pasted list of permalinks as already captured.

    The escape hatch for when MTProto is unavailable: the operator can export
    the old list by hand and the ledger will never fetch those reels. Their
    message ids are unknown, so the rows carry `msg_id = 0` — enough to stop a
    re-download, and a later channel scan fills in the rest.
    """
    n = 0
    for line in (text or "").splitlines():
        can = canonical(line)
        if not can:
            continue
        ledger.adopt(can[0], can[1], 0)
        n += 1
    ledger.conn.commit()
    ledger.log("seed", f"{source}: {n} link(s) marked already captured")
    return {"adopted": n}
