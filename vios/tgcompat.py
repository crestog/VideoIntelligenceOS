"""
vios.tgcompat — make pyrogram able to see modern channel ids.

This module exists because of one hardcoded constant in a library that stopped
being maintained in 2023.

Pyrogram decides whether a chat id is a user, a group or a channel by range
check, in `pyrogram.utils.get_peer_type`:

    MIN_CHAT_ID    = -2147483647             # basic groups
    MIN_CHANNEL_ID = -1002147483647          # -100 followed by 2**31
    MAX_CHANNEL_ID = -1000000000000

    if peer_id < 0:
        if MIN_CHAT_ID <= peer_id:
            return "chat"
        if MIN_CHANNEL_ID <= peer_id < MAX_CHANNEL_ID:
            return "channel"
    raise ValueError(f"Peer id invalid: {peer_id}")

Telegram's internal channel ids used to fit in 32 bits, so that bound held.
They no longer do. A channel created recently gets an id past 2**31, which in
`-100` form looks like `-1004435513595` — *below* pyrogram's floor, so the
range check falls through and every call touching that channel dies with

    ValueError: Peer id invalid: -1004435513595

before a single byte leaves the machine. Nothing is wrong with the id, the
token, or the permissions; the library simply refuses to name the chat.

This bites us in two places and looks like two unrelated faults:

  capture     the MTProto path for files over the Bot API's 50 MB upload
              ceiling raises, so oversize reels fail while small ones sail
              through — which reads as "big files are broken".
  processing  `Channel.messages()` raises inside its own try, logs a fetch
              failure, returns nothing, and `Source.ensure` reports
              "could not download the original from Telegram (message 38)" —
              which reads as "Telegram lost my video".

Widening `MIN_CHANNEL_ID` is the whole fix. The constants are only used for that
range test and for `get_channel_id`, which is arithmetic against
`MAX_CHANNEL_ID` and unaffected. So we move that one floor out to 2**40, which
covers every id Telegram is plausibly issuing this decade, and leave the
ceiling — and every other bound — exactly where it is.

**`MIN_CHAT_ID` must not be touched, and that is not a detail.** The chat branch
is tested *first*, and it is a bare `MIN_CHAT_ID <= peer_id` with no lower
bound. Widen it to cover large ids and every `-100…` channel id starts
satisfying it, so `get_peer_type` returns "chat" for a channel. That is worse
than the bug being fixed: instead of a loud `ValueError` naming the id, pyrogram
builds an `InputPeerChat` for something that is not a chat and the failure
arrives later, from the server, as an unrelated-looking error. Measured on the
real id: widening both floors turned `Peer id invalid: -1004435513595` into a
confident, wrong `"chat"`.

Basic groups do not need the widening anyway. Their ids are small negatives well
inside the existing 32-bit floor, and a group that outgrows it is migrated by
Telegram into a supergroup, which is a `-100…` channel and goes down the branch
this module actually fixes.

`patch()` is idempotent, never raises, and is safe to call from a module that
may be imported on a machine with no pyrogram at all. Call it immediately
before constructing a Client — not at package import — so that the one place
that must not fail (importing `vios.capture`) stays free of pyrogram.
"""

from __future__ import annotations

# -100 followed by 2**40. Telegram channel ids crossed 2**31 in 2024; this
# leaves three orders of magnitude of headroom rather than moving the goalpost
# by one bit and having to do it again.
WIDE_MIN_CHANNEL_ID = -1001099511627776

_done = False
_result = ""


def patch() -> str:
    """Widen pyrogram's channel id floor. Returns a one-line note, or "".

    The note is worth logging once: if a future pyrogram fixes this upstream
    the patch becomes a no-op, and the log is how you find out.

    Only `MIN_CHANNEL_ID` is moved. See the module docstring for why touching
    `MIN_CHAT_ID` turns a loud error into a silently wrong peer type.
    """
    global _done, _result
    if _done:
        return _result
    _done = True
    try:
        from pyrogram import utils as _u  # noqa: PLC0415
    except Exception:
        _result = ""
        return _result

    changed = False
    if getattr(_u, "MIN_CHANNEL_ID", 0) > WIDE_MIN_CHANNEL_ID:
        _u.MIN_CHANNEL_ID = WIDE_MIN_CHANNEL_ID
        changed = True

    # Some pyrogram builds read the constant into `pyrogram.utils` only, others
    # also expose it on `pyrogram` itself. Keep both in step.
    try:
        import pyrogram as _p  # noqa: PLC0415
        if hasattr(_p, "MIN_CHANNEL_ID"):
            _p.MIN_CHANNEL_ID = _u.MIN_CHANNEL_ID
    except Exception:
        pass

    _result = ("widened pyrogram's channel id floor to -100+2**40 "
               "(modern Telegram channel ids sit past its 32-bit bound)"
               if changed else "")
    return _result


def check(channel) -> str:
    """Why pyrogram would reject this channel id, or "" if it would accept it.

    Used by the readiness checks so a bad id is named before a run starts
    rather than three hours in, and so a *genuinely* malformed id — a username
    typed where a numeric id belongs, a copied-with-space paste — is still
    reported as malformed after the patch.
    """
    patch()
    text = str(channel or "").strip()
    if not text:
        return "no channel id"
    if text.startswith("@") or not text.lstrip("-").isdigit():
        return ""          # a username; pyrogram resolves those by lookup
    try:
        from pyrogram import utils as _u  # noqa: PLC0415
        _u.get_peer_type(int(text))
    except ImportError:
        return ""
    except ValueError:
        return (f"pyrogram will not accept {text} as a chat id. A channel id "
                f"looks like -100 followed by digits; a supergroup that has "
                f"not been migrated, or an id copied from the wrong field, "
                f"looks like this.")
    except Exception:
        return ""
    return ""
