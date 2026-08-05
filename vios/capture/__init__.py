"""
vios.capture — permalinks in, permanent Telegram archive out.

The first of the three planes. It has one job and it is allowed to be slow at
it: take the reels the user saved on Instagram and put the original file, plus
everything Instagram will tell us about it, somewhere that will still be there
in five years. Nothing downstream can be better than what this plane captured,
and capture is the only step that cannot be redone later — the post may be
deleted by then — so it optimises for completeness and safety over speed.

The pieces, each usable alone:

    ledger    the permanent record: what is captured, what is queued, what
              failed and why. SQLite, keyed by Instagram shortcode.
    inputs    the two front doors — the Instagram export ZIP and a markdown
              list of links — normalised to (url, collection) pairs.
    pacing    the anti-detection pacer: lognormal gaps, bursts, breaks and
              quiet hours around a two-minute mean.
    fetch     yt-dlp with maximum extraction (comments, raw info JSON,
              thumbnail), gallery-dl fallback, three-way error taxonomy.
    upload    Telegram as permanent storage: video plus a metadata record
              threaded as a reply, and the ledger itself snapshotted and
              pinned.
    seed      rebuild the ledger by reading the channel, so the reels already
              uploaded are never fetched twice.
    engine    the loop that ties them together and survives being killed.

Typical use, which is also exactly what the admin tab does:

    from vios.capture import get_engine
    eng = get_engine()
    eng.configure(bot_token=..., channel_id=..., api_id=..., api_hash=...,
                  cookies_text=..., target=120)
    eng.import_file("saved.zip", data=raw_bytes)
    eng.preflight()
    eng.start()
"""

from .ledger import (Ledger, open_ledger, canonical, dump_json,
                     QUEUED, FETCHING, UPLOADED, FAILED, UNAVAILABLE, SKIPPED)
from .inputs import parse_any, parse_markdown, parse_export_zip
from .pacing import Pacer
# NOTE: this line rebinds the package attribute `vios.capture.fetch` from the
# submodule to the function — which is the useful export, but it means no other
# module in this package may write `from . import fetch`. Import the names
# directly (`from .fetch import fetch, cleanup`) instead; engine.py does.
from .fetch import fetch, FetchError, tool_versions
from .upload import (Telegram, UploadError, publish, build_caption,
                     upload_snapshot, restore_snapshot)
from .seed import seed_from_channel, seed_from_urls, parse_caption
from .engine import CaptureEngine, get_engine

__all__ = [
    "Ledger", "open_ledger", "canonical", "dump_json",
    "QUEUED", "FETCHING", "UPLOADED", "FAILED", "UNAVAILABLE", "SKIPPED",
    "parse_any", "parse_markdown", "parse_export_zip",
    "Pacer",
    "fetch", "FetchError", "tool_versions",
    "Telegram", "UploadError", "publish", "build_caption",
    "upload_snapshot", "restore_snapshot",
    "seed_from_channel", "seed_from_urls", "parse_caption",
    "CaptureEngine", "get_engine",
]
