"""
vios.process — the second plane: bytes in permanent storage become evidence.

Capture is irreversible and slow; processing is disposable and fast. That
asymmetry is the whole reason the two planes are separate packages. A capture
bug loses a reel forever. A processing bug is fixed by deleting some rows and
running the sweep again — which the coverage ledger makes free.

The plane is five pieces:

  registry    every model and script declared as data: what it needs, what it
              costs, what it produces, and what it depends on
  store       the append-only evidence store — claims, not facts
  coverage    (video, component) → state, with leases, so ten notebooks on ten
              accounts can share the work without talking to each other
  resources   what this machine actually has, right now, measured not assumed
  engine      the rotation loop: pack a cohort that fits VRAM, sweep every
              video, upload the shard, unload, pack the next one

The rule that governs all of it, in the user's words:

    "dont force llms or models to do things that they cant do by prompting,
     use mathmatical and proven systems for specific things"

So timestamps come from arithmetic over shot boundaries, never from a model's
opinion about when something happened. Counts come from counters. Colour comes
from histograms. Models are asked only for the things only a model can give —
meaning, tone, subtext — and even then, several of them are asked and all the
answers are kept.
"""

from __future__ import annotations

__all__ = ["SCHEMA_VERSION", "CHANNELS"]

# v2 added per-frame identity: `claim.frame_idx`/`frame_hi` and the packed
# `frame_vector` / `frame_metric` tables. Additive only — a v1 database is
# migrated in place on open and a v1 shard still replays, because the channel
# holds the only copy of everything processed before this change.
SCHEMA_VERSION = 2

# The evidence channels. This is not a loose vocabulary — it is the same list
# the interface colours by (doc 11's channel spectrum), so adding one here means
# adding a colour there. Keep it small on purpose.
CHANNELS = (
    "speech",     # what was said
    "ocr",        # what was written on screen
    "visual",     # what is physically present in frame
    "audio",      # music, sound design, loudness — the non-speech signal
    "narrative",  # what is happening, and why it holds attention
    "style",      # how it was shot and cut
    "caption",    # what the creator wrote, and what the audience wrote back
    "concept",    # the abstractions: entities, topics, techniques
)
