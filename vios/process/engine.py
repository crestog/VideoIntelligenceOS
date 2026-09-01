"""
vios.process.engine — the rotation loop.

The user described this system before any of it existed, and the description is
worth keeping in front of the code:

    "available resources (2 gpu of 15 gb ram 30 gb, 57 gb disk) it loads few
     models according to resources, then use those models to process all
     videos, save in database and uploads into permanent storage... then it
     deletes those models and load next modles and scripts analyzes all videos
     and upload its database to permanent storage and the loop continues, till
     all the videos are processed and entire database is completed"

That is exactly what happens here, and the ordering it implies is the one
non-obvious decision in the file. There are two ways to sweep thirty passes
over five thousand videos:

    pass-major     for each component: for each video: run it
    cohort-major   for each cohort:    for each video: run every resident pass

Pass-major is simpler and wrong. The source mp4 lives in Telegram, not on disk,
so every pass would re-download and re-decode the same reel — thirty times,
five thousand times over. Cohort-major downloads once, decodes once, and runs
every pass that is already in VRAM before moving on. The cost is a cache of
working directories on scratch disk, which is bounded and evicted; the saving
is roughly twenty-nine downloads per video.

Everything else follows from three facts about where this runs:

**A Kaggle session is killed at twelve hours, without warning.** So progress is
durable after every single pass, not at the end of a cohort — the coverage
table is written before the next pass starts. A session that dies mid-cohort
loses at most the pass that was running.

**Ten accounts run this concurrently and cannot talk to each other.** So the
worker takes a static partition and never looks outside it, and publishes its
findings as append-only shards that any other worker can replay.

**Nothing here owns the archive.** The mp4s belong to the channel and the
evidence belongs to the shards. This process is disposable; if it burns down
mid-sweep, the next one re-reads the shards and continues.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import threading
import time
import traceback
from collections import deque

from . import intake, jobs, media, registry, resources
from .. import creds as _creds
from .coverage import Coverage, worker_id
from .runners import get as get_runner
from .runners import missing as missing_runners
from .runners.base import (DeferPass, Job, ModelCache, PassUnavailable,
                           SkipPass)
from .store import Store
from .store import observer_id as store_observer_id

from ..capture.upload import BOT_UPLOAD_LIMIT, Telegram, UploadError

IDLE, RUNNING, PAUSED, STOPPING, ERROR = (
    "idle", "running", "paused", "stopping", "error")

# Videos between shard uploads. One, because the honest unit of "already done"
# is a video: every pass in this rotation has run on it, and until that reaches
# the channel a session that dies has to earn it again. Fifty was a guess sized
# against a four-minute video; the passes now read every frame, so fifty videos
# is hours of GPU work to lose in one stroke.
#
# The floor below is what stops that becoming a chat full of two-kilobyte files
# when a cohort of cheap signal passes flies through thirty reels a minute.
PUBLISH_EVERY = 1

# Never upload two shards closer together than this. A shard costs a Telegram
# round trip and a channel message; below about a minute apart they stop being
# checkpoints and start being noise.
#
# Sixty rather than ninety now that the upload runs on its own thread: the floor
# used to be paying for a worker's stalled time as well as the channel's
# patience, and it no longer is.
PUBLISH_MIN_SECONDS = 60.0

# …and never let unpublished work get older than this, however few videos it
# came from. This is the number that bounds what a killed Kaggle session
# actually costs, so it is the one to change if that cost feels wrong.
#
# Five minutes, halved from ten. The old ceiling was chosen when a publish stopped
# the only worker there was, so making it tighter made the run slower. With the
# publisher on its own thread nobody waits for it, and the honest reading of "we
# cannot trust a Kaggle session" is that the worst-case loss window should be as
# small as the channel will tolerate.
PUBLISH_MAX_SECONDS = 300.0

# How often the publisher thread wakes on its own to re-check the clocks. Short
# enough that the ceiling above means what it says, long enough to be free.
PUBLISH_TICK_SECONDS = 10.0

# One stage bundle part. The Bot API refuses a document over 50 MB and this
# engine's uploader has no MTProto document path, so a bundle that would exceed
# it is split rather than lost. 45 leaves room for the multipart envelope.
STAGE_PART_BYTES = 45 * 1024 * 1024

# One evidence shard, same cap and a wider margin. Measured: `Shard
# bce94897-0001 not uploaded: Request Entity Too Large`, after which `_publish`
# advanced all four watermarks off the export's own stats and nothing ever sent
# that file again — so a 413 was permanent evidence loss the moment the Kaggle
# session's disk went away. `upload.call` does not retry it either, correctly: a
# 4xx whose body is HTML rather than JSON is the request's fault, and sending the
# same bytes again would fail the same way.
#
# 40 rather than 45 because a shard has two costs a stage part does not. The
# coverage section is written *after* the budget stops applying (it is what stops
# a restored database from re-earning finished work, so it must travel whole),
# and the size is probed every `Store.PROBE_EVERY` of uncompressed output rather
# than per row, so the cut can overshoot by up to one probe interval's worth of
# compressed output. Both are small; 5 MB of headroom is enough for either to be
# wrong by a lot. The remaining margin under 50 also matters for time: the
# uploader's `READ_TIMEOUT` is 180 s, and a Kaggle notebook's upstream is not
# fast.
SHARD_BUDGET_BYTES = 40 * 1024 * 1024

# How many held shards to re-offer per publish tick, and how many send attempts
# one shard is worth before it is left alone. A held shard is the only copy of
# its evidence, so it is retried — but a rejection that is about the file rather
# than the network will not change, and re-sending 40 MB every ten seconds for
# the rest of a session costs the bandwidth the *next* shard needs. After this
# many the row keeps its path and stays in the status panel's `held` count, which
# is what makes the loss visible instead of silent.
#
# It also keeps its 40 MB. That is the deliberate half of the trade — the file is
# the evidence — but it means a channel that is refusing everything accumulates
# held bytes with no ceiling, so `_publish` checks the disk floor before writing
# the next one and defers instead. `held_mb` in the status payload is the number
# to watch: it is what the deferral will be about.
HELD_SHARD_PER_TICK = 3
HELD_SHARD_ATTEMPTS = 6

# Bundle file naming. Kept distinct from `intake.SHARD_PREFIX` on purpose: a
# restore walking the channel must be able to tell a replayable delta from a
# whole-database snapshot by filename alone, because downloading the wrong one
# and replaying it would either do nothing or duplicate a database.
STAGE_PREFIX = "vios-stage-"

# How many characters of a bundle caption the error digest may spend. Telegram
# caps a caption at 1024, and the rest of it — file name, counts, worker, part
# number, revision, delta — is already ~300. One run produced 218 errors; an
# unbounded list does not truncate gracefully, it takes the caption past the cap
# and the upload fails, losing the bundle in order to carry errors that would not
# have been readable in a caption anyway. The full list travels inside the file.
STAGE_CAPTION_ERROR_BUDGET = 620

# ══════════════════════════════════════════════════════════════════════════
# How many of each worker
# ══════════════════════════════════════════════════════════════════════════
# One GPU worker per card, by default. The measured problem this fixes: at 44
# minutes into a two-T4 run, card 0 held six gigabytes of loaded models at 0%
# utilisation, card 1 held 105 MiB at 0%, and the CPU was at 264% grinding
# through ffmpeg. Every runner resolves `.to("cuda")` against CUDA's *current
# device*, which is per host thread — so one `torch.cuda.set_device(i)` per
# worker thread puts that worker's whole cohort on card `i`, with no runner
# edits at all. `probe()` reports `usable_vram_mb` as the *minimum* free across
# cards, so a cohort the packer approved already fits on each card
# independently: replicating a cohort per card needs no repacking.
_GPU_LANES = int(os.environ.get("VIOS_GPU_LANES", "0") or 0)

# Two CPU workers, because ffmpeg is already internally multithreaded and four
# vCPUs will not feed more without the two fighting each other.
_CPU_LANES = max(0, int(os.environ.get("VIOS_CPU_LANES", "2") or 0))

# How many videos are kept fully staged — source downloaded, frame tiers
# extracted — ahead of the ones being processed. This is the number that
# produced constant GPU utilisation in the architecture that worked: a GPU
# worker never blocks on Telegram or on ffmpeg, because by the time it asks for
# a video the bytes are already on disk. Three is enough to cover one download
# plus one extraction at the rate a cohort consumes them, without letting the
# working-directory cache run away from `CACHE_BUDGET_MB`.
_PREP_AHEAD = max(1, int(os.environ.get("VIOS_PREP_AHEAD", "3") or 3))

# How many times the prep lane will try to stage one video before its rows are
# failed with that reason. A video that cannot be staged must stop being a
# `candidates` answer, or the planner offers it every second and the cohort never
# drains — and a permanent stall is a far worse outcome than a failed row that
# `revive_failed` picks up next rotation. Three, because the common cause is a
# Telegram hiccup that the second attempt clears.
PREP_MAX_TRIES = 3

# `VIOS_LANES=0` (or `VIOS_WORKERS=1`) forces the old single-threaded sweep. Kept
# because it is the shape eleven months of runs were debugged in: if the job
# plane ever misbehaves, this is how an operator gets a known-good run tonight
# rather than tomorrow. The workers path is the default, not the experiment.
_LANES_ON = (os.environ.get("VIOS_LANES", "1") or "1").strip().lower() \
    not in ("0", "off", "false", "no")

# How long the planner waits for a cohort's lanes to drain before deciding
# something is wedged and finishing the cohort on its own thread. Generous: a
# single Qwen3-VL pass on a long reel is minutes, and a whole cohort slice for
# one video can legitimately be a quarter of an hour.
_COHORT_DRAIN_TIMEOUT = float(os.environ.get("VIOS_COHORT_TIMEOUT", "3600")
                              or 3600)

# Extra VRAM to leave unclaimed, *on top of* the 1 GB `resources.probe` already
# holds back. The allocator fragments over a twelve-hour session in a way a
# single measurement at startup cannot see, and the reel that OOMs is always the
# one with forty shots, three hours in.
VRAM_HEADROOM_MB = 384

# Working-directory cache. Proxy plus frames plus wav is 8–20 MB per reel, so
# 12 GB holds roughly a thousand — a whole cohort's worth for most partitions.
CACHE_BUDGET_MB = 12_000

# Never let free space fall under this. Four places read it, and they are four
# because each owns bytes the others cannot see: the startup check (a disk that
# is already full before any work), `intake.evict` (working directories, the only
# thing that grows per video), `_mark_stage_pending` (a held snapshot, which it
# discards because the database can rebuild one) and `_publish` (which defers the
# next shard rather than discarding anything, because a shard the watermark has
# stepped over is the only copy of its rows).
#
# `shard_dir` was outside all of them until `_reclaim_shards` existed. It is not
# a cache — nothing rebuilds a shard — so eviction never applied, and a published
# file therefore lived as long as the session: 74 of them across the nine
# sessions in this archive, 248.7 MB, every one already in the channel.
DISK_FLOOR_MB = 4_000

IDLE_POLL_SECONDS = 90

# How long a pass that cannot run on this kind of machine waits before it is
# offered again. It is not a backoff — nothing failed and nothing will change in
# five minutes — it is the interval at which the engine re-asks the question,
# because the answer depends on where the coverage database is being read. Six
# hours is long enough that a session never churns on it and short enough that a
# database restored onto a machine that *can* run the pass starts it the same
# working day rather than a month later.
ELSEWHERE_SECONDS = 6 * 3600

# The `meta` key that records that this database has already had its
# environment-fault declines handed back. One repair per database, not one per
# boot: the rows it fixes were written by a build in which a missing package and
# a missing audio track were the same outcome, and after `PassUnavailable` the
# engine stops creating them. Keyed by name and version so a future repair of a
# different shape gets its own key instead of silently inheriting this one's
# "already done".
RECLAIM_MARKER = "reclaimed-unavailable-skips:1"

# How recently the channel must have been scanned for the sweep to trust the
# scan and not repeat it. The boot restore and `restore_on_start` do the same
# walk over thousands of messages, seconds apart, and the second one downloads
# nothing — but it still costs the minutes the first one cost. Ten minutes is
# far longer than the gap between them and far shorter than a session, so a
# sweep restarted hours later scans properly.
SHARDS_FRESH_SECONDS = 10 * 60

# `frame 320/900`, `window 64/1200`, `shots 12-19 of 40` — every runner's
# heartbeat string that carries a position. Matching once here turns the live
# panel's progress bar into a number the engine computed rather than one the
# browser guessed from a label.
_PROGRESS_RE = re.compile(r"(\d+)\s*(?:/|\s+of\s+)\s*(\d+)")

# artifact kind (as the store records it) → the file it writes in the workdir.
# Used to notice that a later cohort's input has been evicted.
_ARTIFACT_FILES = {
    "proxy": "proxy.mp4", "wav": "audio.wav", "poster": "poster.jpg",
    "sprite": "sprite.jpg", "loop": "loop.mp4", "waveform": "waveform.png",
}

# Which of those get a message of their own, in upload order.
#
# The `artifacts` pass spends fourteen seconds a video rendering a 480p
# `+faststart` proxy, a poster, a sprite sheet and a waveform — and every one of
# them lived in a working directory that `intake.evict` deletes. Four messages a
# video is what turns that spend into something Atlas can still play tomorrow,
# and the proxy alone is the difference between streaming a 40 MB original to
# show one frame and streaming two megabytes.
#
# `proxy` is first because it is the one playback needs; if a rate limit is
# going to bite, it should bite the sprite sheet. `wav` and `loop` are left out:
# the wav is an intermediate that `transcribe` re-derives and is the largest
# file of the six, and the loop is a nicety against sixty-two more messages.
_UPLOAD_ARTIFACTS = ("proxy", "poster", "sprite", "waveform")

# Between artifact uploads, and only between artifact uploads. Matches
# `assets.CHUNK_PAUSE`: the Bot API's limit is per chat, and the shard uploads
# that are this session's actual backup share that chat.
_ARTIFACT_PAUSE = float(os.environ.get("VIOS_ARTIFACT_PAUSE", "0.35") or 0.35)



def _default_base() -> str:
    try:
        import config  # noqa: PLC0415
        return getattr(config, "BASE_DIR", os.path.abspath("vios_data"))
    except Exception:
        return os.path.abspath("vios_data")


def _default_scratch(base: str) -> str:
    """Where working directories go.

    `config.SCRATCH_DIR` is the right answer on Kaggle: /kaggle/temp is a real
    mount, it is roomy, and it does not count against the 20 GB output quota.
    It is the wrong answer when the caller named its own base — an engine
    pointed at a throwaway directory should not scatter half its state into a
    Kaggle path that, off Kaggle, `config` happily creates at the drive root.

    So: honour the configured scratch only when this engine is also using the
    configured base. Otherwise scratch sits beside the base it belongs to.
    """
    try:
        import config  # noqa: PLC0415
        if os.path.abspath(base) == os.path.abspath(
                getattr(config, "BASE_DIR", "")):
            return getattr(config, "SCRATCH_DIR",
                           os.path.join(base, "_scratch"))
    except Exception:
        pass
    return os.path.join(base, "_scratch")


def _safe(key: str) -> str:
    """A video key as a directory name. Keys are shortcodes, but never trust
    a string that came from a URL with a path join."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in key)[:64]


def _observer_params(params: dict | None, without=()) -> dict:
    """The parameters an observer id is derived from, given what the run lacked.

    The observer id is documented as derived from "everything that can change the
    output — model, revision, and the parameters", and a soft input that produced
    nothing changes the output as surely as a prompt edit does. So the absence
    goes in the hash, and the two readings become two observers with two ids.

    That is not bookkeeping. A claim's uid is
    `(video, observer, channel, kind, shot, ordinal)` — deliberately not the value,
    so a shard can be replayed twice without doubling anything — and both the
    writer and the shard importer insert with `OR IGNORE`. Under one id the better
    answer from a re-run is therefore *dropped*: silently on the machine that
    earned it, and again in every database the shard reaches. Splitting the id is
    what makes re-running a thin pass worth the GPU time at all.

    An empty `without` returns exactly the dict passed in, so every observer id in
    every archive written before this existed is unchanged, and a complete run
    keeps producing the id `expected_observers` names.

    One definition, used by the writer in `_run_pass` and by the reconcile step
    that has to recognise what the writer produced — the same discipline, and for
    the same reason, as `store.observer_id` beside `Store.observer`.
    """
    out = dict(params or {})
    named = sorted({str(n) for n in (without or ()) if n})
    if named:
        out["without"] = named
    return out


_ENV_UNKNOWN: dict = {}


def _known(ids) -> list:
    """A job payload's component list, filtered to ids this build has.

    A hint is JSON that may have been written by a *different* version of this
    code — Redis outlives a `git pull`, and `recover_processing_jobs` at boot
    hands a restarted session the jobs the old one was holding. An unknown id
    would make `registry.get` raise inside a worker loop, which is a crashed lane
    over a stale string. Dropping it costs one pass one round; the planner posts
    it again under its current name.
    """
    out = []
    for cid in ids or ():
        try:
            registry.get(str(cid))
        except Exception:                              # noqa: BLE001
            continue
        out.append(str(cid))
    return out


def _env_ids(name: str) -> list:
    """A comma-separated env var as a list of component ids, unknowns dropped.

    Unknown ids are dropped rather than raised on: these variables exist to let
    a session be narrowed from a Kaggle cell, and a typo should cost the pass it
    named, not the run.

    A dropped name is still an operator error, so it is recorded in
    `_ENV_UNKNOWN` for `_narrowed` to log through the engine's own log — which
    is what shows up on the process tab. Warning to stderr here would be
    warning into a Kaggle cell that nobody reads back.
    """
    raw = os.environ.get(name, "") or ""
    want = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    known = set(registry.all_ids())
    _ENV_UNKNOWN[name] = [p for p in want if p not in known]
    return [p for p in want if p in known]


def _env_waves(name: str = "VIOS_WAVES") -> set:
    """Which waves this session runs. Default: the spine, then the refinement.

    Wave 3 is off by default for the same reason its members are — it holds the
    passes that need both cards or are not part of the archive's baseline. Set
    `VIOS_WAVES=1` for a session that only wants the searchable spine, or
    `1,2,3` for everything.
    """
    raw = (os.environ.get(name, "") or "").strip()
    if not raw:
        return {registry.WAVE_SPINE, registry.WAVE_FULL}
    out = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out or {registry.WAVE_SPINE, registry.WAVE_FULL}


class Lane:
    """One worker's own resources, so N workers are not N users of one object.

    Everything a pass touches while it runs is per-lane, and each field is here
    because sharing it across threads was either wrong or unreadable:

    * **`store` / `cov`** — its own sqlite connection, from `Store.secondary()`.
      `Store` opens `check_same_thread=False` with WAL and a 30 s busy timeout,
      so one connection *works* from five threads; what it cannot do is keep
      five implicit transactions apart. Every write method commits at its end,
      so worker B's commit lands worker A's half-written batch, and a commit
      during `export_shard`'s multi-megabyte blob reads invalidates a
      partially-consumed cursor. Per-connection rather than one global lock on
      purpose: a call site missed under a lock is silent corruption, a call site
      missed with separate connections is nothing at all.

      The **worker id stays identical** across lanes. `Coverage._mine()` scopes
      by partition and leases are owned by worker id, so two lanes with
      different ids would each think the other's claims were somebody else's.
      Lanes are threads inside one worker, not separate workers.

    * **`cache`** — its own `ModelCache`. Two cards need two copies of a
      cohort's weights; that is not waste, it is what "both GPUs busy" costs,
      and it is a disk read from the shared `MODEL_CACHE_DIR` rather than a
      download. Its log lines carry the lane name because `ModelCache.vram()`
      reads the *current* device: an unattributed leak warning is unactionable
      with two cards.

    * **`slot`** — what this lane is doing, for the tab. `self.current` was one
      dict for one video and could not describe four workers at once.

    * **`device`** — the card index this lane pinned, or -1. Set once, by the
      lane's own thread, before it claims anything.
    """

    __slots__ = ("name", "kind", "device", "store", "cov", "cache", "slot",
                 "videos", "passes", "started_at", "thread")

    def __init__(self, name: str, kind: str, device: int = -1):
        self.name = name
        self.kind = kind                 # "gpu" | "cpu" | "prep" | "cloud"
        self.device = device
        self.store: Store | None = None
        self.cov: Coverage | None = None
        self.cache: ModelCache | None = None
        self.slot: dict = {}
        self.videos = 0
        self.passes = 0
        self.started_at = 0.0
        self.thread: threading.Thread | None = None

    def busy(self) -> bool:
        return bool(self.slot.get("video_key"))

    def as_dict(self) -> dict:
        slot = dict(self.slot)
        if slot.get("since"):
            slot["elapsed"] = round(time.time() - slot["since"], 1)
        return {
            "name": self.name, "kind": self.kind,
            "device": self.device if self.device >= 0 else None,
            "busy": self.busy(), "videos": self.videos, "passes": self.passes,
            "resident": self.cache.loaded() if self.cache else [],
            "current": slot,
        }

    def close(self) -> None:
        try:
            if self.cache is not None:
                self.cache.unload_all()
        except Exception:                             # noqa: BLE001
            pass
        try:
            if self.store is not None:
                self.store.close()
        except Exception:                             # noqa: BLE001
            pass
        self.store = self.cov = None


class ProcessEngine:
    """One instance per process. Configure it, start it, watch it, stop it."""

    def __init__(self, base_dir: str | None = None):
        self.base = base_dir or _default_base()
        self.root = os.path.join(self.base, "process")
        self.db_path = os.path.join(self.root, "evidence.db")
        self.cache_dir = os.path.join(_default_scratch(self.base), "process")
        self.shard_dir = os.path.join(self.root, "shards")
        self.ledger_path = os.path.join(self.base, "capture_ledger.db")
        for d in (self.root, self.cache_dir, self.shard_dir):
            os.makedirs(d, exist_ok=True)

        self.state = IDLE
        self.message = "Not started."
        self.error = ""
        self.started_at: float | None = None
        self.worker = worker_id()

        # ── what to run ──────────────────────────────────────────────────
        self.selected: list = registry.defaults()
        self.partitions = 1
        self.index = 0
        self.follow = True
        self.sync_on_start = True
        self.restore_on_start = True
        self.publish_every = PUBLISH_EVERY
        self.vram_headroom_mb = VRAM_HEADROOM_MB
        self.cache_budget_mb = CACHE_BUDGET_MB
        self.disk_floor_mb = DISK_FLOOR_MB
        self.video_limit = 0            # 0 = the whole partition

        # ── ordering and narrowing, read once from the environment ────────
        # `VIOS_WAVES` picks which waves run, `VIOS_WAVE1` redefines the spine
        # by id, `VIOS_ONLY`/`VIOS_EXCEPT` narrow the selection. All four are
        # applied before the packer, so a narrowed session is a smaller plan
        # rather than a full plan with passes vetoed one video at a time — and
        # every one of them is a Kaggle-cell edit rather than a code change.
        self.waves_on: set = _env_waves()
        self.spine_override: list = _env_ids("VIOS_WAVE1")
        self.only: list = _env_ids("VIOS_ONLY")
        self.exclude: list = _env_ids("VIOS_EXCEPT")
        self._selection_logged = False

        # When this container last replayed shards from the channel. The boot
        # restore and `restore_on_start` are the same scan over thousands of
        # messages, and the autostart runs the first one moments before the
        # second — so the sweep asks this before repeating it. A timestamp
        # rather than a flag on the setting: the operator's choice to restore on
        # start is theirs and is not quietly rewritten, it is only not obeyed
        # twice in the same minute.
        self._shards_at = 0.0

        # What the last reconcile matched. Kept because the number is the whole
        # answer to "will it process the archive again?", and an operator asking
        # that question at hour six cannot scroll back to the boot log to find
        # it. `at=0` means no reconcile has run yet, which is a different
        # statement from "it ran and matched nothing" and is reported as such.
        self._reconciled = {"at": 0.0, "rows": 0, "videos": 0, "matched": 0,
                            "per_component": {}, "error": ""}

        # Folders searched for a reel's bytes before Telegram is asked. The
        # capture engine's own scratch is here by default, which is what lets a
        # session process what it just captured without a round trip through
        # the channel — the single biggest saving in the whole pipeline.
        self.video_dirs: list = [
            os.path.join(_default_scratch(self.base), "capture"),
            os.path.join(self.base, "videos"),
        ]

        # ── credentials ──────────────────────────────────────────────────
        # Never written by this class. Read once at construction from Kaggle
        # Secrets / the environment / the laptop file, so a session that has
        # them stored is configured before the Setup page is ever opened.
        self._tg: Telegram | None = None
        self._api_id = 0
        self._api_hash = ""
        self._hf_token = ""
        self.cred_sources: dict = {}

        # ── live state ───────────────────────────────────────────────────
        self.resources: dict = {}
        self.cohorts: list = []
        self.cohort_index = -1
        self.session = {"videos": 0, "passes": 0, "claims": 0, "vectors": 0,
                        "skipped": 0, "unavailable": 0, "failed": 0,
                        "deferred": 0, "elsewhere": 0,
                        "seconds": 0.0, "downloaded": 0, "shards": 0}
        self.last_publish: float | None = None
        self.since_publish = 0

        # ── the lanes ────────────────────────────────────────────────────
        # `self.current` used to be one dict describing one video, which is the
        # right shape for exactly one worker. It is now a property over these,
        # returning the busiest lane, so every existing reader — the process tab,
        # the browser, `_live()` — keeps working unchanged while the per-lane
        # rows appear beside it.
        self.workers: dict = {}
        self._lanes: list = []
        self._jobs = jobs.Broker(self._log)
        self._lane_error = ""

        # One video, one worker. `claim_for` already stops two lanes taking the
        # same *(video, component)*, but the CPU and cloud lanes claim *different*
        # component subsets of the same video, and two lanes inside one video
        # would share a `workdir` — the same directory `intake.evict` is deciding
        # whether to delete. Held keys are also the eviction keep-set, which fixes
        # a bug the lanes would otherwise create: `keep={_safe(key)}` is a
        # keep-set of one, and under concurrency it deletes a live working
        # directory out from under another lane.
        self._inflight: dict = {}
        self._staged: set = set()          # videos whose prep job has acked
        self._prep_posted: set = set()
        self._prep_fails: dict = {}        # key → consecutive staging failures
        # Hints a lane consumed without processing anything, so the planner can
        # offer them again. Distinct from a *failed* video, which is not re-posted
        # inside the same cohort.
        self._retry_hint: set = set()
        self._lanes_up = False
        self._sweep_lane_obj: Lane | None = None

        # Downloads stay single-file however many lanes there are. pyrogram's
        # `Channel` is not safe for concurrent downloads and the Bot API's rate
        # limit is per chat, so four lanes pulling at once is slower than one and
        # can be a ban. Extraction and inference are what parallelise.
        self._fetch_lock = threading.Lock()

        # And artifact uploads stay single-file for the mirror-image reason: the
        # Bot API's flood limit is per chat, and the shards that are this
        # session's only real backup go to the same chat. Four lanes each posting
        # four files per video would spend the budget that matters on thumbnails.
        # Separate from `_fetch_lock` so an upload never blocks a download.
        self._upload_lock = threading.Lock()

        # The cloud lane's shopping list, recomputed once per rotation from the
        # *runnable* set so the wave filter and `VIOS_ONLY`/`VIOS_EXCEPT` are
        # already applied, plus the keys already posted to it this session. The
        # posted set exists because the cloud lane is fed every planner round: a
        # video whose hints are already in the lane must not be re-posted forty
        # times while its `describe` finishes.
        self._cloud_ids: list = []
        self._cloud_posted: set = set()

        # Components deferred because this machine cannot run them, so their
        # dependents can be told the same thing instead of being retried every
        # five minutes, and so the log says it once rather than once per video.
        self._elsewhere: set = set()
        self._elsewhere_logged: set = set()

        # `(component, missing-set)` pairs already announced. A pass running
        # without an input is worth saying once; saying it per video is 224 lines
        # in an archive this size, and the row itself now carries the record.
        self._thin_logged: set = set()

        # The component ids this rotation planned. Set once per sweep; read by
        # the dependency gate to tell "not run yet" from "never selected".
        self._runnable: set = set()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.RLock()

        # ── the publisher ────────────────────────────────────────────────
        # Exporting and uploading a shard used to happen on the worker thread, so
        # a 45 MB POST stopped the GPU for as long as Telegram took to answer.
        # It runs here instead: workers say "rows written" and carry on.
        #
        # `_pub_lock` rather than `_lock`, and it guards *publishing* rather than
        # engine state. Two reasons it has to be its own lock: the status endpoint
        # reads `_lock` and must not block behind an upload, and `_publish` is now
        # reachable from three threads — the publisher, the sweep (before cutting
        # a stage bundle) and a Flask request (the manual button) — which
        # previously could interleave two exports over one watermark.
        self._pub_lock = threading.Lock()
        self._pub_wake = threading.Event()
        self._pub_thread: threading.Thread | None = None
        self._pub_note = ""
        self._pub_force = False
        self._pub_store: Store | None = None
        self._exit_flushed = False

        self._store: Store | None = None
        self._cov: Coverage | None = None
        self._cache = ModelCache(self._log)
        self._channel: intake.Channel | None = None
        self._source: intake.Source | None = None

        self._activity: deque = deque(maxlen=500)
        self._status_cache: tuple = (0.0, {})
        self._adopt_stored_credentials()

    def _adopt_stored_credentials(self) -> None:
        """Configure from the stored credentials, if there are any.

        Silent on failure: a broken secret store must not stop the tab from
        loading, since the tab is where it would be fixed.
        """
        try:
            from vios.creds import resolve  # noqa: PLC0415
            got = resolve()
        except Exception:
            return
        v = got.get("values") or {}
        if not v:
            return
        self.cred_sources = got.get("sources") or {}
        # Before configure(), not inside it. `configure` takes the credentials
        # in one call, so a bad channel id raises in the Telegram branch and the
        # HF token two branches below is never reached — and the `except` here
        # then hides that from the log. The environment bridge has no such
        # coupling and is idempotent, so running it first means one broken
        # credential can no longer silently disable an unrelated pass.
        try:
            _creds.export_to_env()
        except Exception as exc:            # noqa: BLE001
            self._log(f"credential bridge failed: {type(exc).__name__}", "warn")
        try:
            self.configure(
                bot_token=v.get("bot_token", ""),
                channel_id=v.get("channel_id") or None,
                api_id=int(v.get("api_id") or 0),
                api_hash=v.get("api_hash", ""),
                hf_token=v.get("hf_token", ""))
            if self.cred_sources:
                where = sorted(set(self.cred_sources.values()))
                self._log(f"credentials loaded from {', '.join(where)}")
        except Exception as exc:            # noqa: BLE001
            # Named, not swallowed. This ran silently for a whole session and
            # the only symptom was a pass declining for a reason that was not
            # true.
            self.cred_sources = {}
            self._log(f"stored credentials rejected: "
                      f"{type(exc).__name__}: {exc}", "warn")

    # ══════════════════════════════════════════════════════════════════════
    # Resources this engine owns
    # ══════════════════════════════════════════════════════════════════════

    @property
    def store(self) -> Store:
        with self._lock:
            if self._store is None:
                self._store = Store(self.db_path)
            return self._store

    @property
    def coverage(self) -> Coverage:
        with self._lock:
            if self._cov is None:
                self._cov = Coverage(self.store.conn, self.partitions,
                                     self.index, self.worker)
            return self._cov

    # ── what one worker looked like, when there was one ──────────────────
    @property
    def current(self) -> dict:
        """The busiest lane's slot, so every existing reader keeps working.

        Before the lanes this was a plain attribute holding one video. Making it
        a property rather than deleting it is deliberate: the process tab, the
        browser's live panel, `_note`, `_progress` and `_live` all read
        `current`, and a shape change there is a UI rewrite for no gain. The
        per-lane rows are added *beside* it in the status payload.

        "Busiest" is the lane that has held its video longest — the one an
        operator watching a single line most wants to see, because it is the one
        closest to being stuck.
        """
        lanes = list(self._lanes)
        if self._sweep_lane_obj is not None:
            # The `VIOS_LANES=0` path has no entry in `_lanes` — it runs on the
            # sweep thread — and leaving it out here would blank the live panel
            # for exactly the configuration an operator falls back to when
            # something is wrong.
            lanes.append(self._sweep_lane_obj)
        best, oldest = {}, None
        for lane in lanes:
            slot = lane.slot
            if not slot.get("video_key"):
                continue
            since = slot.get("since") or 0.0
            if oldest is None or since < oldest:
                best, oldest = slot, since
        return dict(best)

    def _bump(self, field: str, n=1) -> None:
        """One session counter, atomically.

        `self.session[k] += 1` is a read-modify-write, and with four lanes doing
        it the tab's totals drift low by however many increments interleaved.
        Every mutation goes through here so there is one place holding the lock
        rather than fourteen call sites each remembering to.
        """
        with self._lock:
            self.session[field] = self.session.get(field, 0) + n

    # ── one video, one worker ────────────────────────────────────────────
    @contextlib.contextmanager
    def _hold(self, key: str, lane: Lane):
        """Take exclusive hold of a video, or yield False.

        The unit that must not be shared is the *working directory*, not the
        (video, component) pair — `claim_for` already guards that. Two lanes
        inside one video would write `frames/`, `proxy.mp4` and `audio.wav`
        underneath each other, and `intake.evict` would be deciding whether to
        delete a directory another lane is mid-pass in.

        Held keys double as the eviction keep-set for exactly that reason.
        """
        with self._lock:
            if key in self._inflight:
                got = False
            else:
                self._inflight[key] = lane.name
                got = True
        try:
            yield got
        finally:
            if got:
                with self._lock:
                    self._inflight.pop(key, None)

    def _held(self) -> set:
        """Every working directory a lane is inside right now."""
        with self._lock:
            return {_safe(k) for k in self._inflight}

    def _log(self, text: str, level: str = "info") -> None:
        """One activity line, and stdout too when it is a fault.

        The ring buffer holds 500 entries and dies with the process, so it is
        the wrong and only place for an error to live. Warnings and errors are
        also printed, which on Kaggle means they land in the cell output that
        outlives the session — the log someone actually reads when asking why
        a twelve-hour run stopped producing.
        """
        self._activity.append({"at": time.time(), "level": level,
                               "text": str(text)[:400]})
        if level in ("warn", "error"):
            print(f"[process] {level}: {str(text)[:400]}", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # Configuration
    # ══════════════════════════════════════════════════════════════════════

    def configure(self, bot_token: str = "", channel_id=None, api_id=0,
                  api_hash: str = "", hf_token: str = "",
                  components=None, partitions: int | None = None,
                  index: int | None = None, follow: bool | None = None,
                  publish_every: int | None = None,
                  vram_headroom_mb: int | None = None,
                  cache_budget_mb: int | None = None,
                  video_limit: int | None = None,
                  sync_on_start: bool | None = None,
                  restore_on_start: bool | None = None,
                  ledger_path: str = "", video_dirs=None) -> dict:
        """Accept settings from the tab. Blank fields keep their value.

        Credentials live here and only here — instance attributes on a running
        process, never written to a file, a notebook cell, or the database.
        Re-typing them after a restart is the intended cost.

        "Here and only here" was true too literally. `self._tg` is what the
        uploader reads, and nothing else in the process reads it: `db_restore`,
        `tg_transport`, the harvester and Atlas all read `os.environ`. So a user
        whose Phase 0 was throttled, who then typed four correct credentials
        into this tab, still got "Telegram is not configured" out of restore —
        the values were in this object and nowhere the restore could see. They
        are now bridged with `creds.adopt` (environment only, still never disk),
        the same way `hf_token` has always been bridged for pyannote below.
        """
        with self._lock:
            # First, before anything that can raise. `Telegram(...)` below
            # rejects a malformed channel id, and the caller's `except` then
            # discards the three credentials that were fine — which is the same
            # coupling `_adopt_stored_credentials` already works around by
            # bridging before it configures. Writing the environment costs
            # nothing and is idempotent, so it goes where it cannot be skipped.
            #
            # Typed overrides what is there — that is what typing it means — and
            # a blank field is skipped rather than written, so submitting this
            # form with one field filled cannot unset the other three.
            _adopted = _creds.adopt({"bot_token": bot_token,
                                     "channel_id": channel_id,
                                     "api_id": api_id,
                                     "api_hash": api_hash})
            if _adopted:
                self._log(f"credentials accepted: "
                          f"{', '.join(sorted(_adopted.values()))}")

            if bot_token or channel_id:
                token = bot_token or (self._tg.token if self._tg else "")
                chan = channel_id or (self._tg.channel if self._tg else None)
                self._api_id = int(api_id or self._api_id or 0)
                self._api_hash = (api_hash or self._api_hash or "").strip()
                self._tg = Telegram(token, chan, self._api_id, self._api_hash)
            elif api_id or api_hash:
                self._api_id = int(api_id or self._api_id or 0)
                self._api_hash = (api_hash or self._api_hash or "").strip()
                if self._tg:
                    self._tg.api_id = self._api_id
                    self._tg.api_hash = self._api_hash

            if hf_token and hf_token.strip():
                self._hf_token = hf_token.strip()
                # huggingface_hub reads the environment directly when it
                # resolves a gated repo, so passing it through job.params
                # alone would not be enough for pyannote. The list of names it
                # might read lives in creds.MIRROR, next to the boot bridge that
                # writes the same names, because two copies of it is how the
                # newest one (HUGGING_FACE_HUB_TOKEN) came to be missing here.
                os.environ["VIOS_HF_TOKEN"] = self._hf_token
                for _label in _creds.MIRROR.get("hf_token", ()):
                    os.environ.setdefault(_label, self._hf_token)

            if components is not None:
                known = set(registry.all_ids())
                picked = [c for c in components if c in known]
                if picked:
                    self.selected = picked
            if partitions is not None or index is not None:
                p = max(1, int(partitions if partitions is not None
                               else self.partitions))
                i = int(index if index is not None else self.index)
                if not (0 <= i < p):
                    raise ValueError(
                        f"worker {i} of {p} is not a valid slice — the index "
                        f"must be between 0 and {p - 1}")
                self.partitions, self.index = p, i
                self._cov = None            # rebuilt with the new slice
            if follow is not None:
                self.follow = bool(follow)
            if sync_on_start is not None:
                self.sync_on_start = bool(sync_on_start)
            if restore_on_start is not None:
                self.restore_on_start = bool(restore_on_start)
            if publish_every:
                self.publish_every = max(1, int(publish_every))
            if vram_headroom_mb is not None:
                self.vram_headroom_mb = max(256, int(vram_headroom_mb))
            if cache_budget_mb is not None:
                self.cache_budget_mb = max(1000, int(cache_budget_mb))
            if video_limit is not None:
                self.video_limit = max(0, int(video_limit))
            if ledger_path and ledger_path.strip():
                self.ledger_path = ledger_path.strip()
            if video_dirs is not None:
                self.video_dirs = [d.strip() for d in video_dirs if d.strip()]
        return self.settings()

    def settings(self) -> dict:
        """Everything the form needs to redraw itself. No secrets leave here —
        a token's presence is a boolean, which is all the interface has ever
        needed to know."""
        try:
            from vios.creds import describe  # noqa: PLC0415
            stored = describe()
        except Exception:
            stored = {}
        return {
            "bot_token_set": bool(self._tg and self._tg.token),
            "api_id_set": bool(self._api_id),
            "api_hash_set": bool(self._api_hash),
            "hf_token_set": bool(self._hf_token),
            "channel_id": (self._tg.channel if self._tg else None),
            "credential_sources": dict(self.cred_sources),
            "stored_credentials": stored,
            "components": list(self.selected),
            "partitions": self.partitions,
            "index": self.index,
            "follow": self.follow,
            "sync_on_start": self.sync_on_start,
            "restore_on_start": self.restore_on_start,
            "publish_every": self.publish_every,
            "vram_headroom_mb": self.vram_headroom_mb,
            "cache_budget_mb": self.cache_budget_mb,
            "video_limit": self.video_limit,
            "ledger_path": self.ledger_path,
            "video_dirs": list(self.video_dirs),
            "db_path": self.db_path,
            "cache_dir": self.cache_dir,
            "worker": self.worker,
        }

    # ══════════════════════════════════════════════════════════════════════
    # Preflight
    # ══════════════════════════════════════════════════════════════════════

    def preflight(self) -> dict:
        """Everything that could stop this run, checked before it starts.

        Split into blocking and advisory on purpose. Missing ffmpeg is
        blocking: nothing downstream works without a proxy. A missing
        `paddleocr` is not — it costs one of thirty passes, and the honest
        response is to run the other twenty-nine and say so in the tab.
        """
        checks, blocking = [], []

        def ok(name, good, detail, block=False):
            checks.append({"name": name, "ok": bool(good), "detail": detail})
            if block and not good:
                blocking.append(f"{name}: {detail}")

        # ── the catalogue itself ─────────────────────────────────────────
        problems = registry.validate()
        ok("Component catalogue", not problems,
           "; ".join(problems) if problems else
           f"{len(registry.all_ids())} components, no structural problems",
           block=True)
        gaps = missing_runners()
        ok("Runners wired", not gaps,
           f"no function for {', '.join(gaps)}" if gaps else
           "every component resolves to a function", block=True)

        # ── media tooling ────────────────────────────────────────────────
        ok("ffmpeg", media.have_ffmpeg(),
           "on PATH" if media.have_ffmpeg() else
           "not found — install ffmpeg, nothing can be decoded without it",
           block=True)

        # ── the store ────────────────────────────────────────────────────
        try:
            stats = self.store.stats()
            ok("Evidence store", True,
               f"{self.db_path} · {stats.get('videos', 0)} videos, "
               f"{stats.get('claims', 0)} claims")
        except Exception as exc:
            stats = {}
            ok("Evidence store", False, f"{type(exc).__name__}: {exc}",
               block=True)

        # ── something to process ─────────────────────────────────────────
        synced = intake.sync(self.store, self.ledger_path) \
            if os.path.exists(self.ledger_path) else {"seen": 0}
        # A folder that was configured but never adopted is picked up here, so
        # "point the engine at a folder and press Start" is one step, not two.
        # The ledger goes in with it: a file named `10.mp4` is a Telegram message
        # id, and the ledger is what turns that back into the reel's shortcode
        # instead of adopting the number as a second identity for a video the
        # sync above already registered.
        adopted, merged = 0, 0
        for d in self.video_dirs:
            if os.path.isdir(d):
                try:
                    got = intake.adopt_folder(self.store, d,
                                              ledger_path=self.ledger_path)
                    adopted += got.get("added", 0)
                    merged += got.get("merged", 0)
                    if got.get("merged") or got.get("refused"):
                        self._log(
                            f"folder {d}: {got.get('added', 0)} adopted, "
                            f"{got.get('merged', 0)} recognised as videos "
                            f"already here, {got.get('refused', 0)} refused "
                            f"— by {got.get('by', {})}")
                except Exception as exc:
                    self._log(f"folder {d}: {type(exc).__name__}: {exc}", "warn")
        have = len(self.store.video_keys())
        ok("Videos to process", have > 0,
           f"{have} in the evidence store, {synced.get('seen', 0)} uploaded in "
           f"the capture ledger"
           + (f", {adopted} adopted from disk" if adopted else "")
           + (f", {merged} folder file(s) matched to videos already here"
              if merged else "") if have else
           f"nothing yet — capture some reels first, point this at an existing "
           f"ledger ({self.ledger_path}), or give it a folder of videos",
           block=True)

        # ── hardware ─────────────────────────────────────────────────────
        # Before the disk is measured, give back what is provably redundant.
        # Kaggle's working directory survives a kernel restart, so a resumed
        # session inherits every shard file the run before it left behind — and
        # until `_drop_shard_file` existed that was all of them, published or not.
        back = self._reclaim_shards()
        if back["removed"]:
            self._log(f"Reclaimed {back['bytes'] / 1048576:.1f} MB from "
                      f"{back['removed']} shard file(s) the channel already has")
        res = resources.probe(self.cache_dir)
        self.resources = res
        ok("Hardware", True, resources.describe(res))
        ok("Scratch disk", res["disk_free_mb"] > self.disk_floor_mb,
           f"{res['disk_free_mb'] / 1024:.1f} GB free on {self.cache_dir}"
           + (f", {back['bytes'] / 1048576:.1f} MB of it reclaimed from "
              f"{back['removed']} published shard(s)" if back["removed"]
              else ""))

        # ── selection ────────────────────────────────────────────────────
        sel = list(self.selected)
        cannot = registry.unrunnable(sel, res)
        ok("Selected passes fit", not cannot,
           "; ".join(f"{k}: {v}" for k, v in cannot.items()) if cannot else
           f"{len(sel)} passes, all within {res['usable_vram_mb']} MB per card")
        gaps = registry.missing_dependencies(sel)
        ok("Dependencies selected", not gaps,
           "; ".join(f"{k} needs {', '.join(v)}" for k, v in gaps.items())
           if gaps else "every prerequisite is in the selection")

        # ── python packages ──────────────────────────────────────────────
        absent = self._missing_packages(sel)
        ok("Python packages", not absent,
           "; ".join(f"{mod} → {', '.join(ids)}"
                     for mod, ids in absent.items()) if absent else
           "every selected pass can import what it needs")

        # ── model ids ────────────────────────────────────────────────────
        # One request per distinct id, before a single weight is fetched. The
        # session this was added for spent its whole language stage failing on a
        # model id nobody had published, once per video per card, because a load
        # fault is a per-video failure and there was nowhere for "that id is
        # wrong" to be said. Non-blocking on purpose: it cannot tell an id that
        # is absent from one it merely could not reach, and a run with a warm
        # cache and no network is a run that should still start.
        try:
            unresolved = registry.unresolvable_models(sel)
        except Exception as exc:                               # noqa: BLE001
            unresolved = {}
            self._log(f"model id preflight skipped — {type(exc).__name__}: "
                      f"{exc}", "warn")
        ok("Model ids resolve", not unresolved,
           "; ".join(f"{cid}: {why}" for cid, why in unresolved.items())
           if unresolved else
           "every selected pass names a model the Hub has")

        # ── telegram ─────────────────────────────────────────────────────
        if self._tg and self._tg.token:
            try:
                # Not `getMe` — a bot can read a channel it cannot post to, and
                # discovering that after eleven hours of inference, with the
                # shard upload failing, is the expensive way to find out.
                p = self._tg.probe()
                ok("Telegram bot", p["ok"],
                   f"@{p['bot']} · {p['channel']} · can post"
                   if p["ok"] else p["error"] or "cannot post to the channel")
            except Exception as exc:
                ok("Telegram bot", False, f"{type(exc).__name__}: {exc}")
            ok("MTProto", bool(self._api_id and self._api_hash),
               "API id and hash present — large files and capture records are "
               "reachable" if self._api_id else
               "no API id — files over 20 MB and all capture records will be "
               "unavailable")
        else:
            ok("Telegram bot", False,
               "no bot token — the engine will run and write locally, but "
               "nothing will be published and no source video can be fetched")

        needs_hf = [c for c in sel if "pyannote" in
                    " ".join(registry.get(c).requires)]
        if needs_hf:
            ok("Hugging Face token", bool(self._hf_token),
               "present" if self._hf_token else
               f"absent — {', '.join(needs_hf)} needs one and will be skipped")

        return {"ok": not blocking, "checks": checks, "blocking": blocking,
                "resources": res, "stats": stats}

    @staticmethod
    def _missing_packages(ids) -> dict:
        """module name → the components that wanted it."""
        import importlib.util  # noqa: PLC0415
        out: dict = {}
        seen: dict = {}
        for cid in ids:
            for mod in registry.get(cid).requires:
                if mod not in seen:
                    try:
                        seen[mod] = importlib.util.find_spec(
                            mod.split(".")[0]) is not None
                    except Exception:
                        seen[mod] = False
                if not seen[mod]:
                    out.setdefault(mod, []).append(cid)
        return out

    # ══════════════════════════════════════════════════════════════════════
    # The plan, before anything runs
    # ══════════════════════════════════════════════════════════════════════

    def plan(self) -> dict:
        """What this machine would do, without doing it. Drives the tab's
        cohort strip and the 'what am I about to start' numbers."""
        res = self.resources or resources.probe(self.cache_dir)
        sel = list(self.selected)
        cannot = registry.unrunnable(sel, res)
        runnable = [c for c in sel if c not in cannot]
        budget = max(res.get("usable_vram_mb", 0) - self.vram_headroom_mb, 512)
        cohorts = registry.plan_cohorts(
            runnable, budget, max(res.get("gpu_count", 0), 1),
            res.get("disk_free_mb", 0))
        try:
            videos = len(self.store.video_keys())
        except Exception:
            videos = 0
        mine = videos if self.partitions == 1 else max(
            videos // self.partitions, 0)
        rows = [c.as_dict(mine) for c in cohorts]

        # The Run tab's rotation strip reads `self.cohorts`, which a sweep owns
        # while it is running — `cohort_index` points into it. So seed it only
        # when nobody is sweeping. Without this the strip stays empty until the
        # first start, which is precisely the moment an operator most wants to
        # see the rotation they are about to commit twelve hours to.
        with self._lock:
            if self.state not in (RUNNING, PAUSED, STOPPING):
                self.resources = res
                self.cohorts = rows
                self.cohort_index = -1

        return {
            "cohorts": rows,
            "unrunnable": cannot,
            "estimate": registry.estimate(runnable, mine,
                                          max(res.get("gpu_count", 0), 1)),
            "videos": videos,
            "my_videos": mine,
            "vram_budget_mb": budget,
            "resources": res,
        }

    def catalog(self) -> list:
        return registry.rows(self.selected)

    # ══════════════════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════════════════

    def start(self) -> dict:
        with self._lock:
            if self.state in (RUNNING, PAUSED, STOPPING):
                return {"ok": False, "error": f"already {self.state}"}
            pre = self.preflight()
            if not pre["ok"]:
                self.state = ERROR
                self.error = "; ".join(pre["blocking"])
                self.message = "Preflight failed."
                return {"ok": False, "error": self.error, "preflight": pre}

            self._stop.clear()
            self._pause.clear()
            self.error = ""
            self.state = RUNNING
            self.message = "Starting."
            self.started_at = time.time()
            self.session = {"videos": 0, "passes": 0, "claims": 0,
                            "vectors": 0, "skipped": 0, "unavailable": 0,
                            "failed": 0, "deferred": 0, "elsewhere": 0,
                            "seconds": 0.0, "downloaded": 0, "shards": 0}
            self._elsewhere_logged = set()
            self._thin_logged = set()
            self._exit_flushed = False
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="vios-process")
            self._thread.start()
            self._start_publisher()
            self._log("Engine started")
            return {"ok": True}

    def pause(self) -> dict:
        with self._lock:
            if self.state != RUNNING:
                return {"ok": False, "error": f"not running ({self.state})"}
            self._pause.set()
            self.state = PAUSED
            self.message = "Paused — finishing the pass in flight."
            self._log("Paused")
            return {"ok": True}

    def resume(self) -> dict:
        with self._lock:
            if self.state != PAUSED:
                return {"ok": False, "error": f"not paused ({self.state})"}
            self._pause.clear()
            self.state = RUNNING
            self.message = "Resumed."
            self._log("Resumed")
            return {"ok": True}

    def stop(self) -> dict:
        with self._lock:
            if self.state not in (RUNNING, PAUSED):
                return {"ok": False, "error": f"not running ({self.state})"}
            self._stop.set()
            self._pause.clear()
            self.state = STOPPING
            self.message = ("Stopping — the pass in flight will finish and be "
                            "recorded.")
            self._log("Stop requested")
            return {"ok": True}

    def shutdown(self) -> None:
        self._stop.set()
        self._pause.clear()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=120)
        # Lanes before the publisher, because a lane still mid-pass is still
        # writing rows the publisher has to carry. `_run`'s `finally` has usually
        # done this already; both paths are idempotent, and this one covers a
        # sweep thread that outlived its join.
        self._stop_lanes()
        # The publisher last, and synchronously: whatever the sweep wrote in the
        # seconds before the stop landed is only real once it is in the channel.
        self._drain_publisher("shutdown")
        self._teardown()
        with self._lock:
            if self._pub_store is not None:
                try:
                    self._pub_store.close()
                except Exception:               # noqa: BLE001
                    pass
                self._pub_store = None
            if self._store is not None:
                self._store.checkpoint()
                self._store.close()
                self._store = None
                self._cov = None

    # ── cooperative stopping ─────────────────────────────────────────────
    def _stopping(self) -> bool:
        return self._stop.is_set()

    def _wait_if_paused(self) -> None:
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════════════
    # The sweep
    # ══════════════════════════════════════════════════════════════════════

    def _run(self) -> None:
        try:
            self._sweep()
        except Exception as exc:
            with self._lock:
                self.state = ERROR
                self.error = f"{type(exc).__name__}: {exc}"
                self.message = "The engine stopped on an error."
            self._log(f"CRASH {self.error}", "error")
            self._log(traceback.format_exc()[-900:], "error")
        finally:
            # Every way out of the sweep ends here: a clean stop, the early
            # return of a one-shot run, and a crash. All three have rows the
            # publisher was told about and may not have sent yet, and a crashed
            # sweep is exactly when losing them would hurt most. Idempotent — the
            # normal path already drained, and a second drain finds nothing.
            #
            # Lanes first, publisher second, teardown last, and the order is
            # load-bearing: a lane still running writes rows the publisher must
            # see, and `_teardown` checkpoints the connection those rows are in.
            self._stop_lanes()
            self._drain_publisher("sweep-end")
            self._teardown()
            with self._lock:
                if self.state not in (ERROR,):
                    self.state = IDLE
                    self.message = (self.message if self._stop.is_set()
                                    else self.message) or "Stopped."
                self.cohort_index = -1
            if self._sweep_lane_obj is not None:
                self._sweep_lane_obj.slot = {}

    def _sweep(self) -> None:
        store, cov = self.store, self.coverage

        if self.sync_on_start:
            self.message = "Reading the capture ledger."
            got = intake.sync(store, self.ledger_path)
            self._log(f"Capture ledger: {got.get('seen', 0)} uploaded, "
                      f"{got.get('added', 0)} new"
                      + (f" — {got['reason']}" if got.get("reason") else ""))

        self._open_channel()
        if (self.restore_on_start and self._channel and self._channel.ready
                and time.time() - self._shards_at > SHARDS_FRESH_SECONDS):
            self.message = "Replaying evidence shards from the channel."
            self._log("Scanning the channel for evidence shards")
            got = intake.restore_shards(
                store, self._tg, self._channel,
                on_progress=lambda seen, head, n: setattr(
                    self, "message",
                    f"Replaying shards — {seen}/{head} messages, {n} imported"),
                should_stop=self._stopping,
                ledger_path=self.ledger_path)
            self._shards_at = time.time()
            self._log(f"Shards: {got.get('imported', 0)} imported, "
                      f"{got.get('skipped', 0)} already held, "
                      f"{got.get('claims', 0)} claims recovered"
                      + (f" — {got['reason']}" if got.get("reason") else ""))
            # Said out loud because it is the archive's own history arriving: an
            # older build named some videos after Telegram message ids, and this
            # is the count of evidence rows put back under the shortcode they
            # describe. `refused` is what nothing could place.
            if got.get("rehomed"):
                self._log(f"Shards: {got['rehomed']} record(s) rehomed onto the "
                          f"video's real identity")
            if got.get("refused"):
                self._log(f"Shards: {got['refused']} record(s) refused — the key "
                          f"is not a video identity and nothing could resolve it",
                          "warn")
            for e in got.get("errors", [])[:5]:
                self._log(f"shard: {e}", "warn")
        elif self.restore_on_start and self._shards_at:
            self._log("Shards were replayed moments ago by the boot restore — "
                      "not scanning the channel twice")

        self._source = intake.Source(self._tg, self._channel, self._log,
                                     local_dirs=self.video_dirs)

        while not self._stopping():
            res = resources.probe(self.cache_dir)
            with self._lock:
                self.resources = res
            self._log(resources.describe(res))

            # The lanes, once, now that the card count is known. Everything below
            # is written to work with `self._lanes` empty, so a machine where this
            # decides on zero lanes runs the original single-threaded sweep with
            # no second code path to keep in step.
            self._start_lanes(res)

            sel = self._narrowed()
            cannot = registry.unrunnable(sel, res)
            for cid, why in cannot.items():
                self._log(f"{cid} cannot run here: {why}", "warn")
            runnable = [c for c in sel if c not in cannot]
            if not runnable:
                raise RuntimeError(
                    "no selected pass can run on this machine — "
                    + "; ".join(f"{k}: {v}" for k, v in cannot.items()))

            # Which components are the spine, and which waves this session runs.
            # Closing an operator's `VIOS_WAVE1` over its own needs is what keeps
            # a hand-written spine safe: a wave whose members need something from
            # a later wave runs passes with nothing to read, and that failure is
            # invisible in the output.
            spine = None
            if self.spine_override:
                spine = registry.close_needs(self.spine_override, runnable)
                pulled = sorted(set(spine) - set(self.spine_override))
                if pulled and not self._selection_logged:
                    self._log(f"VIOS_WAVE1 also pulls in {', '.join(pulled)} — "
                              f"needed by the spine you named")
            runnable = [cid for w, group in registry.waves(runnable, spine)
                        if w in self.waves_on for cid in group]
            if not runnable:
                raise RuntimeError(
                    f"VIOS_WAVES={','.join(str(w) for w in sorted(self.waves_on))} "
                    f"selects no pass that can run here")
            leaks = registry.wave_leaks(runnable, spine)
            for cid, need, w in leaks:
                self._log(f"{cid} needs {need}, which is in wave {w} — a later "
                          f"wave than its own; it will run without it", "warn")
            self._selection_logged = True

            cov.plan(runnable)
            # What "in the plan" means to the dependency gate below. A hard need
            # inside this set is work this session intends to do, so a pass that
            # finds it not yet run waits for it; a hard need outside the set was
            # never selected, and waiting for it would be waiting forever.
            with self._lock:
                self._runnable = set(runnable)
            reclaimed = cov.reclaim_stale()
            if reclaimed:
                self._log(f"Reclaimed {reclaimed} stale leases from a session "
                          f"that did not shut down")

            revived = cov.revive_failed()
            if revived["revived"]:
                self._log(f"Auto-revived {revived['revived']} failed rows "
                          f"(exhausted: {revived['exhausted']})")

            # Rows an earlier build declined for something the video never said.
            # Once per database, because a skip is terminal on purpose and this
            # does not change that rule — it repairs the rows written before
            # `PassUnavailable` existed, when a missing backend and an absent
            # audio track went to the same place. The marker lives in `meta`
            # rather than in a config flag so a resumed session, a second worker
            # and a fresh clone all agree it has already happened; the Kaggle
            # cell clones from scratch with nobody watching, and the Process
            # tab's Requeue button posts `state:'failed'` (process_ui.html:1451),
            # so there is no interactive route to a skipped row at all.
            if store.get_meta(RECLAIM_MARKER) != "1":
                try:
                    back = cov.reclaim_unavailable()
                    store.set_meta(RECLAIM_MARKER, "1")
                    if back["requeued"]:
                        named = ", ".join(
                            f"{c} ×{n}" for c, n in
                            sorted(back["components"].items(),
                                   key=lambda kv: -kv[1])[:4])
                        self._log(f"Requeued {back['requeued']} rows that an "
                                  f"earlier build declined for a missing "
                                  f"package, model or key rather than for "
                                  f"anything about the video — {named}")
                    else:
                        self._log("No rows were declined for a missing package, "
                                  "model or key; nothing to hand back")
                except Exception as exc:               # noqa: BLE001
                    # Never a reason to refuse to process: the worst case is
                    # that the repair is attempted again next boot.
                    self._log(f"Could not reclaim declined rows, continuing: "
                              f"{type(exc).__name__}: {exc}", "warn")

            # The restore above put evidence back without touching the work
            # table, and `plan` has just created a queued row for every pair it
            # covers. This is the moment the two disagree, and the only moment
            # both halves exist to be compared — reconciling before `plan` would
            # update rows that are not there yet, and reconciling in the caller
            # would cover the autostart path and miss the Start button.
            #
            # Repeated once per rotation, not once per session: `plan` adds rows
            # whenever a newly captured video arrives or a component is enabled,
            # and a shard restored by a *different* worker can land at any time.
            # The scan is four grouped counts over indexed columns.
            try:
                with self._lock:
                    self.message = "Matching restored evidence to the work table."
                self.reconcile_now(runnable)
            except Exception as exc:               # noqa: BLE001
                # A reconcile that fails costs repeated work and nothing else:
                # the passes it would have marked done simply run again and
                # collapse on their uids. Never a reason to refuse to process.
                self._log(f"Reconcile failed, continuing: "
                          f"{type(exc).__name__}: {exc}", "warn")
                with self._lock:
                    self._reconciled = {
                        **self._reconciled, "at": time.time(),
                        "error": f"{type(exc).__name__}: {exc}"[:300]}

            # After reconcile, and that order is the whole of it. A pass that ran
            # without its on-screen text carries the names of what it lacked, and
            # this hands the row back once one of those inputs has finished *and*
            # written something. Reconcile is what turns restored evidence into a
            # `done` row with a claim count — so running this first would ask
            # whether OCR had arrived while OCR's row still said `queued`, and
            # answer no on exactly the rotation where the answer was yes.
            #
            # Scoped to `runnable`: this converts a usable answer into a queued
            # row, and doing that for a pass the session will not claim would
            # leave a stage pending with nothing able to finish it.
            try:
                again = cov.reoffer_degraded(runnable)
                if again["reoffered"]:
                    named = ", ".join(
                        f"{c} ×{n}" for c, n in
                        sorted(again["components"].items(),
                               key=lambda kv: -kv[1])[:4])
                    self._log(f"Re-offering {again['reoffered']} completed rows "
                              f"that answered without an input which has since "
                              f"produced evidence — {named}")
            except Exception as exc:               # noqa: BLE001
                # The row already holds a usable answer; the worst case of this
                # failing is that it stays the thinner one until next rotation.
                self._log(f"Could not re-offer thin rows, continuing: "
                          f"{type(exc).__name__}: {exc}", "warn")

            budget = max(res.get("usable_vram_mb", 0) - self.vram_headroom_mb,
                         512)
            # Cohorts are packed per wave rather than once over everything, and
            # that is the whole of the ordering change. The barrier is unaffected:
            # `plan_cohorts` still packs first-fit over `topo_sort`, so a cohort's
            # outputs are still complete and uploaded before the next begins.
            # Waves sit above cohorts and make the boundary coarser, not weaker —
            # wave 1 finishes across every video, publishes, and only then does
            # the wave that refines the same database start.
            #
            # Cohort indices are renumbered across the whole rotation so that a
            # publish label, a log line and `self.cohort_index` each still name
            # exactly one cohort.
            plan: list = []
            for wv, group in registry.waves(runnable, spine):
                cs = registry.plan_cohorts(
                    group, budget, max(res.get("gpu_count", 0), 1),
                    res.get("disk_free_mb", 0))
                for c in cs:
                    c.index += sum(len(p[2]) for p in plan)
                plan.append((wv, group, cs))

            cohorts = [c for _w, _g, cs in plan for c in cs]
            rows: list = []
            for wv, group, cs in plan:
                for c in cs:
                    rows.append({**c.as_dict(), "wave": wv,
                                 "wave_name": registry.WAVE_NAMES.get(
                                     wv, str(wv))})
            with self._lock:
                self.cohorts = rows
            self._log(f"Plan: {len(plan)} waves, {len(cohorts)} cohorts over "
                      f"{len(runnable)} passes, {budget} MB per card")
            for wv, group, cs in plan:
                self._log(f"  wave {wv} ({registry.WAVE_NAMES.get(wv, wv)}): "
                          f"{len(group)} passes in {len(cs)} cohorts, "
                          f"{sum(registry.get(i).seconds for i in group):.0f} s "
                          f"per video")
            self._preflight_cloud(runnable)

            # What the cloud lane is allowed to take, for this rotation. Derived
            # from `runnable` rather than from the catalogue so the wave filter
            # and `VIOS_ONLY`/`VIOS_EXCEPT` are already applied — the lane crosses
            # the cohort barrier, but never the operator's selection.
            with self._lock:
                self._cloud_ids = [c for c in runnable
                                   if registry.get(c).family == "cloud"]

            worked = 0
            preempted = False
            for wv, group, cs in plan:
                if self._stopping() or preempted:
                    break
                for cohort in cs:
                    if self._stopping():
                        break
                    # Wave 1 has absolute priority, permanently. A video posted
                    # to the channel during wave 2 gets its spine before wave 2
                    # continues — the rotation restarts at wave 1 and the
                    # ordering self-heals, with no special case for late
                    # arrivals anywhere in the loop.
                    if self._earlier_wave_waiting(plan, wv, runnable):
                        preempted = True
                        break
                    worked += self._run_cohort(cohort, wv)
                    self._unload(f"cohort {cohort.index} complete")
                    # Forced, because a cohort boundary is a checkpoint worth
                    # taking whatever the counters say — and still not worth
                    # making a card wait for. The publisher has it from here.
                    self._rows_written(f"cohort-{cohort.index}", force=True)
                    self._retry_pending_stages(runnable)
                if preempted or self._stopping():
                    break
                # The wave finished across every video. That is a database in
                # which a whole layer is complete rather than a cohort boundary
                # nobody outside this loop can interpret, so it is the boundary
                # the snapshot bundle is cut on.
                self._publish_stage(registry.WAVE_NAMES.get(wv, str(wv)), group)

            if self._stopping():
                break
            # A preemption means an earlier wave has work waiting *right now*,
            # so the rotation restarts immediately. Falling into the idle branch
            # here would sleep on work we have just been told exists — the one
            # case where "nothing ran this rotation" does not mean "nothing to
            # run".
            if worked == 0 and not preempted:
                deferred = cov.deferred()
                if deferred["rows"] and self.follow:
                    due = deferred["due_at"]
                    wait = max(int(due - time.time()), 60) if due else IDLE_POLL_SECONDS
                    with self._lock:
                        self.message = (f"{deferred['rows']} rows in backoff — "
                                        f"next batch due in {wait}s.")
                    self._log(f"{deferred['rows']} rows deferred, "
                              f"checking back in {wait}s")
                    self._sleep(wait)
                    if self.sync_on_start:
                        intake.sync(store, self.ledger_path)
                    continue

                if not self.follow:
                    with self._lock:
                        self.message = (
                            "Sweep complete — every selected pass has run on "
                            "every video in this slice."
                            if not deferred["rows"] else
                            f"Sweep complete for now — {deferred['rows']} rows "
                            f"are in a retry backoff and will be picked up by "
                            f"the next run.")
                    self._log("Sweep complete"
                              + (f"; {deferred['rows']} rows deferred to a "
                                 f"later run" if deferred["rows"] else ""))
                    # The sweep is the point at which a stage's standing is
                    # final for this run, so any stage whose bundle never went
                    # up gets its last chance here rather than next session.
                    self._retry_pending_stages(runnable)
                    return
                with self._lock:
                    self.message = ("Caught up. Watching for new captures — "
                                    f"next check in {IDLE_POLL_SECONDS}s.")
                self._log("Caught up; watching for new captures")
                self._sleep(IDLE_POLL_SECONDS)
                if self.sync_on_start:
                    intake.sync(store, self.ledger_path)

        with self._lock:
            self.message = "Stopped. Everything finished is recorded."
        self._drain_publisher("final")
        self._log("Stopped cleanly")

    # ── selection ────────────────────────────────────────────────────────
    def _narrowed(self) -> list:
        """The selected components, after `VIOS_ONLY` and `VIOS_EXCEPT`.

        The escape hatch for a pass that misbehaves on one machine, and for
        running one pass across the archive from a second machine — a shard
        replays into any database by uid, so a host that runs only `ocr` is a
        contributor and not a fork. Narrowing happens here, above the wave
        filter and well above `plan_cohorts`, so everything downstream sees
        one list and no stage has to know these variables exist.

        An empty result is an operator error, not a state to soldier on in:
        the caller raises rather than sweeping zero passes and reporting
        success.
        """
        sel = list(self.selected)
        keep, drop = self.only, self.exclude
        if keep:
            sel = [c for c in sel if c in keep]
        if drop:
            sel = [c for c in sel if c not in drop]
        if not self._selection_logged:
            for var, names in _ENV_UNKNOWN.items():
                if names:
                    self._log(f"{var} names {', '.join(names)}, which is not a "
                              f"component id — ignored", "warn")
        if (keep or drop) and not self._selection_logged:
            if keep:
                self._log(f"VIOS_ONLY: running only "
                          f"{', '.join(c for c in keep if c in self.selected)}")
            if drop:
                self._log(f"VIOS_EXCEPT: skipping "
                          f"{', '.join(c for c in drop if c in self.selected)}")
        if not sel:
            raise RuntimeError(
                "VIOS_ONLY/VIOS_EXCEPT leave no pass selected — "
                f"only={','.join(keep) or '-'} except={','.join(drop) or '-'}")
        return sel

    def _earlier_wave_waiting(self, plan: list, wave: int,
                              runnable: list) -> bool:
        """Is there outstanding work in a wave before this one?

        This is what makes wave 1's priority permanent rather than a one-time
        ordering. A video posted to the channel while wave 2 is running has no
        transcript, no vectors and no description — it is invisible to search
        until its spine runs — so the rotation abandons wave 2 at the next
        cohort boundary, restarts at wave 1, and picks wave 2 back up
        afterwards. Nothing is lost by restarting: coverage remembers every
        row that is already `done`, so the second pass over wave 2 resumes
        where the first stopped.

        A cohort boundary is the only place this may be answered, because it
        is the only place the barrier holds — preempting mid-cohort would
        leave a cohort's outputs half-written and unpublished.

        In follow mode the check first pulls the ledger and re-plans, because
        a video captured minutes ago has no coverage rows at all yet and
        `candidates` cannot see work that was never planned.
        """
        earlier = [cid for w, group, _cs in plan if w < wave for cid in group]
        if not earlier:
            return False
        if self.follow and self.sync_on_start:
            try:
                got = intake.sync(self.store, self.ledger_path)
                if got.get("added"):
                    self._log(f"{got['added']} new videos arrived during "
                              f"wave {wave}")
                    self.coverage.plan(runnable)
            except Exception as exc:                   # noqa: BLE001
                # A ledger that cannot be read costs freshness, never the
                # rotation: the wave keeps running and the next boundary asks
                # again.
                self._log(f"Ledger check failed, continuing: "
                          f"{type(exc).__name__}: {exc}", "warn")
        if not self.coverage.candidates(earlier, limit=1):
            return False
        self._log(f"Wave {wave} yields — an earlier wave has work outstanding; "
                  f"restarting the rotation so the searchable spine goes first")
        with self._lock:
            self.message = (f"Pausing wave {wave} — new videos need their "
                            f"searchable spine first.")
        return True

    def _sleep(self, seconds: float) -> None:
        """Interruptible wait, in short slices, so stop is felt immediately."""
        end = time.time() + seconds
        while time.time() < end and not self._stopping():
            time.sleep(min(1.0, max(end - time.time(), 0)))

    # ── one cohort ───────────────────────────────────────────────────────
    def _run_cohort(self, cohort, wave: int = 0) -> int:
        """Get this cohort done across every video, then return.

        Two execution shapes behind one contract, and the barrier means the same
        thing in both. With lanes up this is a *planner*: it stages the next
        videos' bytes ahead of the work, posts one job per video per lane kind,
        and does not return until every pass lane is idle and its queues are
        empty. With `VIOS_LANES=0` it runs the same work inline on this thread,
        which is exactly the sweep that shipped before the lanes existed.

        `wave` is carried for the log and the status line only. A cohort runs
        the same way whichever wave admitted it; naming the wave is what makes
        "the spine is still going" readable in a log that is otherwise a list
        of cohort numbers.
        """
        ids = list(cohort.components)
        wname = registry.WAVE_NAMES.get(wave, "") if wave else ""
        tag = f" [{wname}]" if wname else ""
        with self._lock:
            self.cohort_index = cohort.index
            self.message = (f"Cohort {cohort.index + 1}{tag}: "
                            f"{', '.join(registry.get(i).title for i in ids[:4])}"
                            + (f" +{len(ids) - 4} more" if len(ids) > 4 else ""))
        self._log(f"Cohort {cohort.index}{tag}: {len(ids)} passes, "
                  f"{cohort.vram_mb} MB, {len(cohort.loads)} model loads")

        if self._lanes:
            return self._plan_cohort(cohort, wave)
        return self._sweep_cohort(cohort, ids)

    def _sweep_cohort(self, cohort, ids: list) -> int:
        """Every video, one at a time, on this thread — the pre-lanes sweep.

        `seen` is why a failure does not become a spin: a video whose pass
        failed is still claimable — that is the point of retries — and without
        remembering that it was attempted this cohort, `candidates` would hand
        it straight back and the loop would grind on the same reel until its
        attempts ran out. Retries belong to the next rotation, not this one.
        """
        lane = self._sweep_lane()
        seen: set = set()
        worked = 0
        while not self._stopping():
            self._wait_if_paused()
            keys = [k for k in lane.cov.candidates(ids, limit=64)
                    if k not in seen]
            if not keys:
                break
            for key in keys:
                if self._stopping():
                    break
                self._wait_if_paused()
                seen.add(key)
                if self.video_limit and self.session["videos"] >= self.video_limit:
                    self._log(f"Reached the {self.video_limit}-video limit "
                              f"for this session")
                    return worked
                if self._process_video(key, ids, cohort.index, lane):
                    worked += 1
        return worked

    def _split_cohort(self, ids: list) -> dict:
        """Which lane kind may take which of this cohort's passes.

        CPU components cost no VRAM, so `plan_cohorts` drops them into whichever
        cohort is open when topo order reaches them — sometimes directly beside
        the GPU passes that consume them (`allframes`, 45 s, is a hard need of
        most perception passes). Handing one of those to a different lane would
        put a cohort's own dependency edge across two threads, and the consumer
        would find its input still `queued` and defer for sixty seconds.

        So the CPU lane takes only passes with **no edge to a GPU pass of this
        cohort, in either direction** — computed from the registry rather than
        listed by hand, so a component added later is classified correctly
        without anybody remembering this function exists. Everything else stays
        in the video's own `topo_sort` order on one lane, exactly as before.

        Cloud passes come out separately: they are network and no local compute,
        and `_cloud_loop` explains why they are allowed across the barrier.
        """
        gpu_ids = {c for c in ids if registry.get(c).device == "gpu"}
        cloud = [c for c in ids if registry.get(c).family == "cloud"]
        rest = [c for c in ids if c not in cloud]
        edge_free = [c for c in rest
                     if registry.get(c).device == "cpu"
                     and not (set(registry.get(c).needs) & gpu_ids)
                     and not any(c in registry.get(g).needs for g in gpu_ids)]
        return {"gpu": [c for c in rest if c not in edge_free],
                "cpu": edge_free, "cloud": cloud}

    def _plan_cohort(self, cohort, wave: int) -> int:
        """Post this cohort's work, keep the lanes fed, then hold the barrier.

        The planner never runs a pass itself. It answers one question in a loop —
        *which videos still owe this cohort something, and are their bytes on
        disk?* — and posts a hint per (video, lane kind). Ownership is settled in
        sqlite by `claim_for`, so a hint posted twice, or posted for a video
        another lane already took, costs one ack and nothing else.

        It returns when the pass lanes are idle, their queues are empty and
        `candidates` has nothing left. That is precisely the barrier
        `plan_cohorts` documents — a cohort's outputs complete before the next
        begins — held by four workers instead of one. The cloud lane is
        deliberately not part of that test.
        """
        split = self._split_cohort(list(cohort.components))
        gpu_ids, cpu_ids = split["gpu"], split["cpu"]
        self._log(f"  lanes: {len(gpu_ids)} on the GPU lanes, {len(cpu_ids)} "
                  f"edge-free on the CPU lanes"
                  + (f", {len(split['cloud'])} cloud (runs across the barrier)"
                     if split["cloud"] else "")
                  + (f" — CPU lane takes {', '.join(cpu_ids)}" if cpu_ids
                     else ""))

        before = self.session["videos"]
        mine = list(gpu_ids) + list(cpu_ids)
        deadline = time.time() + _COHORT_DRAIN_TIMEOUT
        posted: set = set()
        quiet = 0

        while not self._stopping():
            self._wait_if_paused()
            if self.video_limit and self.session["videos"] >= self.video_limit:
                self._log(f"Reached the {self.video_limit}-video limit for "
                          f"this session")
                break

            # The cloud lane first, and every round: it is fed independently of
            # the cohort so that a 25 s network pass is never what a card is
            # waiting behind.
            self._post_cloud()

            # Hints a lane consumed without processing anything — it found the
            # video held by another lane, or `claim_for` had already given the
            # rows away. Those are the only cases worth posting twice: a video
            # that *was* processed stays in `posted` however its passes went, so
            # a reel whose every pass fails costs one attempt this cohort and its
            # retries belong to the next rotation.
            with self._lock:
                again, self._retry_hint = self._retry_hint, set()
            posted -= again

            # `posted` is also what stops a failed video spinning: it is still a
            # `candidates` answer — that is what retries are — and without
            # remembering the attempt this loop would grind on the same reel.
            todo = [k for k in self.coverage.candidates(mine, limit=64)
                    if k not in posted] if mine else []
            staged = self._stage_ahead(todo, mine)

            # A hint is only posted for a video whose bytes are already down.
            # That single rule is what keeps a GPU lane from ever blocking on
            # Telegram or on ffmpeg — the whole point of the prep lane.
            fed = 0
            for key in staged:
                posted.add(key)
                if gpu_ids:
                    fed += bool(self._jobs.push(jobs.QUEUE_V2_GPU, {
                        "video_key": key, "cohort": cohort.index, "wave": wave,
                        "components": gpu_ids}))
                if cpu_ids:
                    fed += bool(self._jobs.push(jobs.QUEUE_V2_CPU, {
                        "video_key": key, "cohort": cohort.index, "wave": wave,
                        "components": cpu_ids}))

            # Drained? Three things have to be true at once, and they are
            # checked twice in a row before believing it: a lane can be between
            # `claim` returning and its slot being filled, which for one instant
            # looks exactly like idle.
            work_left = bool(todo) or fed > 0
            busy = any(l.busy() for l in self._lanes if l.kind in ("gpu", "cpu"))
            depth = (self._jobs.depth(jobs.QUEUE_V2_GPU)
                     + self._jobs.depth(jobs.QUEUE_V2_CPU)
                     + self._jobs.depth(jobs.QUEUE_V2_PREP))
            if not work_left and not busy and depth == 0:
                quiet += 1
                if quiet >= 2:
                    break
            else:
                quiet = 0

            if time.time() > deadline:
                # Never silently: a cohort that will not drain is a stuck lane
                # or a video that hangs a runner, and the next cohort starting
                # on top of it is how a barrier becomes a suggestion.
                self._log(f"Cohort {cohort.index} did not drain within "
                          f"{_COHORT_DRAIN_TIMEOUT:.0f}s — "
                          f"{'a lane is still busy' if busy else 'queues stalled'}"
                          f", {depth} hints queued. Moving on; the rows stay "
                          f"claimable and the next rotation retries them",
                          "error")
                break

            with self._lock:
                self.message = (
                    f"Cohort {cohort.index + 1}: "
                    f"{sum(1 for l in self._lanes if l.busy())}/"
                    f"{len(self._lanes)} lanes working, {len(todo)} videos "
                    f"outstanding, {depth} queued")
            self._sleep(1.0 if (work_left or busy) else 0.5)

        # Hints for a cohort nobody is going to admit again are noise the next
        # cohort's lanes would claim and immediately ack.
        for q in (jobs.QUEUE_V2_GPU, jobs.QUEUE_V2_CPU):
            self._jobs.drain(q)
        return max(self.session["videos"] - before, 0)

    def _stage_ahead(self, todo: list, ids: list) -> list:
        """Post prep for the next few videos; answer which are ready now.

        `_PREP_AHEAD` bounds how far ahead the prep lane runs, because staging is
        the one activity that consumes disk without producing evidence — and
        `intake.evict` protects held directories, so unbounded staging would fill
        the cache with videos no lane has reached.

        The bookkeeping is done under the lock and the pushes outside it: the
        prep lanes mutate both sets, and a Redis round trip is not something to
        hold an engine lock across.
        """
        with self._lock:
            ready = [k for k in todo if k in self._staged]
            want = max(_PREP_AHEAD - len(ready), 1)
            post: list = []
            for key in todo:
                if want <= 0:
                    break
                if key in self._staged or key in self._prep_posted:
                    continue
                self._prep_posted.add(key)
                post.append(key)
                want -= 1
        for key in post:
            self._jobs.push(jobs.QUEUE_V2_PREP,
                            {"video_key": key, "components": ids})
        return ready

    def _post_cloud(self) -> None:
        """Feed the cloud lane, whatever cohort the cards are on."""
        ids = list(self._cloud_ids)
        if not ids or not any(l.kind == "cloud" for l in self._lanes):
            return
        for key in self.coverage.candidates(ids, limit=16):
            if key in self._cloud_posted:
                continue
            self._cloud_posted.add(key)
            self._jobs.push(jobs.QUEUE_V2_CLOUD,
                            {"video_key": key, "components": ids})
        # Bounded, because this set is the only thing in the plane that grows
        # for the whole session. A key dropped from it is re-posted and
        # re-acked — the cost of forgetting is one wasted claim.
        if len(self._cloud_posted) > 4096:
            self._cloud_posted.clear()

    # ── one video, every resident pass ───────────────────────────────────
    def _process_video(self, key: str, ids: list, cohort_index: int,
                       lane: "Lane") -> bool:
        cov, store = lane.cov, lane.store
        video = store.video(key)
        if video is None:
            return False

        mine = cov.claim_for(key, ids)
        if not mine:
            return False                      # another worker got there first
        order = registry.topo_sort(mine)
        workdir = os.path.join(self.cache_dir, _safe(key))
        started = time.time()

        lane.slot = {"video_key": key, "cohort": cohort_index,
                     "component": "", "title": "fetching",
                     "since": started, "passes": len(order),
                     "lane": lane.name, "device": lane.device}

        try:
            source = self._fetch(video, workdir)
        except intake.SourceError as exc:
            for cid in order:
                cov.fail(key, cid, str(exc))
            self._log(f"{key}: {exc}", "warn")
            self._bump("failed", len(order))
            lane.slot = {}
            return False
        except Exception as exc:
            for cid in order:
                cov.fail(key, cid, f"{type(exc).__name__}: {exc}")
            self._log(f"{key}: fetch failed — {type(exc).__name__}: {exc}",
                      "error")
            self._bump("failed", len(order))
            lane.slot = {}
            return False
        intake.touch(workdir)

        states = {r["component"]: r["state"] for r in cov.for_video(key)}
        for cid in order:
            if self._stopping():
                cov.release(key, cid)
                continue
            self._wait_if_paused()
            states[cid] = self._run_pass(key, cid, video, source, workdir,
                                         states, lane)

        lane.videos += 1
        self._bump("videos")
        self._bump("seconds", time.time() - started)
        with self._lock:
            self.since_publish += 1
        # Not `if due: publish` any more. The whole point of the publisher thread
        # is that the decision and the 45 MB POST happen somewhere this loop is
        # not — so this says "there are rows" and goes back to the next video.
        # The publisher applies the same `_publish_due` gate on its own tick.
        self._rows_written(f"cohort-{cohort_index}")

        # `keep` is every working directory any lane is inside, not just this
        # one. A keep-set of one is what would delete a live workdir out from
        # under another lane mid-pass.
        freed = intake.evict(self.cache_dir, self.cache_budget_mb,
                             keep=self._held() | {_safe(key)},
                             floor_mb=self.disk_floor_mb)
        if freed["removed"]:
            self._log(f"Cache: removed {freed['removed']} working directories, "
                      f"{freed['freed_mb']} MB")
            # Whatever was evicted is no longer staged, whether or not this lane
            # was the one that staged it. Saying otherwise would have the planner
            # post a pass job for a video whose bytes are gone.
            with self._lock:
                self._staged -= {k for k in self._staged
                                 if _safe(k) in set(freed.get("names") or ())}
        lane.slot = {}
        with self._lock:
            self._staged.discard(key)
        return True

    def _fetch(self, video: dict, workdir: str) -> str:
        """`Source.ensure`, one lane at a time.

        Idempotent and usually instant — `source.mp4` and `record.json` present
        means `reused += 1` and return — so the lock costs nothing on the common
        path. It is held across the download because pyrogram's `Channel` is not
        safe for concurrent downloads and the Bot API's rate limit is per chat:
        four lanes pulling at once is slower than one, and can be a ban.
        """
        with self._fetch_lock:
            return self._source.ensure(video, workdir)

    # ── one pass ─────────────────────────────────────────────────────────
    def _run_pass(self, key: str, cid: str, video: dict, source: str,
                  workdir: str, states: dict, lane: "Lane") -> str:
        cov, store = lane.cov, lane.store
        comp = registry.get(cid)

        # Can this machine run it at all?
        #
        # `kaggle_ok=False` means the pass is known not to work on Kaggle's
        # 2×T4 — a 38B that will not shard across PCIe inside nine hours, a
        # kernel that needs Ampere. The point of recording that as data is that
        # the pass is neither deleted from the registry nor hacked into fitting:
        # the coverage row says `deferred` with the reason, so the matrix
        # distinguishes "cannot run here" from "broke", and a machine that can
        # run it finds the work waiting rather than marked done.
        #
        # Every component ships `kaggle_ok=True`, so today this never fires. It
        # is the door, not the room: when a pass proves unrunnable on a T4 the
        # honest fix is one field, not a code change.
        if not comp.kaggle_ok and self.resources.get("host") == "kaggle":
            try:
                where = resources.describe(self.resources)
            except Exception:               # noqa: BLE001
                where = "this host"
            why = (f"not runnable on {where} — flagged kaggle_ok=False in the "
                   f"registry, held for a machine that can run it")
            cov.defer(key, cid, why, ELSEWHERE_SECONDS)
            self._bump("elsewhere")
            self._elsewhere.add(cid)
            if cid not in self._elsewhere_logged:
                self._elsewhere_logged.add(cid)
                self._log(f"{cid}: deferred for another machine — "
                          f"{comp.title} is flagged as not runnable here", "warn")
            return "deferred"

        # Dependency gate. A pass whose input was never produced should not
        # spend a GPU second discovering that for itself.
        #
        # `soft` narrows the veto to inputs that supply a pass's subject. The
        # runners were written for absence — `Job.text_bundle` returns an empty
        # `on_screen` list when OCR wrote nothing, and `narrate`, `keyphrase`,
        # `concepts` and `text-embed` each raise their own skip naming every
        # source they looked in — so one engine that could not start was
        # deleting six passes that would have run without it. Membership in
        # `needs` is unchanged, so `topo_sort` and `plan_cohorts` still order
        # the work exactly as before; only the veto is narrower.
        broken = [n for n in comp.needs
                  if n not in comp.soft
                  and states.get(n) in ("failed", "skipped")]
        if broken:
            # `failed` and `skipped` are not the same answer, and treating them
            # as one is a second way this engine used to lose evidence
            # permanently. A `skipped` need has declined: it will decline again
            # next session, so retiring its consumer with it is correct. A
            # `failed` need with revivals left has declined nothing — it is on
            # the retry ladder and `revive_failed` hands it back in four hours.
            # One OOM on `shots` therefore used to write a *terminal* skip for
            # `keyframes`, `poster` and everything downstream of them; `shots`
            # came back on the next revival and succeeded, and its consumers were
            # already unreachable to `claim`, `candidates`, `revive_failed` and
            # `reconcile` alike.
            #
            # So wait for the ones that are still coming, and wait exactly as
            # long as they are waiting — `unsettled` returns their own
            # `next_try_at`, which stops this from re-asking every sixty seconds
            # for four hours. Only when every broken need has genuinely settled
            # does the skip below become the truth.
            waiting = cov.unsettled(key, broken)
            if waiting:
                due = max(waiting.values())
                wait = max(60.0, min(due - time.time(), ELSEWHERE_SECONDS))
                cov.defer(key, cid, f"waiting on "
                                    f"{', '.join(sorted(waiting))}, which "
                                    f"failed but has retries left", wait)
                self._bump("deferred")
                return "deferred"
            why = (f"depends on {', '.join(broken)}, which produced nothing "
                   f"for this video")
            cov.skip(key, cid, why)
            self._bump("skipped")
            return "skipped"

        degraded = [n for n in comp.soft
                    if states.get(n) in ("failed", "skipped")]
        # Everything this pass will not have, which is a wider question than the
        # one above and a different one. `degraded` is the *veto* list: soft
        # inputs that reached a terminal state, which is what decides whether to
        # wait for them. This is the *provenance* list: soft inputs that have not
        # produced anything, terminal or not — a need still queued behind a wave
        # boundary (`wave_leaks` warns about exactly this at startup), one still
        # running in another lane, one deferred against a rate limit, and one that
        # was never selected this session at all, which `states.get` returns None
        # for and the veto list therefore cannot see.
        #
        # That last case is the common one and the reason this is not just
        # `degraded`: a session that runs the language passes without `ocr` in its
        # selection produces the same thin answer as one where `ocr` failed, and
        # only this list notices. It is what `cov.done` records and what the
        # observer id below is derived from, so the row and the evidence agree on
        # what the answer was made of.
        thin = [n for n in comp.soft if states.get(n) != "done"]
        # Which of the missing soft inputs might still arrive. Read once, here,
        # because the `except SkipPass` handler needs it: a pass that declines
        # *because* a source was empty has not declined the video, and marking
        # that terminal is how a retired `ocr` took `keyphrase`, `concepts`,
        # `text-embed` and `narrate` down with it — each of those raises its own
        # skip naming every source it looked in, and every one of those skips is
        # forever.
        soft_waiting = cov.unsettled(key, degraded) if degraded else {}
        if degraded:
            # Recorded, not vetoed: the claim this pass writes carries a note
            # saying which inputs were missing, so a thinner answer is legible
            # as thinner rather than passed off as complete.
            self._log(f"{key} {cid}: running without {', '.join(degraded)} "
                      f"— soft input, the pass reports what it had", "warn")
        if thin:
            # The wider set, said once per (pass, missing-set) rather than once
            # per video, and it names the mechanism because the mechanism is new:
            # the row keeps this list, and `reoffer_degraded` hands the row back
            # when one of these inputs finally writes something.
            seen = (cid, ",".join(sorted(thin)))
            if seen not in self._thin_logged:
                self._thin_logged.add(seen)
                self._log(f"{cid}: answering without {', '.join(sorted(thin))} "
                          f"for now — recorded on each row, and re-offered if "
                          f"that evidence arrives later")

        # A hard need that has not run yet. The serial cohort barrier means this
        # cannot fire today: `topo_sort` puts every need ahead of its consumer
        # inside one video, and a cross-cohort need is complete before this
        # cohort begins. It is the net under that invariant — a future packer,
        # a wave boundary or a second worker taking a different slice of the
        # same video degrades into a retry in sixty seconds instead of a pass
        # silently fed nothing.
        pending = [n for n in comp.needs
                   if n not in comp.soft
                   and n in self._runnable
                   and states.get(n) not in ("done", "failed", "skipped",
                                             "deferred")]
        if pending:
            cov.defer(key, cid, f"waiting on {', '.join(pending)}, not yet run "
                                f"for this video", 60.0)
            self._bump("deferred")
            return "deferred"

        # A dependency that deferred has not failed — it is waiting on a clock.
        # Skipping this pass would make a rate limit permanent for the video,
        # so it waits with it.
        waiting = [n for n in comp.needs if states.get(n) == "deferred"]
        if waiting:
            # …and if what it is waiting on cannot run here at all, it waits on
            # the same clock. Five minutes is the right interval for a rate
            # limit and pure churn for a machine that will never be able to run
            # the dependency, which would otherwise re-ask every sweep forever.
            elsewhere = [n for n in waiting if n in self._elsewhere]
            why = f"waiting on {', '.join(waiting)}, which is deferred"
            if elsewhere:
                why = (f"waiting on {', '.join(elsewhere)}, which needs a "
                       f"different machine")
            cov.defer(key, cid, why,
                      ELSEWHERE_SECONDS if elsewhere else 300.0)
            self._bump("deferred")
            return "deferred"

        lane.slot.update({"component": cid, "title": comp.title,
                          "component_since": time.time(), "note": "",
                          "detail": "", "detail_at": 0.0,
                          "frames_done": 0, "frames_total": 0})
        # So a load or unload fault names the pass that provoked it rather than
        # whichever pass happened to be running when the log was read.
        lane.cache.context = cid

        params = dict(comp.params)
        if self._hf_token:
            params.setdefault("hf_token", self._hf_token)
        job = Job(
            video=dict(video), component=comp, store=store, source=source,
            workdir=workdir, params=params, resources=self.resources,
            cache=lane.cache,
            renew=lambda progress="", k=key, c=cid: cov.renew(k, c, progress),
            progress=lambda d, _l=lane: self._progress(d, _l),
            log=lambda m, _l=lane: self._note(m, _l))

        t0 = time.time()
        try:
            self._ensure_inputs(job, comp)
            fn = get_runner(cid)
            em = fn(job)
        except PassUnavailable as exc:
            # Before `except SkipPass`, because it is a subclass of it, and the
            # ordering is the whole fix. A pass that could not *start* has said
            # nothing about the video, so retiring the row is a lie that costs
            # the channel: `cov.skip` writes a state `claim`, `candidates`,
            # `revive_failed` and `reconcile` all refuse to reconsider, so one
            # session without a library retired `ocr` for twenty-eight of thirty
            # videos and the EasyOCR fallback that landed nine days later could
            # never reach them. Routing it to `fail` puts it on the retry ladder
            # that `revive_failed`'s docstring was written for — "a package that
            # was missing until the next session installed it" — bounded at
            # three attempts and six revivals, so a machine that will never have
            # the library costs 21 fast load attempts and then stops.
            #
            # Logged as a warning, not an error: nothing here is a bug in the
            # pass. The reason is carried into `last_error` exactly as a skip
            # would have carried it, so the tab still says what is missing.
            reason = f"unavailable: {exc}"
            state = cov.fail(key, cid, reason)
            self._bump("unavailable")
            self._log(f"{key} · {cid}: could not start — {exc}"
                      + (" (retrying later)" if state == "queued"
                         else " (out of attempts; will be revived)"), "warn")
            return state
        except SkipPass as exc:
            # A decline is terminal — but only once the sources it read are
            # settled. `soft_waiting` is the soft inputs that failed and still
            # have revivals left, and a pass that raises "no transcript, caption
            # or on-screen text for this video" while one of them is mid-ladder
            # is reporting the state of the pipeline, not of the reel. Skipping
            # it there is what made a retired `ocr` retire its six readers too.
            # Waiting costs at most one extra execution per revival of the input,
            # and it stops as soon as that input reaches a state that will not
            # change; the skip below is then written with the same reason it
            # would have been written with now.
            if soft_waiting:
                due = max(soft_waiting.values())
                wait = max(60.0, min(due - time.time(), ELSEWHERE_SECONDS))
                cov.defer(key, cid, f"{exc} — and "
                                    f"{', '.join(sorted(soft_waiting))} may "
                                    f"still produce some", wait)
                self._bump("deferred")
                self._log(f"{key} · {cid}: {exc} — waiting for "
                          f"{', '.join(sorted(soft_waiting))} to finish "
                          f"retrying before calling that final", "warn")
                return "deferred"
            cov.skip(key, cid, str(exc))
            self._bump("skipped")
            self._log(f"{key} · {cid}: skipped — {exc}")
            return "skipped"
        except DeferPass as exc:
            # Not a failure and not a skip: the work is runnable, the moment
            # is wrong. No attempt is spent, so an archive cannot exhaust its
            # retries against a rate limit that clears on its own.
            wait = max(30.0, min(float(getattr(exc, "retry_after", 300.0)),
                                 3600.0))
            cov.defer(key, cid, str(exc), wait)
            self._bump("deferred")
            self._log(f"{key} · {cid}: deferred {wait:.0f}s — {exc}")
            return "deferred"
        except (KeyboardInterrupt, SystemExit):
            cov.release(key, cid)
            raise
        except Exception as exc:
            # Name the weights if the fault happened while loading them. A row
            # that reads "OutOfMemoryError" tells the operator nothing they can
            # act on; one that reads "loading Qwen/Qwen3-VL-8B-AWQ" tells them
            # which model to move off this card.
            blame = self._loading_blame(exc, t0, lane)
            reason = f"{type(exc).__name__}: {exc}"
            if blame:
                reason = f"{reason} [{blame}]"
            state = cov.fail(key, cid, reason)
            self._bump("failed")
            self._log(f"{key} · {cid}: {type(exc).__name__}: "
                      f"{str(exc)[:200]}" + (f" — while {blame}" if blame else ""),
                      "error")
            if self._is_oom(exc):
                v = ModelCache.vram()
                self._log(f"Out of VRAM on {cid}"
                          + (f" while {blame}" if blame else "")
                          + (f" — {v.get('allocated', 0)} MB allocated, "
                             f"{v.get('free', 0)} MB free of "
                             f"{v.get('total', 0)}; resident: "
                             f"{', '.join(lane.cache.loaded()) or 'nothing'}"
                             if v else ""), "error")
                # This lane's models only. Another lane's card has its own
                # budget and is very likely mid-pass; dropping its weights to
                # answer an OOM over here would turn one failure into two.
                self._unload("out of VRAM — dropping every resident model",
                             lane)
            return state

        # ── accept the emission ──────────────────────────────────────────
        try:
            observer = store.observer(
                cid, comp.model or comp.family, comp.revision,
                _observer_params(comp.params, thin), comp.device)
            n_claims = store.add_claims(key, observer, em.claims) if em.claims else 0
            n_vectors = 0
            for v in em.vectors:
                if store.add_vector(key, v["space"], v["values"], observer,
                                    v.get("shot_idx")):
                    n_vectors += 1
            # Per-frame embeddings and scalars are packed one row per
            # (video, space|name, observer), so nine hundred frames of SigLIP
            # is a single row of about 1.4 MB rather than nine hundred rows of
            # base64. Counted as vectors for the session tally because that is
            # what they are to everything downstream.
            for fv in getattr(em, "frame_vectors", ()):
                if store.add_frame_vectors(key, fv["space"], fv["frames"],
                                           fv["matrix"], observer):
                    n_vectors += 1
            for fm in getattr(em, "frame_metrics", ()):
                if store.add_frame_metric(key, fm["name"], fm["frames"],
                                          fm["values"], observer):
                    n_vectors += 1
            # What the channel already holds for this video, read *before* the
            # loop below overwrites it. `set_artifact` is INSERT OR REPLACE on
            # `(video_key, kind)`, so recording the fresh emission plainly would
            # erase the message id of an artifact uploaded on an earlier
            # rotation — and `revive_failed` guarantees earlier rotations
            # happen. Reading first, then carrying a still-valid id forward, is
            # the whole of what makes the upload idempotent.
            prior = {}
            if em.artifacts:
                try:
                    prior = store.artifacts(key)
                except Exception:                           # noqa: BLE001
                    prior = {}
            for a in em.artifacts:
                size = 0
                try:
                    size = (os.path.getsize(a["path"])
                            if os.path.isfile(a["path"]) else 0)
                except OSError:
                    pass
                # Same bytes as the copy already in the channel → that message
                # still describes this file, so keep it. Any other size means a
                # re-render, and pointing at the old message would be a lie.
                old = prior.get(a["name"]) or {}
                msg_id, file_id = None, ""
                try:
                    if old.get("msg_id") and int(old.get("bytes") or 0) == size:
                        msg_id = int(old["msg_id"])
                        file_id = old.get("file_id") or ""
                except (TypeError, ValueError):
                    msg_id, file_id = None, ""
                store.set_artifact(key, a["name"], msg_id, file_id, size,
                                   a["meta"])
            # Then, and only for the four kinds worth a message, put them
            # somewhere that survives the working directory being evicted. After
            # `set_artifact`, never instead of it: the row is evidence that the
            # pass produced the file, and it has to land whether or not Telegram
            # is reachable.
            if em.artifacts:
                self._upload_artifacts(key, em.artifacts, store, lane)
        except Exception as exc:
            state = cov.fail(key, cid, f"store rejected the emission: "
                                       f"{type(exc).__name__}: {exc}")
            self._bump("failed")
            self._log(f"{key} · {cid}: store rejected the emission — "
                      f"{type(exc).__name__}: {exc}", "error")
            return state

        seconds = time.time() - t0
        # `thin` travels into the row. It is the same list the observer id above
        # was derived from, so the coverage row and the evidence agree on what this
        # answer was made of — and it is what lets `reoffer_degraded` find the row
        # again if one of those inputs eventually writes something.
        cov.done(key, cid, seconds, n_claims, n_vectors, observer, thin)

        lane.passes += 1
        self._bump("passes")
        self._bump("claims", n_claims)
        self._bump("vectors", n_vectors)
        return "done"

    def _note(self, message: str, lane: "Lane | None" = None) -> None:
        slot = self._slot_of(lane)
        if slot is not None:
            slot["note"] = str(message)[:200]
        self._log(message)

    def _slot_of(self, lane: "Lane | None"):
        """The dict to write a live detail into — the object, never a copy.

        Every real call site passes its lane. The fallback exists because
        `self.current` returns a *copy*, so writing to it would silently discard
        the note, and a defaulted argument that quietly does nothing is worse
        than one that raises.
        """
        if lane is not None:
            return lane.slot
        if self._sweep_lane_obj is not None:
            return self._sweep_lane_obj.slot
        for l in list(self._lanes):
            if l.busy():
                return l.slot
        return None

    def _progress(self, detail: str, lane: "Lane | None" = None) -> None:
        """A pass's own report of where it is inside itself.

        Deliberately not logged. `job.heartbeat` fires once per batch, so a
        900-frame OCR pass produces about thirty of these per video and several
        thousand per sweep; putting them in the activity ring would push out
        every error the ring exists to keep. They go to the lane's slot only,
        which is what the live panel reads.

        `frame 320/900` is parsed into counters here rather than in the browser
        so the shape stays in one place: the runners already emit that string,
        and a progress bar in the interface should not depend on a regex in
        JavaScript agreeing with a format string in Python.

        No lock. A lane's slot is written by exactly one thread — its own — and
        every reader takes a copy; taking the engine lock thirty times a video
        per lane would put four workers in a queue behind a status poll.
        """
        detail = str(detail)[:200]
        slot = self._slot_of(lane)
        if not slot:
            return
        slot["detail"] = detail
        done = total = 0
        hit = _PROGRESS_RE.search(detail)
        if hit:
            done, total = int(hit.group(1)), int(hit.group(2))
        slot["frames_done"] = done
        slot["frames_total"] = total
        slot["detail_at"] = time.time()

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        name = type(exc).__name__
        return ("OutOfMemory" in name
                or "CUDA out of memory" in str(exc)
                or "CUBLAS_STATUS_ALLOC_FAILED" in str(exc))

    def _loading_blame(self, exc: Exception, since: float,
                       lane: "Lane | None" = None) -> str:
        """"loading <key>", when this failure came out of a model load.

        `ModelCache` records every load fault as it happens, with the key it
        was loading. Matching on the exception text rather than re-raising a
        wrapper keeps the original traceback intact — which is what someone
        reading the Kaggle log actually needs — while still putting the model
        name in the coverage row, where it is queryable.

        The lane's own cache, because with two cards there are two of them and
        the other lane's most recent load failure is a different card's problem.
        """
        try:
            cache = lane.cache if lane is not None else self._cache
            msg = str(exc)[:300]
            for f in reversed(cache.recent_failures(8)):
                if f.get("at", 0) < since - 1.0:
                    break
                if f.get("phase") == "load" and f.get("key"):
                    same = msg and msg[:120] in f.get("error", "")
                    return (f"loading {f['key']}" if same
                            else f"loading {f['key']} had just failed")
        except Exception:
            pass
        # Nothing recorded: the fault happened during inference, not during a
        # load. Saying nothing is the honest answer — naming the resident model
        # here would blame whichever one happened to be loaded.
        return ""

    # ══════════════════════════════════════════════════════════════════════
    # The lanes
    # ══════════════════════════════════════════════════════════════════════

    def _sweep_lane(self) -> "Lane":
        """The sweep thread's own lane, for the `VIOS_LANES=0` path.

        Wraps the primary store, coverage and cache, so `_process_video` has one
        signature whether it is running on a worker thread or on the sweep.
        """
        with self._lock:
            if self._sweep_lane_obj is None:
                lane = Lane("sweep", "gpu", -1)
                lane.store = self.store
                lane.cov = self.coverage
                lane.cache = self._cache
                lane.started_at = time.time()
                self._sweep_lane_obj = lane
            return self._sweep_lane_obj

    def _start_lanes(self, res: dict) -> None:
        """Bring up the workers, once, at the top of the first rotation.

        Here and not in `__init__` because the lane count comes from `probe()` —
        one GPU lane per card, and the number of cards is not known until the
        first probe. Everything downstream reads `self._lanes`, so a machine
        where this decides on zero lanes runs the single-threaded sweep with no
        further branching anywhere.
        """
        if self._lanes or self._lanes_up:
            return
        self._lanes_up = True
        if not _LANES_ON:
            self._log("VIOS_LANES=0 — single-threaded sweep: one video, one "
                      "pass, one card at a time")
            return

        gpus = int(res.get("gpu_count", 0) or 0)
        n_gpu = _GPU_LANES if _GPU_LANES else gpus
        n_cpu = _CPU_LANES
        n_prep = max(1, min(2, _PREP_AHEAD))
        if n_gpu <= 0 and gpus <= 0:
            # No card at all: CPU passes still parallelise, and the CPU lane's
            # `_split_cohort` sees no GPU ids so every pass is edge-free.
            n_gpu = 0
        kind = self._jobs.start(wait_seconds=5.0)

        specs = [(f"gpu{i}", "gpu", (i % gpus) if gpus else -1)
                 for i in range(n_gpu)]
        specs += [(f"cpu{i}", "cpu", -1) for i in range(n_cpu)]
        specs += [(f"prep{i}", "prep", -1) for i in range(n_prep)]
        if self._cloud_possible():
            specs.append(("cloud", "cloud", -1))
        if not specs:
            self._log("No lane could be started — falling back to the "
                      "single-threaded sweep", "warn")
            return

        # Orphans from a session that died holding jobs. The rows' leases would
        # free them eventually; recovering the hints means a lane takes them now.
        for q in jobs.V2_QUEUES:
            got = self._jobs.recover(q)
            if got:
                self._log(f"Recovered {got} orphaned hints from {q}")

        for name, lane_kind, device in specs:
            lane = Lane(name, lane_kind, device)
            t = threading.Thread(target=self._lane_main, args=(lane,),
                                 name=f"vios-{name}", daemon=True)
            lane.thread = t
            with self._lock:
                self._lanes.append(lane)
            t.start()

        with self._lock:
            self.workers = {l.name: l.as_dict() for l in self._lanes}
        self._log(
            f"Lanes up on {kind}: {n_gpu} GPU"
            + (f" (cards {', '.join(str(s[2]) for s in specs if s[1] == 'gpu')})"
               if n_gpu else "")
            + f", {n_cpu} CPU, {n_prep} prep"
            + (", 1 cloud" if any(s[1] == "cloud" for s in specs) else "")
            + f" — {len(specs)} workers claiming in parallel")

    def _cloud_possible(self) -> bool:
        """Is there a cloud pass to run, and a key to run it with?

        Asked before the lane is created rather than inside it: a lane that
        exists only to discover on every claim that there is no API key is a
        thread and a queue that mean nothing on the tab.
        """
        try:
            if not any(registry.get(c).family == "cloud"
                       for c in self._narrowed()):
                return False
            from .runners.cloud import client      # noqa: PLC0415
            return bool(client().configured())
        except Exception:                          # noqa: BLE001
            return False

    def _stop_lanes(self) -> None:
        """Join every lane, then close its connection and drop its weights.

        `_stop` is already set by every path that gets here, so the loops are
        on their way out; the join is what makes "the sweep has stopped" mean
        "nothing is still writing". Generous timeout because a lane may be
        inside a 110 s Florence-2 pass, and killing that would abandon a claim
        with a live lease.
        """
        lanes = list(self._lanes)
        if not lanes:
            return
        self._log(f"Stopping {len(lanes)} lanes")
        for lane in lanes:
            t = lane.thread
            if t and t.is_alive():
                t.join(timeout=300)
        still = [l.name for l in lanes if l.thread and l.thread.is_alive()]
        if still:
            self._log(f"Lanes still running after 300s: {', '.join(still)} — "
                      f"their claims stay leased and are reclaimed as stale",
                      "warn")
        for lane in lanes:
            lane.close()
        with self._lock:
            self._lanes = []
            self._lanes_up = False
            self.workers = {}

    def _lane_open(self, lane: "Lane") -> None:
        """Pin the card and open this lane's own resources, on its own thread.

        `torch.cuda.set_device` sets CUDA's *current device for this host
        thread*, and `device_and_dtype` returns a bare `"cuda"` — so this one
        call is what puts every `.to("cuda")` in every runner this thread ever
        calls onto card `lane.device`, with no runner edited and nothing passed
        down. It has to happen here, on the lane's thread, which is why lane
        setup is not in `_start_lanes`.
        """
        if lane.device >= 0:
            try:
                import torch                       # noqa: PLC0415
                torch.cuda.set_device(lane.device)
            except Exception as exc:               # noqa: BLE001
                self._log(f"{lane.name}: could not pin card {lane.device} — "
                          f"{type(exc).__name__}: {str(exc)[:120]}; it will "
                          f"share whichever card CUDA defaults to", "warn")
        lane.store = self.store.secondary()
        lane.cov = Coverage(lane.store.conn, self.partitions, self.index,
                            self.worker)
        lane.cache = ModelCache(
            lambda text, level="info", _n=lane.name:
            self._log(f"[{_n}] {text}", level))
        lane.started_at = time.time()

    def _lane_main(self, lane: "Lane") -> None:
        try:
            self._lane_open(lane)
        except Exception as exc:                   # noqa: BLE001
            self._log(f"{lane.name}: could not start — {type(exc).__name__}: "
                      f"{str(exc)[:160]}", "error")
            with self._lock:
                self._lane_error = f"{lane.name}: {type(exc).__name__}"
            return
        try:
            if lane.kind == "prep":
                self._prep_loop(lane)
            elif lane.kind == "cloud":
                self._cloud_loop(lane)
            else:
                self._pass_loop(lane)
        except Exception as exc:                   # noqa: BLE001
            self._log(f"{lane.name} stopped on {type(exc).__name__}: "
                      f"{str(exc)[:200]}", "error")
            self._log(traceback.format_exc()[-700:], "error")
            with self._lock:
                self._lane_error = f"{lane.name}: {type(exc).__name__}"
        finally:
            lane.slot = {}
            lane.close()

    def _pass_loop(self, lane: "Lane") -> None:
        """Claim a video's slice of the admitted cohort, run it, ack.

        The hint says which video and which components. `claim_for` says whether
        they are actually this lane's — and when it says no, that is the normal
        answer to two lanes racing for the same reel, so the job is acked and
        the loop moves on without a word.
        """
        qname = jobs.QUEUE_V2_GPU if lane.kind == "gpu" else jobs.QUEUE_V2_CPU
        while not self._stopping():
            self._wait_if_paused()
            job, raw = self._jobs.claim(qname)
            if not job:
                continue
            p = job.get("payload") or {}
            key = str(p.get("video_key") or "")
            ids = _known(p.get("components") or [])
            if not key or not ids:
                self._jobs.ack(qname, job, raw)
                continue
            try:
                with self._hold(key, lane) as got:
                    ran = (self._process_video(
                        key, ids, int(p.get("cohort", -1) or -1), lane)
                        if got else False)
                if not ran:
                    # Nothing was attempted: another lane holds the video, or
                    # `claim_for` had already given the rows away. The planner
                    # re-posts this key so a reel is not left queued when the
                    # cohort barrier lifts — and only this case re-posts, so a
                    # video whose passes all failed still costs one attempt.
                    with self._lock:
                        self._retry_hint.add(key)
            except (KeyboardInterrupt, SystemExit):
                self._jobs.fail(qname, job, raw, "worker stopped")
                raise
            except Exception as exc:               # noqa: BLE001
                # `_process_video` records its own failures in coverage, so
                # anything reaching here is a bug in the plane rather than in a
                # pass. Retried three times, then the DLQ — which is why a
                # non-zero DLQ is worth reading.
                verdict = self._jobs.fail(qname, job, raw,
                                          f"{type(exc).__name__}: {exc}")
                self._log(f"{lane.name} {key}: job {verdict.lower()} — "
                          f"{type(exc).__name__}: {str(exc)[:160]}", "error")
                lane.slot = {}
                continue
            self._jobs.ack(qname, job, raw)

    def _prep_loop(self, lane: "Lane") -> None:
        """Stage the next videos' bytes, so no card ever waits on Telegram."""
        while not self._stopping():
            self._wait_if_paused()
            job, raw = self._jobs.claim(jobs.QUEUE_V2_PREP)
            if not job:
                continue
            p = job.get("payload") or {}
            key = str(p.get("video_key") or "")
            ids = _known(p.get("components") or [])
            if not key:
                self._jobs.ack(jobs.QUEUE_V2_PREP, job, raw)
                continue
            ok = False
            try:
                with self._hold(key, lane) as got:
                    if got:
                        ok = self._prep_video(key, ids, lane)
            except (KeyboardInterrupt, SystemExit):
                self._jobs.fail(jobs.QUEUE_V2_PREP, job, raw, "worker stopped")
                raise
            except Exception as exc:               # noqa: BLE001
                self._jobs.fail(jobs.QUEUE_V2_PREP, job, raw,
                                f"{type(exc).__name__}: {exc}")
                lane.slot = {}
                continue
            self._jobs.ack(jobs.QUEUE_V2_PREP, job, raw)
            # Dropped from `_prep_posted` whatever happened, so a video the
            # planner still wants is posted again rather than waiting forever on
            # a hint that has already been consumed. `_staged` is the positive
            # answer, and only success writes it.
            with self._lock:
                self._prep_posted.discard(key)
                if ok:
                    self._staged.add(key)
                    self._prep_fails.pop(key, None)
                    tries = 0
                else:
                    tries = self._prep_fails[key] = \
                        self._prep_fails.get(key, 0) + 1
            # A video that cannot be staged has to stop being a candidate, or the
            # planner offers it every second and the cohort never drains. Failing
            # the rows is the honest way to do that: the reason lands in the
            # coverage table where the matrix shows it, and `revive_failed` gives
            # it another go next rotation rather than this one.
            if tries >= PREP_MAX_TRIES and ids:
                why = f"could not be staged after {tries} attempts"
                for cid in ids:
                    lane.cov.fail(key, cid, why)
                self._bump("failed", len(ids))
                self._log(f"{key}: {why} — its rows are failed so the cohort "
                          f"can finish; the next rotation retries them", "warn")
                with self._lock:
                    self._prep_fails.pop(key, None)

    def _prep_video(self, key: str, ids: list, lane: "Lane") -> bool:
        """Get this video's bytes and derived files onto disk. Claims nothing.

        Two jobs, and neither writes evidence — which is what makes prep safe to
        run for the *next* cohort's videos while this one is still going:

        1. `Source.ensure`, behind the fetch lock. Idempotent, so a video already
           down costs a `stat`.
        2. `_ensure_inputs` for each of this cohort's components — the rebuild of
           `proxy.mp4`, `audio.wav` and the frame tiers that coverage says are
           `done` but eviction or another worker's shard has left absent. That is
           45 s of ffmpeg a GPU lane would otherwise pay for at the head of its
           own pass, and nothing about it needs a claim: `_rebuild` throws the
           emission away because the store already holds it.
        """
        store = lane.store
        video = store.video(key)
        if video is None:
            return False
        workdir = os.path.join(self.cache_dir, _safe(key))
        lane.slot = {"video_key": key, "cohort": -1, "component": "prep",
                     "title": "staging", "since": time.time(), "passes": 0,
                     "lane": lane.name, "device": -1}
        try:
            source = self._fetch(video, workdir)
        except intake.SourceError as exc:
            # The same verdict `_process_video` reaches, reached earlier. Marking
            # the rows failed is what stops the planner posting this video again
            # every second for the rest of the session.
            for cid in ids:
                lane.cov.fail(key, cid, str(exc))
            self._log(f"{key}: {exc} (staging)", "warn")
            self._bump("failed", len(ids))
            lane.slot = {}
            return False
        except Exception as exc:                   # noqa: BLE001
            self._log(f"{key}: staging failed — {type(exc).__name__}: "
                      f"{str(exc)[:160]}", "warn")
            lane.slot = {}
            return False
        intake.touch(workdir)

        for cid in ids:
            if self._stopping():
                break
            comp = registry.get(cid)
            if not (set(comp.needs) & {"artifacts", "keyframes", "allframes"}):
                continue
            job = Job(video=dict(video), component=comp, store=store,
                      source=source, workdir=workdir,
                      params=dict(comp.params), resources=self.resources,
                      cache=lane.cache,
                      renew=lambda progress="": None,
                      progress=lambda d, _l=lane: self._progress(d, _l),
                      log=lambda m, _l=lane: self._note(m, _l))
            try:
                self._ensure_inputs(job, comp)
            except Exception as exc:               # noqa: BLE001
                # Never fatal to staging: the pass lane runs `_ensure_inputs`
                # again for itself and will report the same fault where it can
                # be attributed to a component.
                self._log(f"{key}: staging {cid}'s inputs raised "
                          f"{type(exc).__name__}: {str(exc)[:120]}", "warn")
        lane.videos += 1
        lane.slot = {}
        return True

    def _cloud_loop(self, lane: "Lane") -> None:
        """The cloud lane, and why it is allowed across the cohort barrier.

        `narrate-cloud` is 25 s of network and no local compute, and topo order
        puts it after `describe`, `detect` and `ocr` — but position in the plan
        is not what makes crossing safe. *Checking* is. This lane reads the
        video's coverage row and hands the components to `_process_video`, whose
        dependency gate refuses anything whose hard needs are not `done` and
        defers it for sixty seconds. So the lane can be a cohort behind or a
        cohort ahead and still never run on an input that does not exist.

        What it buys: a network pass never holds a serial slot again, and a key
        that is sitting there unused starts contributing from the first video
        whose description lands.
        """
        while not self._stopping():
            self._wait_if_paused()
            job, raw = self._jobs.claim(jobs.QUEUE_V2_CLOUD)
            if not job:
                continue
            p = job.get("payload") or {}
            key = str(p.get("video_key") or "")
            ids = _known(p.get("components") or [])
            if not key or not ids:
                self._jobs.ack(jobs.QUEUE_V2_CLOUD, job, raw)
                continue
            try:
                with self._hold(key, lane) as got:
                    if got:
                        self._process_video(key, ids, -1, lane)
            except (KeyboardInterrupt, SystemExit):
                self._jobs.fail(jobs.QUEUE_V2_CLOUD, job, raw, "worker stopped")
                raise
            except Exception as exc:               # noqa: BLE001
                self._jobs.fail(jobs.QUEUE_V2_CLOUD, job, raw,
                                f"{type(exc).__name__}: {exc}")
                lane.slot = {}
                continue
            self._jobs.ack(jobs.QUEUE_V2_CLOUD, job, raw)

    # ── the cross-cohort problem ─────────────────────────────────────────
    def _ensure_inputs(self, job: Job, comp) -> None:
        """Rebuild the derived files a later cohort needs but did not make.

        Cohort 1 reads `proxy.mp4`, `audio.wav` and `frames/index.json`, all of
        which were produced in cohort 0 — and may since have been evicted, or
        may have been produced by another worker entirely and arrived here as
        a shard. The coverage table says those passes are `done`, so they will
        never run again, and every pass in cohort 1 would skip with "artifact
        is missing".

        Regenerating is cheap and correct: both functions are deterministic
        given the source and the shot table, they write the same paths, and
        their claims collapse on the uid. The emission is discarded because the
        store already holds it.
        """
        needs = set(comp.needs)
        if not needs & {"artifacts", "keyframes", "allframes"}:
            return

        # The artifact table is the authority on what *should* be on disk. A
        # reel with no audio stream has no `wav` row, and checking the disk
        # alone would rebuild every artifact it has, for every audio pass, on
        # every silent video — an ffmpeg re-encode to discover the same silence.
        held = job.store.artifacts(job.key)
        if "artifacts" in needs:
            gone = [name for name, path in _ARTIFACT_FILES.items()
                    if name in held and not os.path.exists(job.path(path))]
            if gone:
                self._rebuild(job, "artifacts",
                              f"{', '.join(gone)} evicted or made elsewhere")
        if "keyframes" in needs and not self._keyframes_present(job):
            self._rebuild(job, "keyframes",
                          "shot frames evicted or made elsewhere")

        # The complete frame set is the expensive one, and the one every
        # perception pass now depends on. Without this branch a cohort that
        # starts after the workdir was evicted would see every full-coverage
        # pass skip with "no allframes manifest" — coverage marked `allframes`
        # done in an earlier cohort, so it will never run again on its own.
        #
        # The manifest alone is not proof. It lists the frames by name, and the
        # cache eviction that removed the JPEGs may well have left the small
        # JSON behind, so a spot check on the first and last frame of the tier
        # this component actually reads is what decides. Two `os.path.exists`
        # calls against a re-extraction that costs half a minute.
        if "allframes" in needs and not self._allframes_present(job, comp):
            self._rebuild(job, "allframes",
                          "complete frame set evicted or made elsewhere")

    @staticmethod
    def _keyframes_present(job: Job) -> bool:
        """Is the shot-frame set on disk, or only the index that names it?

        `frames/index.json` is a few hundred bytes and the JPEGs beside it are
        tens of megabytes, so an eviction — or a partial arrival from another
        worker — can leave the index describing frames that are gone. Every
        keyframe consumer reads the index and hands the paths straight to
        `cv2.imread`, which answers `None` and logs
        `imread_('.../frames/s0010_0003.jpg'): can't open/read file` once per
        frame before the pass averages an empty list. Checking the first and
        last frame the index names costs two `os.path.exists` calls; the same
        spot check `allframes` already gets.
        """
        ipath = job.path("frames/index.json")
        if not os.path.exists(ipath):
            return False
        try:
            with open(ipath, "r", encoding="utf-8") as fh:
                index = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(index, list) or not index:
            return False
        for probe in (index[0], index[-1]):
            path = str((probe or {}).get("path") or "")
            if not path or not os.path.exists(path):
                return False
        return True

    @staticmethod
    def _allframes_present(job: Job, comp) -> bool:
        """Is the frame set this component reads actually on disk right now?"""
        root = job.path("allframes")
        mpath = os.path.join(root, "manifest.json")
        if not os.path.exists(mpath):
            return False
        try:
            with open(mpath, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return False

        entries = manifest.get("frames") or []
        if not entries:
            return False
        tier = str((comp.params or {}).get("tier") or "analysis")
        if tier == "full" and not manifest.get("full_tier"):
            # The extractor was asked for one tier and this component wants the
            # other. Re-extract, so OCR reads source resolution as configured
            # instead of silently falling back to 384 px for the whole archive.
            return False
        for probe in (entries[0], entries[-1]):
            rel = str(probe.get("file") or "")
            if tier != "analysis" and rel.startswith("analysis/"):
                rel = tier + rel[len("analysis"):]
            if not os.path.exists(os.path.join(root, rel.replace("/", os.sep))):
                return False
        return True

    def _rebuild(self, job: Job, what: str, why: str) -> None:
        """Re-run a structural pass whose output this cohort needs.

        Safe to do at any time: both passes are deterministic given the source
        and the shot table, they write the same paths, and their claims collapse
        on the uid. The emission is thrown away — the store already holds it,
        from the run that marked the pass `done`.
        """
        comp = registry.get(what)
        fn = get_runner(what)
        sub = Job(video=job.video, component=comp, store=job.store,
                  source=job.source, workdir=job.workdir,
                  params=dict(comp.params), resources=job.resources,
                  cache=job.cache, renew=job.renew, log=job.log)
        t0 = time.time()
        try:
            fn(sub)
            self._log(f"{job.key}: rebuilt {what} in {time.time() - t0:.1f}s "
                      f"({why})")
        except SkipPass as exc:
            # Say nothing louder than a note: the real pass is about to raise
            # its own, more specific skip, and that is the one worth reading.
            self._log(f"{job.key}: cannot rebuild {what} — {exc}", "warn")
        except Exception as exc:
            self._log(f"{job.key}: rebuilding {what} failed — "
                      f"{type(exc).__name__}: {exc}", "warn")

    # ── artifacts: the proxy that was being computed and thrown away ──────
    def _upload_artifacts(self, key: str, made: list, store: Store,
                          lane: "Lane") -> None:
        """Put the proxy, poster, sprite and waveform in the channel.

        **This never raises and never fails the pass.** By the time it runs the
        emission is already in the database and `cov.done` is one line away; the
        files are an optimisation over evidence that is already safe, exactly as
        `capture.assets.publish_assets` treats clips. A reel that plays from its
        original instead of its proxy is slower, not broken.

        Threaded as replies to the video's own channel message, so the artifacts
        travel with the thing they describe and a channel export keeps them
        together. `allow_sending_without_reply` is set inside `send_document`, so
        a video message deleted by hand costs the threading and not the upload.

        Re-runs cost nothing. `revive_failed` flips a `failed` row back to
        `queued` every rotation, so this method has to assume it will see the
        same video again: an artifact already recorded with a real `msg_id` and
        the same byte count is left alone rather than posted twice.
        """
        if not (self._tg and self._tg.token):
            return
        anchor = 0
        try:
            anchor = int((store.video(key) or {}).get("msg_id") or 0)
        except (TypeError, ValueError):
            anchor = 0

        try:
            have = store.artifacts(key)
        except Exception:                                   # noqa: BLE001
            have = {}
        by_name = {a["name"]: a for a in made}

        sent, kept, failed = 0, 0, []
        for name in _UPLOAD_ARTIFACTS:
            a = by_name.get(name)
            if not a:
                continue
            path = a.get("path") or ""
            try:
                size = os.path.getsize(path) if os.path.isfile(path) else 0
            except OSError:
                size = 0
            if not size:
                continue

            old = have.get(name) or {}
            try:
                same = bool(old.get("msg_id")) and \
                    int(old.get("bytes") or 0) == size
            except (TypeError, ValueError):
                # Nothing writes a non-numeric size, but this method's promise is
                # that it cannot fail the pass — and the emission is already in
                # the database by the time it runs, so a raise here would fail a
                # video over a thumbnail. Unreadable row → treat it as absent.
                same = False
            if same:
                # Same bytes, already in the channel. Nothing to do, and saying
                # so would be noise on every revived video.
                kept += 1
                continue

            if size > BOT_UPLOAD_LIMIT:
                # `send_document` has no MTProto fallback — only `send_video`
                # does — so this would raise inside the loop and lose the
                # artifacts queued behind it. A 480p proxy over 50 MB means a
                # very long video, which is the case where the original is
                # unpleasant to stream and the proxy would have helped most, so
                # it is worth a warning rather than silence.
                failed.append(f"{name} is {size / 1048576:.0f} MB, over the "
                              f"Bot API's upload cap")
                continue

            try:
                with self._upload_lock:
                    got = self._tg.send_document(
                        path, caption=f"{name} · vios:{key}",
                        reply_to=anchor or None,
                        file_name=f"{key}-{os.path.basename(path)}")
                    if _ARTIFACT_PAUSE > 0:
                        time.sleep(_ARTIFACT_PAUSE)
            except UploadError as exc:
                failed.append(f"{name}: {str(exc)[:120]}")
                continue
            except Exception as exc:                        # noqa: BLE001
                failed.append(f"{name}: {type(exc).__name__}: "
                              f"{str(exc)[:120]}")
                continue

            # Same reasoning as the byte comparison above: `send_document`
            # returns the Bot API's `result` object and a message id in it is an
            # integer, but an unexpected shape must not fail the pass.
            try:
                msg_id = int((got or {}).get("message_id") or 0) or None
            except (AttributeError, TypeError, ValueError):
                msg_id = None
            if not msg_id:
                failed.append(f"{name}: upload returned no message id")
                continue
            try:
                store.set_artifact(key, name, msg_id,
                                   got.get("file_id") or "", size, a["meta"])
            except Exception as exc:                        # noqa: BLE001
                # The file is in the channel but this database does not know
                # where. Worth an error line, because the next run will upload
                # it again and the duplicate is the visible symptom.
                self._log(f"{key}: {name} uploaded as message {msg_id} but the "
                          f"row would not update — {type(exc).__name__}: "
                          f"{exc}", "error")
                continue
            sent += 1

        if sent:
            self._progress(f"artifacts: {sent} uploaded", lane=lane)
            self._log(f"{key}: {sent} artifact(s) uploaded"
                      + (f", {kept} already in the channel" if kept else "")
                      + (f" — threaded under message {anchor}" if anchor
                         else " (no anchor message, posted unthreaded)"))
        if failed:
            self._log(f"{key}: artifact(s) not uploaded — "
                      + "; ".join(failed[:4]), "warn")

    # ══════════════════════════════════════════════════════════════════════
    # Unloading and publishing
    # ══════════════════════════════════════════════════════════════════════

    def _unload(self, why: str = "", lane: "Lane | None" = None) -> None:
        """Drop every resident model and say what actually came back.

        The reclaimed total is logged against the cohort that held the models,
        because that is the only moment the number is attributable. A leak
        found later — as an OOM in the next cohort — names the wrong pass.

        With no `lane`, every lane's cache plus the sweep's own: that is the
        cohort boundary, where by construction no lane is mid-pass. With a lane,
        only that lane's — an OOM on card 0 is not a reason to drop card 1's
        weights while it is using them.

        "By construction" has one hole, and it is skipped rather than trusted: a
        cohort that hits `_COHORT_DRAIN_TIMEOUT` moves on with a lane still
        inside a pass, and freeing that lane's weights from under it is a
        segfault, not an exception. A busy lane keeps its cache; the next
        boundary it is idle at collects it.
        """
        if lane is not None:
            self._unload_cache(lane.cache, why, lane.name)
            return
        for l in list(self._lanes):
            if l.cache is None:
                continue
            if l.busy():
                held = len(l.cache.loaded())
                if held:
                    self._log(f"[{l.name}] still mid-pass — keeping its "
                              f"{held} resident model(s) until it is idle",
                              "warn")
                continue
            self._unload_cache(l.cache, why, l.name)
        self._unload_cache(self._cache, why, "")

    def _unload_cache(self, cache: ModelCache, why: str, who: str) -> None:
        tag = f"[{who}] " if who else ""
        loaded = cache.loaded()
        expected = sum(cache.footprints().get(k, 0) for k in loaded)
        before = ModelCache.vram()
        freed = cache.unload_all()
        after = ModelCache.vram()
        if not loaded:
            return

        line = (f"{tag}Unloaded {len(loaded)} models — {freed} MB reclaimed"
                + (f" of {expected} MB held" if expected else "")
                + (f" — {why}" if why else ""))
        self._log(line)
        if after:
            self._log(f"{tag}VRAM now {after.get('allocated', 0)} MB allocated, "
                      f"{after.get('free', 0)} MB free of "
                      f"{after.get('total', 0)}; resident: "
                      f"{', '.join(cache.loaded()) or 'nothing'}")
        # Still resident after an unload_all means a drop raised and the
        # reference survived. Loud, because the packer's next plan is wrong.
        if cache.loaded():
            self._log(f"{tag}Models still resident after unload: "
                      f"{', '.join(cache.loaded())} — the next cohort "
                      f"is planned against VRAM that is not free", "error")
        elif expected >= 512 and freed < expected // 2 and before:
            self._log(f"{tag}Only {freed} MB of {expected} MB came back after "
                      f"{why or 'unload'} — the next cohort has that much "
                      f"less than the packer assumed", "warn")

    def _publish_due(self) -> bool:
        """Is it time to push a shard?

        Three conditions, and the answer is yes when the count is met *and* the
        rate limit has passed, or when the age limit has been exceeded whatever
        the count.

        The user's question was "will this export work automatically, and
        perfectly, after each video is processed?" — and the honest answer for
        the previous build was no: `PUBLISH_EVERY = 50` meant a session could
        run for hours and be killed with everything still local. Publishing on
        every video is the right default because a video is the unit of work
        that is either wholly done or not done at all. The two clocks are what
        keep that from degenerating: the floor stops a fast cohort from filling
        the channel with tiny files, and the ceiling makes sure a slow one still
        checkpoints, so the maximum loss is bounded by time rather than by how
        many reels happened to fit in a cohort.
        """
        with self._lock:
            n = self.since_publish
            last = self.last_publish or 0.0
        if n <= 0:
            return False
        age = time.time() - last if last else float("inf")
        if age >= PUBLISH_MAX_SECONDS:
            return True
        if n < max(1, int(self.publish_every)):
            return False
        return age >= PUBLISH_MIN_SECONDS

    def _publish(self, note: str, store: Store | None = None) -> str:
        """Export everything written since the last shard and send it up.

        Best effort by design. A session with no bot token still produces a
        complete local database, and a shard that fails to upload stays on disk
        with its path recorded, so `_retry_held` re-offers it on later ticks
        rather than losing it. The watermark still advances past it — see the
        comment at the advance — so the retry is the whole of the recovery, and
        `held_shards()` is what makes an unrecovered one visible.

        A shard that *does* upload no longer stays on disk. It used to, for the
        length of the session, because nothing here and nothing in `store` ever
        unlinked it — see `_reclaim_shards`. And under `DISK_FLOOR_MB` this method
        writes nothing at all: the rows are already durable in the evidence
        database, so it defers rather than adding a copy there is no room for.

        Runs under `_pub_lock` from start to finish, so the read of the watermark,
        the export against it and the write that advances it are one operation
        even though three threads can reach this method. `_lock` is taken only
        around the counters — an upload must not hold the lock the status endpoint
        reads, or the tab freezes for as long as Telegram is slow.

        `store` is the connection to work on. The publisher thread passes its own
        (`Store.secondary`); everyone else gets the engine's. Two threads on one
        SQLite connection share one implicit transaction, and this method commits.
        """
        store = store or self.store
        with self._pub_lock:
            self._retry_held(store)
            lo_id = int(store.get_meta("shard_lo_id", "0") or 0)
            lo_vec = int(store.get_meta("shard_lo_vec", "0") or 0)
            # Per-frame rows are watermarked separately for the same reason
            # claims and vectors always were: a cohort of embedding passes
            # writes `frame_vector` rows and nothing else, so keying their
            # export off the claim range would upload an empty shard and then
            # advance past the work as though it had been published.
            lo_fvec = int(store.get_meta("shard_lo_fvec", "0") or 0)
            lo_fmet = int(store.get_meta("shard_lo_fmet", "0") or 0)
            hi_id = store.max_claim_id()
            hi_vec = store.max_vector_id()
            hi_fvec = store.max_frame_vector_id()
            hi_fmet = store.max_frame_metric_id()
            if (hi_id <= lo_id and hi_vec <= lo_vec
                    and hi_fvec <= lo_fvec and hi_fmet <= lo_fmet):
                with self._lock:
                    self.since_publish = 0
                return ""

            # There are rows. Before writing a 40 MB copy of them, check there is
            # room for one — and reclaim before deciding, because the likeliest
            # thing occupying this directory is the engine's own published
            # shards.
            #
            # A shard is a copy made for transport; the rows themselves are
            # already durable in the evidence database, which is on this same
            # disk and still has to grow. So under the floor the answer is to
            # wait and say so: the watermark has not moved, the rows are still
            # queued, and the next tick tries again.
            #
            # Deliberately the opposite trade from `_mark_stage_pending`, which
            # *discards* held bytes under this same floor. A stage bundle is a
            # snapshot that can be rebuilt from the database; a held shard is the
            # only copy of rows the watermark has stepped over, so deleting one to
            # make room is the single thing this engine must not do. The pressure
            # is relieved by not making more, and the line below names how much of
            # it is held evidence so an operator can tell which problem this is.
            free = intake.free_mb(self.shard_dir)
            if free and free < self.disk_floor_mb:
                back = self._reclaim_shards(store)
                if back["removed"]:
                    self._log(f"Reclaimed {back['bytes'] / 1048576:.1f} MB "
                              f"from {back['removed']} shard file(s) the "
                              f"channel already has")
                free = intake.free_mb(self.shard_dir)
            if free and free < self.disk_floor_mb:
                try:
                    n_held, held_bytes = store.held_shard_bytes()
                except Exception:               # noqa: BLE001
                    n_held, held_bytes = 0, 0
                with self._lock:
                    # Reads as "just published" to `_publish_due`, so the retry
                    # waits out PUBLISH_MIN_SECONDS rather than repeating this
                    # line every ten seconds. `since_publish` is left alone: the
                    # rows are still unpublished and the next attempt must know.
                    self.last_publish = time.time()
                self._log(
                    f"Shard deferred — {free} MB free on {self.shard_dir} is "
                    f"under the {self.disk_floor_mb} MB floor"
                    + (f", with {n_held} held shard(s) "
                       f"({held_bytes / 1048576:.1f} MB) the channel has not "
                       f"taken" if n_held else "")
                    + ". The rows stay in the evidence database and the "
                      "watermark has not moved, so nothing is lost unless the "
                      "session ends here.", "warn")
                return ""

            # The quick early return is the common case for a cohort that
            # wrote nothing. Anything past it is a real export, and the
            # age gate that decided it deserves a real watermark.
            with self._lock:
                self.last_publish = time.time()

            seq = int(store.get_meta("shard_seq", "0") or 0) + 1
            sid = f"{intake.site_id(store)}-{seq:04d}"
            path = os.path.join(self.shard_dir, intake.shard_name(sid))
            try:
                stats = store.export_shard(path, lo_id, hi_id, lo_vec, hi_vec,
                                           note, lo_fvec, hi_fvec,
                                           lo_fmet, hi_fmet,
                                           budget_bytes=SHARD_BUDGET_BYTES)
            except Exception as exc:
                self._log(f"Shard export failed: {type(exc).__name__}: {exc}",
                          "error")
                return ""

            frame_note = ""
            if stats["frame_vectors"] or stats["frame_metrics"]:
                frame_note = (f"\n{stats['frame_vectors']} frame-vector rows, "
                              f"{stats['frame_metrics']} frame-metric rows")
            cov_note = ""
            if stats.get("coverage"):
                cov_note = f"\n{stats['coverage']} coverage rows"
            msg_id = None
            if self._tg and self._tg.token:
                caption = (f"vios evidence · {sid}\n"
                           f"{stats['claims']} claims, {stats['vectors']} "
                           f"vectors{frame_note}{cov_note}\nworker "
                           f"{self.index + 1}/{self.partitions} · {note}")
                try:
                    res = self._tg.send_document(
                        path, caption, file_name=intake.shard_name(sid))
                    msg_id = res.get("message_id")
                except UploadError as exc:
                    self._log(f"Shard {sid} not uploaded: {exc}", "warn")
                except Exception as exc:
                    self._log(f"Shard {sid} not uploaded: "
                              f"{type(exc).__name__}: {exc}", "warn")

            store.note_shard(sid, note, msg_id, stats)
            # Only advance the watermark once the rows are safely in a file, and
            # only over the rows that are actually *in* it — `stats["hi_*"]` is
            # where the export stopped, not where it was asked to stop, so a
            # shard cut at `SHARD_BUDGET_BYTES` leaves the remainder queued for
            # the next one instead of stepping over it.
            #
            # The advance happens even when the upload failed, which is the part
            # that used to lose evidence: the file is the only copy, and this
            # docstring's promise that "the next attempt includes it" was never
            # true. `_retry_held` above is what makes it true — the row keeps its
            # path and gets re-offered every tick until it goes or runs out of
            # attempts. Not advancing instead would be worse: the same rows would
            # be re-exported into a bigger file every time, and a channel that is
            # merely slow today would never catch up.
            store.set_meta("shard_lo_id", str(stats["hi_id"]))
            store.set_meta("shard_lo_vec", str(stats["hi_vec"]))
            store.set_meta("shard_lo_fvec", str(stats["hi_fvec"]))
            store.set_meta("shard_lo_fmet", str(stats["hi_fmet"]))
            store.set_meta("shard_seq", str(seq))
            store.checkpoint()

            # Only now is the local copy scratch: the row carries its message id
            # and the four watermarks past it are committed. Same order and the
            # same reason as `_finish_stage`. Without this line the file stayed in
            # `shard_dir` for the rest of the session with nothing that would ever
            # read it again — 30 of them, 135.8 MB, in the busiest session this
            # archive holds, and no mechanism in the engine looks at this
            # directory at all.
            if msg_id:
                self._drop_shard_file(path, sid)

            with self._lock:
                self.since_publish = 0
                self.last_publish = time.time()
                self.session["shards"] += 1
                if stats.get("cut"):
                    # There is more behind this shard, and `_publish_due` gates
                    # on `since_publish > 0`. Left at zero the remainder would
                    # wait for the next video to finish — which, on a stage that
                    # has just written its last frame vectors, is never.
                    #
                    # `last_publish = 0` reads as "no publish yet" in
                    # `_publish_due`, so the age gate returns True and the next
                    # publisher tick (≤ PUBLISH_TICK_SECONDS) takes the
                    # remainder rather than waiting out PUBLISH_MIN_SECONDS. That
                    # floor exists to stop a fast cohort filling the channel with
                    # tiny files; a shard that just hit a 40 MB cut is the
                    # opposite case, and the upload itself is the rate limit.
                    self.since_publish = 1
                    self.last_publish = 0.0
            self._log(f"Shard {sid}: {stats['claims']} claims, "
                      f"{stats['vectors']} vectors, "
                      f"{stats['frame_vectors']} frame-vector rows, "
                      f"{stats['frame_metrics']} frame-metric rows, "
                      f"{stats.get('coverage', 0)} coverage rows, "
                      f"{stats['bytes'] / 1024:.0f} KB"
                      + (f" → message {msg_id}" if msg_id else
                         " (held locally — no upload)"))
            if stats.get("cut"):
                want = stats.get("want") or {}
                self._log(
                    f"Shard {sid} cut at "
                    f"{SHARD_BUDGET_BYTES / 1024 / 1024:.0f} MB — claims to "
                    f"{stats['hi_id']}/{want.get('hi_id', stats['hi_id'])}, "
                    f"vectors to {stats['hi_vec']}/"
                    f"{want.get('hi_vec', stats['hi_vec'])}, frame vectors to "
                    f"{stats['hi_fvec']}/{want.get('hi_fvec', stats['hi_fvec'])}"
                    f", frame metrics to {stats['hi_fmet']}/"
                    f"{want.get('hi_fmet', stats['hi_fmet'])}; the rest "
                    f"publishes next tick", "info")
            if msg_id is None and self._tg and self._tg.token:
                return ""
            return sid

    def _retry_held(self, store: Store) -> int:
        """Re-offer shards that were written and never published.

        Called at the top of every `_publish`, under `_pub_lock`, before the new
        export — a held shard is older evidence than anything about to be
        written, and the channel is read back in `created_at` order on restore.

        Returns the number that went. Best effort in the same sense as the rest
        of this method: nothing here can fail a session.
        """
        if not (self._tg and self._tg.token):
            return 0
        try:
            held = store.held_shards(HELD_SHARD_PER_TICK, HELD_SHARD_ATTEMPTS)
        except Exception:
            return 0                        # pre-migration database; nothing held
        sent = 0
        for row in held:
            sid, path = row["shard_id"], row["path"]
            tries = int(row.get("attempts") or 0)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            caption = (f"vios evidence · {sid} (held, attempt {tries + 1})\n"
                       f"{row.get('claims', 0)} claims, "
                       f"{row.get('vectors', 0)} vectors\nworker "
                       f"{self.index + 1}/{self.partitions} · "
                       f"{row.get('component') or 'retry'}")
            try:
                res = self._tg.send_document(
                    path, caption, file_name=intake.shard_name(sid))
                store.mark_shard_sent(sid, int(res.get("message_id") or 0))
                sent += 1
                # The row has let go of the path; this lets go of the bytes. They
                # were kept only because the file was the sole copy of rows the
                # watermark had already stepped over, and it is not any more.
                self._drop_shard_file(path, sid)
                self._log(f"Held shard {sid} went up on attempt {tries + 1} "
                          f"({size / 1024:.0f} KB) → message "
                          f"{res.get('message_id')}")
            except Exception as exc:
                n = store.held_shard_failed(sid)
                # One line per attempt, at `warn`, because the alternative is a
                # session that quietly ends with megabytes of evidence on a disk
                # that is about to be deleted. The count is in the message so
                # the last one before the cap is recognisable as the last one.
                self._log(f"Held shard {sid} still not uploaded "
                          f"({size / 1024:.0f} KB, attempt {n} of "
                          f"{HELD_SHARD_ATTEMPTS}): {type(exc).__name__}: "
                          f"{exc}", "warn")
        return sent

    def _drop_shard_file(self, path: str, sid: str = "") -> int:
        """Delete a shard file the channel has taken. Returns bytes freed.

        Only ever called on the far side of the write that records the message
        id — the same order `_finish_stage` uses, for the same reason. The row is
        what makes the file redundant, so the row goes first; losing the process
        in between costs a duplicate file that the next sweep collects, and doing
        it the other way round costs the evidence.
        """
        try:
            n = os.path.getsize(path)
        except OSError:
            return 0
        try:
            os.remove(path)
        except OSError as exc:
            self._log(f"Shard {sid or os.path.basename(path)} is in the channel "
                      f"but its local copy could not be removed: {exc}", "warn")
            return 0
        return n

    def _reclaim_shards(self, store: Store | None = None) -> dict:
        """Delete every shard file whose row says the channel already has it.

        Not a cache and not an optimisation: each of these is a second copy of
        evidence that is in the channel *and* in the evidence database beside it,
        and nothing in this engine will read one again. Until this existed
        nothing deleted them at all. `mark_shard_sent` clears the `path` column
        without touching the file, `_publish` never unlinked what it had just
        uploaded, and `intake.evict` walks `cache_dir` for working *directories*
        — so `shard_dir` sat outside every mechanism the engine has for disk
        pressure, and a published shard was permanent for the length of the
        session. Measured in this archive's own record: 74 shard files across
        nine sessions, every one of them published, 248.7 MB in total, of which
        135.8 MB was the single session that cut 30.

        Matched by name against published rows rather than by looking for
        anything shard-shaped, so a held shard, a stage bundle, a `.partNNN` and
        whatever a later version writes into this directory are all untouchable
        by construction. Kaggle's working directory outlives a kernel restart and
        the evidence database sits beside these files, so a resumed session
        inherits both the rows and the files; this is what reconciles them.
        """
        store = store or self.store
        out = {"removed": 0, "bytes": 0}
        try:
            rows = store.shards()
        except Exception:                       # noqa: BLE001
            return out                          # pre-migration; nothing to match
        for row in rows:
            if not row.get("msg_id"):
                continue
            n = self._drop_shard_file(
                os.path.join(self.shard_dir,
                             intake.shard_name(row["shard_id"])),
                row["shard_id"])
            if n:
                out["removed"] += 1
                out["bytes"] += n
        return out

    # ── the publisher thread ─────────────────────────────────────────────
    #
    # Everything above is the work of publishing; everything below is *when*.
    # The split exists because a worker asking "have I produced enough to
    # checkpoint?" and a worker waiting for the answer to be uploaded are two
    # different things, and only the first belongs on the critical path.

    def _publisher_store(self) -> Store:
        """The publisher's own connection, opened on first use."""
        if self._pub_store is None:
            self._pub_store = self.store.secondary()
        return self._pub_store

    def _start_publisher(self) -> None:
        t = self._pub_thread
        if t is not None and t.is_alive():
            return
        self._pub_wake.clear()
        self._pub_thread = threading.Thread(
            target=self._publisher_loop, name="vios-publish", daemon=True)
        self._pub_thread.start()

    def _publisher_loop(self) -> None:
        """Wake on a signal or on a tick, publish if either clock says so.

        The tick is what makes `PUBLISH_MAX_SECONDS` mean anything: a cohort of
        long passes can go five minutes without finishing a video, and the whole
        point of the ceiling is that such a cohort still checkpoints. A loop that
        only woke on a signal would sleep straight through it.

        It exits when `_stop` is set *or* when it is no longer the engine's
        publisher — `_drain_publisher` retires it by clearing `_pub_thread`, and
        that has to be a second condition rather than a second flag because the
        two paths that end a sweep without stopping it (a crash, and a one-shot
        run reaching the end of its work) would otherwise leave the drain
        waiting on a thread that had no reason to exit.
        """
        me = threading.current_thread()
        while not self._stop.is_set() and self._pub_thread is me:
            self._pub_wake.wait(PUBLISH_TICK_SECONDS)
            self._pub_wake.clear()
            if self._stop.is_set() or self._pub_thread is not me:
                break
            with self._lock:
                note = self._pub_note or "rows"
                forced = self._pub_force
                self._pub_force = False
            if not (forced or self._publish_due()):
                continue
            try:
                self._publish(note, store=self._publisher_store())
            except Exception as exc:            # noqa: BLE001
                # A publisher that dies takes every later checkpoint with it, so
                # it does not die. The shard stays local with its watermark
                # unmoved, which is the same state a failed upload leaves.
                self._log(f"Publisher tick failed: {type(exc).__name__}: "
                          f"{str(exc)[:200]}", "error")
                self._stop.wait(5.0)

    def _rows_written(self, note: str, force: bool = False) -> None:
        """A worker's whole involvement in publishing: say that rows exist.

        `force` skips the count and floor gates but not the publisher — a cohort
        boundary is a checkpoint worth taking immediately, and still not worth
        making a GPU wait for.
        """
        with self._lock:
            self._pub_note = note
            if force:
                self._pub_force = True
        self._pub_wake.set()

    def _drain_publisher(self, note: str) -> str:
        """Retire the publisher thread and publish what is left, synchronously.

        For the moments where "eventually" is not good enough: the sweep
        finishing, and this process being asked to exit. Everything else signals.

        The thread reference is taken under `_lock` so two callers racing here —
        `shutdown()` and the sweep's `finally`, or a signal arriving during
        either — cannot both try to join it. Clearing `_pub_thread` is also what
        tells the loop to exit, so a drain on a path that has not set `_stop`
        still terminates.

        It publishes on the *publisher's* connection, not the engine's. This is
        called from three threads and one of them is a signal handler running on
        the main thread while the sweep may be mid-pass: two threads sharing one
        SQLite connection share one implicit transaction, so exporting on
        `self.store` from here could commit the sweep's half-written batch and
        invalidate the cursor it is reading. `_pub_lock` keeps this connection
        single-user even if the retired thread outlives the join.
        """
        with self._lock:
            t = self._pub_thread
            self._pub_thread = None
            live = self._store is not None
        self._pub_wake.set()
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=180)
        if not live:
            return ""
        try:
            return self._publish(note, store=self._publisher_store())
        except Exception as exc:                # noqa: BLE001
            self._log(f"Final publish failed: {type(exc).__name__}: "
                      f"{str(exc)[:200]}", "error")
            return ""

    def flush_for_exit(self, why: str = "exit") -> dict:
        """Publish the work in hand because this process is going away.

        Called from the signal handlers and from `atexit`. A Kaggle session ends
        by having its container removed, so a shard still on local disk is a shard
        that never existed — the last window of work is only real once it is in
        the channel. Idempotent: two paths can both fire and only the first
        publishes.
        """
        with self._lock:
            if self._exit_flushed:
                return {"ok": False, "reason": "already flushed"}
            self._exit_flushed = True
            # `self._store`, not `self.store`: the property opens the database on
            # first touch, and a process exiting without ever having run the
            # engine must not create one on its way out.
            live = self._store is not None
        if not live:
            return {"ok": False, "reason": "engine never opened a database"}
        self._log(f"Flushing before exit ({why}) — publishing the last window "
                  f"of work")
        sid = self._drain_publisher(f"exit-{why}")
        try:
            self._store.checkpoint()
        except Exception:                       # noqa: BLE001
            pass
        return {"ok": bool(sid), "shard": sid}

    def publish_now(self) -> dict:
        """The tab's manual button. Also the right thing to press before
        closing a Kaggle session early.

        On the publisher's connection, because this arrives on a Flask request
        thread and the sweep may be mid-pass on the engine's own — same reason
        `_drain_publisher` uses it.
        """
        store = self._publisher_store() if self._store is not None else None
        sid = self._publish("manual", store=store)
        return {"ok": bool(sid), "shard": sid}


    # ── stage bundles ────────────────────────────────────────────────────
    #
    # A shard is a delta and a bundle is a database. Both are needed and they
    # answer different questions. The shard is what makes ten workers converge:
    # it replays into any database by uid, idempotently, in any order. The
    # bundle is what makes one *stage* debuggable on its own — it is the whole
    # file as it stood when perception finished, with the stage's own error list
    # beside it, so "language went wrong" can be investigated without first
    # replaying four hundred shards to reconstruct the state it went wrong from.
    #
    # Coverage lives inside `evidence.db` (`Coverage` is constructed on the
    # store's own connection), so one file carries both what was learned and
    # what is left to learn, consistent with each other. That is why the bundle
    # is one database and not two.

    def _stage_signature(self, stage: str, components: list,
                         report: dict) -> str:
        """A fingerprint of a stage's standing, used to decide whether to build.

        Re-uploading an identical database because a later cohort happened to
        re-enter the same stage would cost a multi-hundred-megabyte transfer and
        add nothing, so a bundle is cut only when this string changes.

        It delegates to `Coverage.stage_fingerprint`, which hashes the rows
        themselves. The counter this used to be — `total:done:skipped:failed:
        claims:vectors` — moved every time `revive_failed()` flipped one row from
        `failed` to `queued`, which happens once per rotation, so the same
        database shipped twice. The counter survives only as the fallback for a
        fingerprint that raises: cutting a duplicate bundle is a waste, and not
        cutting one at all is a lost snapshot, so the degradation goes the way
        that keeps the snapshot.
        """
        try:
            return self.coverage.stage_fingerprint(components)
        except Exception as exc:                # noqa: BLE001
            c = report.get("counts", {})
            self._log(f"stage {stage}: content fingerprint unavailable "
                      f"({type(exc).__name__}) — falling back to counters, "
                      f"which may re-ship an unchanged database", "warn")
            return (f"counts:{c.get('total', 0)}:{c.get('done', 0)}:"
                    f"{c.get('skipped', 0)}:{c.get('failed', 0)}:"
                    f"{report.get('claims', 0)}:{report.get('vectors', 0)}")

    def _error_digest(self, report: dict, limit: int = 12) -> str:
        """The first `limit` distinct (component, message) pairs, plus a total.

        `stage_report`'s `errors` is deliberately unbounded — the full list is the
        reason to read a bundle — and one run produced 218 of them. A Telegram
        caption is capped at 1024 characters, so an unbounded list does not
        truncate gracefully; it takes the caption over the limit and the upload
        fails, which loses the bundle to make room for errors nobody can read in
        a caption anyway.

        Distinct pairs rather than the first twelve rows: forty videos failing
        `ocr-alt` the same way is one fact, and spending the whole budget saying
        it forty times hides the other three causes. The full list stays inside
        the bundle, where it has room.
        """
        errs = report.get("errors") or []
        if not errs:
            return ""
        seen, lines = set(), []
        for e in errs:
            comp = e.get("component") or "?"
            msg = re.sub(r"\s+", " ", str(e.get("error") or "")).strip()[:110]
            k = (comp, msg[:60])
            if k in seen:
                continue
            seen.add(k)
            lines.append(f"• {comp}: {msg}")
            if len(lines) >= limit:
                break
        head = f"{len(errs)} errors, {len(seen)} distinct:"
        out, budget = [head], STAGE_CAPTION_ERROR_BUDGET - len(head)
        for ln in lines:
            if budget - len(ln) - 1 < 0:
                break
            out.append(ln)
            budget -= len(ln) + 1
        if len(out) - 1 < len(lines):
            out.append(f"…and {len(lines) - (len(out) - 1)} more kinds")
        return "\n".join(out)

    def _stage_delta(self, stage: str, report: dict) -> str:
        """What changed since the bundle that is already in the channel.

        A second file for the same stage is legitimate — a revived pass really
        did produce evidence the first snapshot lacked — but a second 135 MB
        upload with an identical-looking caption is indistinguishable from the
        duplicate bug, which is what made `language-0013`/`-0014` so hard to
        read. Naming the delta is what makes revision 2 legible.
        """
        try:
            prev = json.loads(
                self.store.get_meta(f"stage_done:{stage}", "") or "{}")
        except Exception:                       # noqa: BLE001
            prev = {}
        if not prev:
            return ""
        bits = []
        for label, now, before in (
                ("claims", report.get("claims", 0), prev.get("claims", 0)),
                ("vectors", report.get("vectors", 0), prev.get("vectors", 0)),
                ("done", report.get("counts", {}).get("done", 0),
                 (prev.get("counts") or {}).get("done", prev.get("done", 0))),
                ("videos", report.get("videos", 0), prev.get("videos", 0))):
            d = int(now or 0) - int(before or 0)
            if d:
                bits.append(f"{d:+d} {label}")
        return ("changed: " + ", ".join(bits)) if bits else \
               "changed: re-observed, same totals"

    def _stage_bundle_file(self, stage: str, seq: int) -> str:
        return (f"{STAGE_PREFIX}{intake.site_id(self.store)}-{stage}-"
                f"{seq:04d}.tar.gz")

    def _build_stage_bundle(self, stage: str, report: dict, path: str) -> dict:
        """Snapshot the database and the stage report into one gzip tarball.

        `VACUUM INTO` rather than a file copy: it takes its own read transaction,
        so the snapshot is consistent even though this engine is the writer, and
        it compacts free pages that a copy would faithfully reproduce. The copy
        is the fallback for an SQLite too old to have it (3.27, 2019 — Kaggle is
        far newer, but a fallback that costs six lines is cheaper than a bundle
        that does not exist).
        """
        import shutil          # noqa: PLC0415
        import tarfile         # noqa: PLC0415

        work = os.path.join(self.shard_dir, f"_stage-{stage}")
        os.makedirs(work, exist_ok=True)
        snap = os.path.join(work, "evidence.sqlite")
        meta_path = os.path.join(work, "stage.json")
        for p in (snap, meta_path):
            try:
                os.remove(p)
            except OSError:
                pass

        self.store.checkpoint()
        try:
            self.store.conn.execute("VACUUM INTO ?", (snap,))
        except Exception as exc:                # noqa: BLE001
            self._log(f"stage {stage}: VACUUM INTO unavailable "
                      f"({type(exc).__name__}) — falling back to a file copy",
                      "warn")
            shutil.copy2(self.db_path, snap)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        try:
            os.remove(path)
        except OSError:
            pass
        with tarfile.open(path, "w:gz", compresslevel=6) as tar:
            tar.add(meta_path, arcname="stage.json")
            tar.add(snap, arcname="evidence.sqlite")

        out = {"bytes": os.path.getsize(path),
               "db_bytes": os.path.getsize(snap)}
        for p in (snap, meta_path):
            try:
                os.remove(p)
            except OSError:
                pass
        return out

    def _split_parts(self, path: str) -> list:
        """[path] when it fits, or the .partNNN files when it does not.

        The Bot API refuses a document over 50 MB and this uploader has no
        MTProto document path. Splitting is the difference between a large
        archive publishing its bundles and silently publishing none of them.
        """
        size = os.path.getsize(path)
        if size <= STAGE_PART_BYTES:
            return [path]
        parts, idx = [], 0
        with open(path, "rb") as fin:
            while True:
                part = f"{path}.part{idx:03d}"
                written = 0
                with open(part, "wb") as fout:
                    while written < STAGE_PART_BYTES:
                        block = fin.read(min(1 << 20,
                                             STAGE_PART_BYTES - written))
                        if not block:
                            break
                        fout.write(block)
                        written += len(block)
                if written == 0:
                    os.remove(part)
                    break
                parts.append(part)
                idx += 1
                if written < STAGE_PART_BYTES:
                    break
        try:
            os.remove(path)
        except OSError:
            pass
        return parts

    def _publish_stage(self, stage: str, components: list,
                       force: bool = False) -> dict:
        """Cut this group's bundle and send it up. Never raises.

        `stage` is a *group label*, not necessarily a registry stage: it is a
        stage name when a caller asks for one, and a wave name — `spine`,
        `refine`, `deep` — when the rotation cuts on a wave boundary, which is
        the usual case. Nothing below reads it as anything but a label and a
        meta key; `stage_report` takes it purely to name what it describes, and
        `group_members` resolves either kind back on a retry.

        Order is load-bearing, and it is the same order `_publish` uses for the
        same reason: the watermark advances **last**. A bundle whose upload fails
        leaves `stage_done:<stage>` untouched and an entry in `stage_pending`
        holding the built parts, so the next tick re-sends those bytes rather
        than vacuuming and gzipping the database a second time. Nothing is
        recorded as published that is not in the channel.

        `force` now means "cut one even though the content is unchanged" — the
        manual button and nothing else. It used to be passed by every retry,
        which is how a transport hiccup turned into a full rebuild each cohort.

        A failure here never propagates. The bundle is a convenience for
        debugging a stage in isolation; the database and the shards are the
        durable record, and losing a snapshot must not stop the next stage from
        running.
        """
        out = {"stage": stage, "ok": False, "reason": ""}
        try:
            report = self.coverage.stage_report(stage, components)
        except Exception as exc:                # noqa: BLE001
            out["reason"] = f"stage report failed: {type(exc).__name__}: {exc}"
            self._log(f"stage {stage}: {out['reason']}", "error")
            return out

        sig = self._stage_signature(stage, components, report)
        store = self.store
        if not force and store.get_meta(f"stage_sig:{stage}", "") == sig:
            out["reason"] = "unchanged since the last bundle"
            return out
        if not report.get("counts", {}).get("total"):
            out["reason"] = "no rows in this stage"
            return out

        # Revisions are numbered rather than merely sequenced. `stage_seq` is a
        # per-session counter across all stages, so `-0013` and `-0014` said
        # nothing about which was the second look at *language*. This does.
        rev = int(store.get_meta(f"stage_rev:{stage}", "0") or 0) + 1
        delta = self._stage_delta(stage, report) if rev > 1 else ""

        # The delta first, so the shard stream stays continuous and a reader
        # that only follows shards is never behind the bundle.
        self._publish(f"stage-{stage}")

        stats = store.stats()
        report = dict(report)
        report.update({
            "site_id": intake.site_id(store),
            "worker": {"index": self.index, "partitions": self.partitions,
                       "id": self.worker},
            "schema": stats.get("schema"),
            "store": {k: stats.get(k) for k in (
                "videos", "shots", "claims", "frame_claims", "frames_claimed",
                "vectors", "frame_vectors", "frame_vector_frames",
                "frame_metrics", "frame_metric_frames", "observers",
                "artifacts", "shards")},
            "frames": stats.get("frames_claimed", 0),
            "cut_at": time.time(),
            "signature": sig,
            "revision": rev,
            "changed": delta,
        })

        seq = int(store.get_meta("stage_seq", "0") or 0) + 1
        name = self._stage_bundle_file(stage, seq)
        path = os.path.join(self.shard_dir, name)
        try:
            built = self._build_stage_bundle(stage, report, path)
        except Exception as exc:                # noqa: BLE001
            out["reason"] = f"bundle build failed: {type(exc).__name__}: {exc}"
            self._log(f"stage {stage}: {out['reason']}", "error")
            # No parts to keep — a build that failed leaves nothing to re-send,
            # so this pending entry asks for a rebuild next tick.
            self._mark_stage_pending(stage, None)
            return out

        mb = built["bytes"] / 1048576.0
        self._log(f"stage {stage}: bundle {name} rev {rev} — "
                  f"{report['videos']} videos, {report['claims']} claims, "
                  f"{report['vectors']} vectors, "
                  f"{len(report['errors'])} errors, {mb:.1f} MB"
                  + (f" ({delta})" if delta else ""))

        parts = self._split_parts(path)
        entry = {"stage": stage, "seq": seq, "file": name, "parts": parts,
                 "captions": self._stage_captions(stage, report, parts, rev,
                                                  delta),
                 "message_ids": [], "bytes": built["bytes"],
                 "signature": sig, "revision": rev, "at": time.time(),
                 "videos": report["videos"], "claims": report["claims"],
                 "vectors": report["vectors"],
                 "counts": dict(report.get("counts") or {}),
                 "errors": len(report["errors"]),
                 "complete": report.get("complete", False)}

        if not (self._tg and self._tg.token):
            # A local-only session still gets the file, and still records that
            # it has not been published. The bundle stays on disk to be uploaded
            # by hand or by a later session that has credentials.
            out.update({"ok": False, "reason": "no Telegram credentials",
                        "path": path, "bytes": built["bytes"]})
            self._mark_stage_pending(stage, entry)
            return out

        msg_ids, failed = self._send_stage_parts(stage, entry)
        if failed or len(msg_ids) != len(parts):
            out["reason"] = failed or "upload incomplete"
            self._log(f"stage {stage}: bundle not uploaded — {out['reason']} "
                      f"({len(msg_ids)}/{len(parts)} parts up; the rest are "
                      f"held on disk for the next tick)", "warn")
            self._mark_stage_pending(stage, entry)
            return out

        self._finish_stage(stage, entry)
        out.update({"ok": True, "record": entry.get("record") or {}})
        return out

    def _stage_captions(self, stage: str, report: dict, parts: list,
                        rev: int, delta: str) -> list:
        """One caption per part, built once and kept with the bytes.

        Built here rather than at send time because a retry must re-send the same
        bytes *and* describe them the same way: a caption regenerated from a
        database that has moved on would describe a snapshot that does not match
        the file it is attached to, which is worse than no caption at all.
        """
        site = intake.site_id(self.store)
        digest = self._error_digest(report)
        head = (f"vios stage bundle · {stage} · {site}"
                + (f" · revision {rev}" if rev > 1 else "") + "\n"
                + f"{report['videos']} videos · {report['claims']} claims · "
                + f"{report['vectors']} vectors\n"
                + f"{report['counts'].get('done', 0)} done, "
                + f"{report['counts'].get('skipped', 0)} skipped, "
                + f"{report['counts'].get('failed', 0)} failed\n"
                + (f"{delta}\n" if delta else ""))
        out = []
        for i in range(len(parts)):
            tail = (f"worker {self.index + 1}/{self.partitions}"
                    + (f" · part {i + 1}/{len(parts)}" if len(parts) > 1
                       else ""))
            cap = head + tail + (f"\n{digest}" if digest else "")
            out.append(cap[:1020])
        return out

    def _send_stage_parts(self, stage: str, entry: dict) -> tuple:
        """Send the parts this entry has not sent yet. Returns (msg_ids, error).

        Resumes rather than restarts: `message_ids` records what is already in
        the channel, so a bundle that failed on part 3 of 4 sends one part on the
        retry instead of four, and does not leave two copies of parts 1 and 2 in
        the channel for a restore to trip over.
        """
        parts = list(entry.get("parts") or [])
        caps = list(entry.get("captions") or [])
        msg_ids = [int(m) for m in (entry.get("message_ids") or [])]
        failed = ""
        for i in range(len(msg_ids), len(parts)):
            if self._stopping():
                failed = "stopping"
                break
            part = parts[i]
            if not os.path.exists(part):
                failed = (f"part {i + 1}/{len(parts)} is no longer on disk — "
                          f"rebuilding next time")
                entry["parts"] = []
                break
            cap = caps[i] if i < len(caps) else f"vios stage bundle · {stage}"
            try:
                res = self._tg.send_document(
                    part, cap, file_name=os.path.basename(part))
                msg_ids.append(int(res.get("message_id") or 0))
            except UploadError as exc:
                failed = str(exc)[:200]
                break
            except Exception as exc:            # noqa: BLE001
                failed = f"{type(exc).__name__}: {str(exc)[:200]}"
                break
        entry["message_ids"] = msg_ids
        return msg_ids, failed

    def _finish_stage(self, stage: str, entry: dict) -> None:
        """Pin, record, clear the pending entry, delete the local parts."""
        store = self.store
        msg_ids = [int(m) for m in (entry.get("message_ids") or [])]
        # Pinned like `db_export`'s manifest so a restore finds the newest
        # bundle in one getChat rather than a history walk the Bot API cannot do.
        try:
            pinned = bool(self._tg.pin(msg_ids[-1])) if msg_ids else False
        except Exception:                       # noqa: BLE001
            pinned = False

        record = {k: entry.get(k) for k in (
            "stage", "seq", "file", "signature", "revision", "bytes", "videos",
            "claims", "vectors", "counts", "errors", "complete")}
        record.update({"parts": len(entry.get("parts") or []),
                       "message_ids": msg_ids, "pinned": pinned,
                       "at": time.time()})
        entry["record"] = record

        # Watermark last. Everything above is inert until these four writes.
        store.set_meta("stage_seq", str(int(entry.get("seq") or 0)))
        store.set_meta(f"stage_rev:{stage}", str(int(entry.get("revision") or 1)))
        store.set_meta(f"stage_done:{stage}", json.dumps(record))
        store.set_meta(f"stage_sig:{stage}", str(entry.get("signature") or ""))
        self._mark_stage_pending(stage, None, drop=True)
        store.checkpoint()

        # Only now are the local copies scratch: they are in the channel.
        self._discard_stage_parts(entry)
        self._log(f"stage {stage}: bundle published"
                  + (f" rev {entry.get('revision')}"
                     if int(entry.get("revision") or 1) > 1 else "")
                  + (f" → message {msg_ids[-1]}" if msg_ids else "")
                  + ("" if pinned else " (pin failed)"))

    def _discard_stage_parts(self, entry: dict) -> None:
        for part in (entry.get("parts") or []):
            try:
                os.remove(part)
            except OSError:
                pass
        entry["parts"] = []

    def _pending_stages(self) -> dict:
        """`stage_pending` as a dict, whatever shape it is on disk.

        It was a list of stage names before the built parts were worth keeping.
        A session that upgrades mid-run reads its own old value, so both shapes
        are accepted here and only the dict is ever written — a bare name simply
        means "rebuild", which is exactly what the old behaviour was.
        """
        try:
            cur = json.loads(self.store.get_meta("stage_pending", "{}") or "{}")
        except Exception:                       # noqa: BLE001
            return {}
        if isinstance(cur, dict):
            return {k: v for k, v in cur.items() if isinstance(v, dict)}
        if isinstance(cur, list):
            return {s: {"stage": s, "parts": []} for s in cur
                    if isinstance(s, str)}
        return {}

    def _mark_stage_pending(self, stage: str, entry: dict | None,
                            drop: bool = False) -> None:
        """Remember a bundle that did not make it up, with its bytes, or forget it.

        Kept in `meta` rather than in memory because the reason a bundle fails is
        usually that Telegram is refusing connections, and that outlives the
        process far more often than it outlives the session.

        `entry` carries the built parts so the retry re-sends them. The one case
        where that is the wrong trade is a disk with no room left: a held 135 MB
        bundle competing with the next cohort's working directories would fail
        the run to save a snapshot, so below the floor the bytes are dropped and
        the entry degrades to "rebuild later".
        """
        cur = self._pending_stages()
        if drop:
            cur.pop(stage, None)
        elif entry is None:
            cur[stage] = {"stage": stage, "parts": [], "at": time.time()}
        else:
            keep = dict(entry)
            keep.pop("record", None)
            if keep.get("parts"):
                free = intake.free_mb(self.shard_dir)
                if free and free < self.disk_floor_mb:
                    self._log(f"stage {stage}: {free} MB free is under the "
                              f"{self.disk_floor_mb} MB floor — discarding the "
                              f"held bundle and rebuilding on the retry", "warn")
                    self._discard_stage_parts(keep)
                    keep["message_ids"] = []
            cur[stage] = keep
        try:
            self.store.set_meta("stage_pending", json.dumps(cur, default=str))
        except Exception:                       # noqa: BLE001
            pass

    def _retry_pending_stages(self, planned: list) -> None:
        """Re-attempt every bundle that failed to upload earlier.

        Called on the ordinary publish tick, which is the point: a group never
        waits on the one before it, and a transport outage costs a retry rather
        than a bundle.

        Two behaviours, decided by whether the bytes survived. A pending entry
        that still has its parts re-sends **those parts** — no `VACUUM INTO`, no
        gzip, no split, and the same captions, so the file in the channel matches
        what its caption says. An entry with no parts (build failed, or the disk
        floor took them) rebuilds. The old code passed `force=True` here and
        rebuilt every time, which is how one transport failure became a
        multi-hundred-megabyte re-upload after every cohort.

        The pending label may name a wave rather than a stage, because that is
        the boundary the rotation cuts on — `group_members` resolves either, and
        it takes the same spine override the rotation planned with so a wave
        label resolves to the components the bundle actually described.
        """
        pending = self._pending_stages()
        if not pending:
            return
        spine = registry.close_needs(self.spine_override, planned) \
            if self.spine_override else None
        for stage, entry in list(pending.items()):
            if self._stopping():
                return
            parts = [p for p in (entry.get("parts") or []) if os.path.exists(p)]
            if parts and len(parts) == len(entry.get("parts") or []):
                if not (self._tg and self._tg.token):
                    continue
                sent = len(entry.get("message_ids") or [])
                self._log(f"stage {stage}: re-sending the bundle already built "
                          f"({len(parts) - sent} of {len(parts)} parts left)")
                msg_ids, failed = self._send_stage_parts(stage, entry)
                if failed or len(msg_ids) != len(parts):
                    self._log(f"stage {stage}: still not uploaded — "
                              f"{failed or 'upload incomplete'}", "warn")
                    self._mark_stage_pending(stage, entry)
                else:
                    self._finish_stage(stage, entry)
                continue

            comps = registry.group_members(stage, planned, spine)
            if not comps:
                self._mark_stage_pending(stage, None, drop=True)
                continue
            self._log(f"stage {stage}: rebuilding the bundle — the built copy "
                      f"is gone")
            self._mark_stage_pending(stage, None, drop=True)
            self._publish_stage(stage, comps, force=True)

    def publish_stage_now(self, stage: str = "") -> dict:
        """Manual cut, for the tab and for "I am about to close this session"."""
        planned = list(self.selected)
        names = ([stage] if stage else registry.stages_of(planned))
        out = []
        for s in names:
            comps = registry.components_in_stage(s, planned)
            if comps:
                out.append(self._publish_stage(s, comps, force=True))
        return {"ok": any(r.get("ok") for r in out), "stages": out}

    def stage_status(self) -> list:
        """Per-stage standing plus its last bundle, for the tab."""
        planned = list(self.selected)
        rows = []
        try:
            store, cov = self.store, self.coverage
        except Exception:                       # noqa: BLE001
            return rows
        pending = self._pending_stages()
        for s in registry.stages_of(planned):
            comps = registry.components_in_stage(s, planned)
            try:
                rep = cov.stage_report(s, comps)
            except Exception as exc:            # noqa: BLE001
                rows.append({"stage": s, "components": comps,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            try:
                last = json.loads(store.get_meta(f"stage_done:{s}", "") or "{}")
            except Exception:                   # noqa: BLE001
                last = {}
            rep["bundle"] = last or None
            rep["bundle_pending"] = s in pending
            # A pending entry that still holds its parts will re-send those bytes
            # rather than rebuild, and that is worth showing: it is the difference
            # between "waiting on Telegram" and "about to vacuum the database
            # again".
            held = pending.get(s) or {}
            rep["bundle_held_parts"] = len(held.get("parts") or [])
            # The error list travels in the bundle in full; the tab gets the
            # most recent slice so the panel stays readable on a bad day.
            rep["error_count"] = len(rep.get("errors", []))
            rep["errors"] = rep.get("errors", [])[-25:]
            rows.append(rep)
        return rows

    # ══════════════════════════════════════════════════════════════════════
    # Transport
    # ══════════════════════════════════════════════════════════════════════

    def _open_channel(self) -> None:
        if not (self._tg and self._tg.token):
            self._log("No Telegram credentials — running locally only", "warn")
            return
        self._channel = intake.Channel(self._tg, self._log)
        if not self._channel.start():
            self._log(f"MTProto unavailable: {self._channel.reason} — falling "
                      f"back to the Bot API, which cannot fetch capture "
                      f"records or files over 20 MB", "warn")

    def _teardown(self) -> None:
        self._unload("shutting down")
        if self._channel is not None:
            self._channel.stop()
            self._channel = None
        try:
            if self._store is not None:
                self._store.checkpoint()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # What the tab reads
    # ══════════════════════════════════════════════════════════════════════

    def status(self, fresh: bool = False) -> dict:
        """One call, everything the tab draws.

        Cached for two seconds. The interface polls once a second, the matrix
        is a `GROUP BY` over a table with thirty rows per video, and the worker
        is writing to the same connection — recomputing it on every poll would
        put a read in front of the writer for no visible gain.
        """
        now = time.time()
        age, cached = self._status_cache
        if not fresh and cached and now - age < 2.0:
            live = dict(cached)
            live.update(self._live())
            return live

        pending = 0
        stages: list = []
        try:
            stats = self.store.stats()
            matrix = self.coverage.matrix()
            counts = self.coverage.counts()
            failures = self.coverage.failures(12)
            running = self.coverage.running()
            retry = self.coverage.retry_state()
            shards = self.store.shards()
            # Completed passes that answered without one of their soft inputs,
            # and how many of those are already fixable. Nothing else on the panel
            # can say it: those rows are `done`, so they count as done in the
            # matrix, in every cohort total and in every stage's "complete" — and
            # for 28 of this archive's 30 videos that word covered a reading taken
            # with no on-screen text at all.
            thin = self.coverage.thin()
            # Per-stage standing, including each stage's last channel bundle.
            # Cheap enough for the two-second cache: it is the same GROUP BY the
            # matrix already runs, sliced four ways rather than thirty-four.
            stages = self.stage_status()
            # Unpublished rows across every watermarked table, not just claims.
            # An embedding cohort writes no claims at all, and reporting "0
            # pending" while a gigabyte of per-frame vectors sits unexported is
            # exactly the kind of quiet lie this panel exists to prevent.
            pending = 0
            for table, mark in (
                    (self.store.max_claim_id, "shard_lo_id"),
                    (self.store.max_vector_id, "shard_lo_vec"),
                    (self.store.max_frame_vector_id, "shard_lo_fvec"),
                    (self.store.max_frame_metric_id, "shard_lo_fmet")):
                pending += max(
                    table() - int(self.store.get_meta(mark, "0") or 0), 0)
        except Exception as exc:
            stats, matrix, counts, failures, running, shards = (
                {}, [], {}, [], [], [])
            retry = {}
            thin = {"rows": 0, "per_component": {}, "waiting": 0}
            self._log(f"status: {type(exc).__name__}: {exc}", "warn")

        by_id = {m["component"]: m for m in matrix}
        cohorts = []
        for c in self.cohorts:
            done = sum(by_id.get(i, {}).get("done", 0) for i in c["components"])
            total = sum(by_id.get(i, {}).get("total", 0)
                        for i in c["components"])
            cohorts.append({**c, "done": done, "total": total,
                            "state": ("running" if c["index"] == self.cohort_index
                                      else "done" if total and done >= total
                                      else "pending")})

        out = {
            "state": self.state,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "worker": self.worker,
            "slice": {"index": self.index, "partitions": self.partitions},
            "resources": self.resources,
            "resources_line": (resources.describe(self.resources)
                               if self.resources else ""),
            "cohorts": cohorts,
            "cohort_index": self.cohort_index,
            "matrix": matrix,
            "counts": counts,
            "failures": failures,
            "running": running,
            "retry": retry,
            "store": stats,
            "shards": {"count": len(shards),
                       "last": shards[-1] if shards else None,
                       "published": self.session["shards"],
                       "last_at": self.last_publish,
                       # Written, never published, file still on disk. Distinct
                       # from `pending` above, which is rows not yet exported:
                       # this is bytes that exist only inside a Kaggle session
                       # and will not survive it. Any number here other than zero
                       # is evidence about to be lost.
                       "held": sum(1 for s in shards
                                   if not s.get("msg_id") and s.get("path")),
                       # …and how much disk that is. The count on its own says a
                       # session will lose *something*; the bytes say whether it
                       # is a rounding error or the run. Read off the recorded
                       # column rather than the filesystem, because this endpoint
                       # is polled and must not stat once per shard to answer.
                       "held_mb": round(sum(s.get("bytes") or 0 for s in shards
                                            if not s.get("msg_id")
                                            and s.get("path")) / 1048576.0, 1),
                       "pending": pending},
            "stages": stages,
            # What the last reconcile spared. Sits next to `shards` because the
            # two are one story: the channel gave the evidence back, and this is
            # how much of the work table that evidence already answered.
            "reconciled": dict(self._reconciled),
            # …and what the archive is still short of. `rows` is how many completed
            # passes hold a thinner answer than they could; `waiting` is how many of
            # those the next rotation will hand back, because the input they went
            # without has since written something. The difference is the honest one:
            # the rest are waiting on evidence that does not exist yet.
            "thin": thin,
            "telegram": {
                "configured": bool(self._tg and self._tg.token),
                "mtproto": bool(self._channel and self._channel.ready),
                "mtproto_reason": (self._channel.reason if self._channel
                                   else ""),
                "channel": (self._tg.channel if self._tg else None)},
            "source": {"downloaded": self._source.downloaded if self._source else 0,
                       "reused": self._source.reused if self._source else 0,
                       "mb": round((self._source.bytes if self._source else 0)
                                   / 1048576.0, 1)},
            "disk_free_mb": intake.free_mb(self.cache_dir),
        }
        out.update(self._live())
        self._status_cache = (now, out)
        return out

    def _live(self) -> dict:
        """The parts that must never be two seconds stale."""
        cur = dict(self.current) if self.current else {}
        if cur.get("since"):
            cur["elapsed"] = round(time.time() - cur["since"], 1)
        lanes = list(self._lanes)
        rows = [l.as_dict() for l in lanes]
        with self._lock:
            self.workers = {r["name"]: r for r in rows}
        # Every lane's resident set, unioned, plus the sweep's own. Reporting only
        # `self._cache.loaded()` here was true when there was one cache; with a
        # cache per card it would report an empty resident set for a run holding
        # twelve gigabytes of weights, which is exactly the reading that made
        # "card 1 is idle" invisible for 44 minutes.
        resident: list = []
        for cache in [l.cache for l in lanes] + [self._cache]:
            if cache is None:
                continue
            try:
                for name in cache.loaded():
                    if name not in resident:
                        resident.append(name)
            except Exception:                          # noqa: BLE001
                pass
        footprints: dict = {}
        for cache in [l.cache for l in lanes] + [self._cache]:
            if cache is None:
                continue
            try:
                footprints.update(cache.footprints())
            except Exception:                          # noqa: BLE001
                pass
        return {
            "state": self.state,
            "message": self.message,
            "current": cur,
            "session": dict(self.session),
            "loaded": resident,
            # What the GPU is actually doing, next to what the packer assumed
            # it would be doing. A leak is visible here as a resident set whose
            # footprints do not add up to the allocated total.
            "models": {
                "resident": resident,
                "footprints_mb": footprints,
                "vram": ModelCache.vram(),
                "failures": self._cache.recent_failures(10),
            },
            # One row per lane, and the queue depths behind them. This panel is
            # how "is everything busy?" stops being a guess: four rows on four
            # different video keys with non-zero queue depths is the run working,
            # and a dead-letter count above zero is always a real bug.
            "workers": rows,
            "lanes": {
                "count": len(rows),
                "busy": sum(1 for r in rows if r["busy"]),
                "backend": self._jobs.kind,
                "error": self._lane_error,
                "inflight": len(self._inflight),
                "staged": len(self._staged),
                "queues": self._jobs.metrics() if self._jobs.ready() else {},
                "dead": self._jobs.dead() if self._jobs.ready() else 0,
            },
            "uptime": (round(time.time() - self.started_at, 1)
                       if self.started_at else 0),
            "cloud": self._cloud_status(),
        }

    def _preflight_cloud(self, runnable: list) -> None:
        """Ask NIM what this key can reach, once, before the sweep starts.

        Only when a cloud pass is actually planned — a run with no cloud
        component should not spend a request, and an operator with no key
        should see nothing about NIM at all. Doing it here rather than on the
        first video means a wrong `VIOS_NIM_MODEL` is one clear line at the top
        of the log, naming the models that do exist, instead of five thousand
        identical 404s.
        """
        try:
            if not any(registry.get(cid).family == "cloud"
                       for cid in runnable):
                return
            from .runners.cloud import client  # noqa: PLC0415
            nim = client()
            if not nim.configured():
                self._log("A cloud pass is planned but VIOS_NIM_API_KEY is "
                          "not set — it will be skipped with that reason",
                          "warn")
                return
            nim.preflight(self._log)
        except Exception as exc:                # noqa: BLE001
            self._log(f"NIM preflight raised {type(exc).__name__}: "
                      f"{str(exc)[:160]}", "warn")

    def _cloud_status(self) -> dict:
        """NIM budget and usage, for the tab.

        Imported here rather than at module scope so a web process with no
        `openai` package still renders the tab. Exhaustion must never be
        silent: calls made, deferrals and the last limit error are all on
        screen, because the alternative is an archive that quietly stops
        gaining its deepest reading and nobody notices for a day.
        """
        try:
            from .runners.cloud import client  # noqa: PLC0415
            return client().status()
        except Exception:
            return {"configured": False}

    def activity(self, limit: int = 120) -> list:
        items = list(self._activity)
        return items[-limit:][::-1]

    # ══════════════════════════════════════════════════════════════════════
    # Operator actions
    # ══════════════════════════════════════════════════════════════════════

    def requeue(self, component: str = "", state: str = "failed") -> dict:
        n = self.coverage.requeue(component, state)
        self._log(f"Requeued {n} rows"
                  + (f" for {component}" if component else ""))
        return {"ok": True, "requeued": n}

    def reset_component(self, component: str) -> dict:
        """Re-run a pass from scratch — the button for 'I changed the prompt'.

        The claims already written are *not* deleted. They carry the old
        observer id, the new run carries a new one, and both stay. That is the
        append-only rule doing its job: a database that forgets what it used to
        think cannot be audited.
        """
        n = self.coverage.reset_component(component)
        self._log(f"Reset {component}: {n} rows queued again. Existing claims "
                  f"are kept under their original observer.")
        return {"ok": True, "reset": n}

    def sync_now(self) -> dict:
        got = intake.sync(self.store, self.ledger_path)
        self._log(f"Sync: {got.get('added', 0)} new videos of "
                  f"{got.get('seen', 0)} uploaded"
                  + (f", {got['refused']} ledger row(s) refused — the key is "
                     f"not a video identity" if got.get("refused") else ""))
        return got

    # ── reconcile: what the channel already answered ─────────────────────
    def expected_observers(self, components=None) -> dict:
        """`{component: observer_id}` for the build that is about to run.

        Derived, never registered: `Store.observer` bumps a run counter, and
        asking "has this already been done" must not leave a fingerprint saying
        it was done here.

        The complete reading only. A pass that ran without a soft input writes
        under a different id by design — see `_observer_params` — and those ids
        are enumerated separately by `degraded_observers`, because this dict also
        answers "what should the next run produce", and the answer to that is
        always the complete one.
        """
        out = {}
        for cid in (components if components is not None else self.selected):
            comp = registry.BY_ID.get(cid)
            if comp is None:
                continue
            out[cid] = store_observer_id(
                cid, comp.model or comp.family, comp.revision, comp.params)
        return out

    def degraded_observers(self, components=None) -> dict:
        """`{observer_id: (component, [missing inputs])}` for every thin reading.

        The other half of `expected_observers`, and reconcile needs both. A pass
        that ran without one of its soft inputs registered its observer with that
        absence in the hashed params, so its claims carry an id that
        `expected_observers` does not name — and reconcile treats evidence under
        an unnamed id as a superseded revision and ignores it. After a restore
        that means the row stays `queued`, the pass runs again, and every claim it
        writes collides with one already there on its uid: full GPU cost, zero new
        rows, and `done(claims=0)` overwriting the count that was true.

        Enumerated from the registry rather than read out of the local coverage
        table, because the case that matters most is the one where there is no
        local coverage table to read: the Kaggle container starts empty, restores
        shards into the evidence tables, and asks this question with nothing else
        to go on. Every non-empty subset of a pass's soft inputs, which the
        registry bounds at four — 72 ids across the eight soft-consuming passes,
        one `IN` clause, once per rotation.

        `Coverage.degraded_variants()` supplements it for absences the current
        registry can no longer express: a soft edge that was removed after some
        run had already gone without it leaves claims under an id no subset of
        today's declaration can reproduce, and the coverage row is then the only
        record of it.
        """
        sel = list(components if components is not None else self.selected)
        out: dict = {}

        def add(cid, comp, names):
            names = sorted({n for n in names if n})
            if not names:
                return
            oid = store_observer_id(cid, comp.model or comp.family,
                                    comp.revision,
                                    _observer_params(comp.params, names))
            out.setdefault(oid, (cid, names))

        for cid in sel:
            comp = registry.BY_ID.get(cid)
            if comp is None or not comp.soft:
                continue
            soft = sorted(comp.soft)
            for k in range(1, len(soft) + 1):
                for combo in itertools.combinations(soft, k):
                    add(cid, comp, combo)
        try:
            for cid, without in self.coverage.degraded_variants():
                comp = registry.BY_ID.get(cid)
                if comp is not None and cid in set(sel):
                    add(cid, comp, str(without or "").split(","))
        except Exception:                          # noqa: BLE001
            # The enumeration above is the load-bearing half and it has already
            # run. A missing `degraded` column on an old database is not a reason
            # to refuse to reconcile.
            pass
        return out


    def reconcile_now(self, components=None) -> dict:
        """Mark done every selected pass whose evidence is already here.

        Runs after a restore and again at the top of every sweep. Without it a
        fresh container replays five thousand videos' worth of shards into the
        evidence tables, opens an empty coverage table beside them, and
        processes the entire archive a second time — which is the "it
        reprocesses everything" complaint exactly.

        Two different questions, answered two different ways, and the split is
        the whole design:

        **A pass that writes evidence is judged only on its own evidence**,
        keyed on `observer_id`. A revision bump is how a pass says "the old
        reading is superseded"; matching on the component alone would let the
        superseded reading satisfy the new one and silently cancel the upgrade
        the bump was for.

        **A pass that writes no evidence is inferred from its dependants.**
        `probe`, `artifacts`, `shots`, `keyframes` and `allframes` produce
        working files, not rows — a restored database holds no trace of them,
        so an evidence check would leave all five queued for every video in the
        archive and the sweep would download five thousand videos to rebuild
        frames nothing is waiting for. But if `ocr` has claims for a video from
        the current observer, then `allframes` demonstrably ran for that video:
        the dependency edge is the proof. Their outputs are rebuilt on demand
        by `_ensure_inputs` the moment a pass actually needs them, which is the
        same guarantee that already lets cohort 1 run after cohort 0's workdir
        was evicted.

        The inference is applied *only* to passes that write nothing. Closing
        the dependency graph over evidence-writing passes would be the exact
        inversion described above — `tag` at an unchanged revision would mark a
        freshly bumped `visual-embed` as done — so it is deliberately not done.
        """
        sel = list(components if components is not None else self.selected)
        if not sel:
            return {"rows": 0, "matched": 0, "per_component": {}, "videos": 0}

        store, cov = self.store, self.coverage
        expected = self.expected_observers(sel)
        writes_rows = {cid for cid in sel
                       if {"claims", "vectors", "metrics"}
                       & set(registry.BY_ID[cid].produces)}
        by_observer = {expected[cid]: cid for cid in writes_rows
                       if expected.get(cid)}
        # And the thin readings of the same passes, under the ids that record what
        # they went without. `degraded_observers` says why they cannot be left out:
        # unnamed here, their claims read as a superseded revision, the row stays
        # queued, and the pass re-runs to write uids that already exist.
        thin_of: dict = {}
        for oid, (cid, names) in self.degraded_observers(sel).items():
            if cid in writes_rows and oid not in by_observer:
                by_observer[oid] = cid
                thin_of[oid] = names


        evidence = store.evidence_by_observer(set(by_observer))
        if not evidence:
            # Recorded, not just returned. "Reconciled nothing" and "never
            # reconciled" look identical on a panel that only shows a count, and
            # they mean opposite things: the first says the archive is genuinely
            # new to this build, the second says the step did not happen and the
            # sweep is about to redo work. The timestamp is what separates them.
            out = {"rows": 0, "matched": 0, "per_component": {}, "videos": 0,
                   "message": "No restored evidence matches this build."}
            with self._lock:
                self._reconciled = {"at": time.time(), "error": "", **out}
            return out

        # Which structure pass each evidence-writing pass proves, transitively.
        silent = [cid for cid in sel if cid not in writes_rows]
        implied: dict = {}
        for cid in writes_rows:
            seen, stack = set(), list(registry.BY_ID[cid].needs)
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                stack.extend(registry.BY_ID[n].needs if n in registry.BY_ID
                             else ())
            implied[cid] = [n for n in seen if n in silent]

        rows = []
        for video_key, seen in evidence.items():
            proved = set()
            # Best reading per pass, not every reading. A video can hold both a
            # thin set of claims and a complete one — that is exactly what a
            # re-offer produces, and both stay, because the archive is append-only
            # — and the coverage row has to describe the *best* answer now held.
            # Without this collapse the row would take whichever id `evidence`
            # happened to yield last, so a video that was repaired could reconcile
            # back to thin and be re-offered forever.
            best: dict = {}
            for oid, counts in seen.items():
                cid = by_observer.get(oid)
                if not cid:
                    continue                      # a superseded revision
                names = thin_of.get(oid) or []
                got = (counts.get("claims", 0), counts.get("vectors", 0))
                rank = (len(names), -(got[0] + got[1]))
                if cid not in best or rank < best[cid][0]:
                    best[cid] = (rank, oid, got, names)
                proved.update(implied.get(cid, ()))
            for cid, (_, oid, got, names) in best.items():
                rows.append((video_key, cid, oid, got[0], got[1], 0.0, names))
            for cid in proved:
                # Zero seconds and zero counts: this row was inferred, not run,
                # and the coverage matrix should not claim a duration for work
                # that happened on another machine on another day.
                rows.append((video_key, cid, expected.get(cid, ""), 0, 0, 0.0, ()))


        out = cov.reconcile(rows)
        with self._lock:
            self._reconciled = {"at": time.time(), "rows": out.get("rows", 0),
                                "videos": out.get("videos", 0),
                                "matched": out.get("matched", 0),
                                "per_component": dict(
                                    out.get("per_component") or {}),
                                "error": ""}
        if out.get("rows"):
            top = sorted(out["per_component"].items(),
                         key=lambda kv: -kv[1])[:6]
            self._log(
                f"Reconciled {out['rows']} passes across {out['videos']} "
                f"videos from evidence already held — "
                + ", ".join(f"{c} {n}" for c, n in top)
                + (" …" if len(out["per_component"]) > 6 else ""))
        return out

    def adopt_folder_now(self, folder: str) -> dict:
        """Take every video in a folder into the work table.

        The other half of `sync_now`. A reel that is already on this disk needs
        no ledger row and no Telegram message to be worth processing, and the
        engine could previously not be pointed at one at all.
        """
        folder = os.path.expanduser((folder or "").strip().strip('"').strip("'"))
        if not folder:
            raise ValueError("Give the folder that holds the videos.")
        if not os.path.isdir(folder):
            raise ValueError(f"No folder at {folder}.")
        got = intake.adopt_folder(self.store, folder,
                                  ledger_path=self.ledger_path)
        with self._lock:
            if folder not in self.video_dirs:
                self.video_dirs.append(folder)
        self._cov = None                  # the partition counts just changed
        # What was merged is more interesting than what was added: it is the
        # answer to "did pointing this at my downloads folder just duplicate my
        # archive?", and the answer is now no, with the count to show it.
        self._log(f"Folder: {got.get('added', 0)} new videos of "
                  f"{got.get('seen', 0)} files in {folder}"
                  + (f", {got['merged']} already here under another filename"
                     if got.get("merged") else "")
                  + (f", {got['refused']} unreadable"
                     if got.get("refused") else ""))
        return got

    def video_detail(self, key: str) -> dict:
        store = self.store
        video = store.video(key)
        if not video:
            return {}
        return {"video": video, "shots": store.shots(key),
                "coverage": self.coverage.for_video(key),
                "artifacts": store.artifacts(key),
                "claims": store.claims(key)[:400]}


# ══════════════════════════════════════════════════════════════════════════
# The process-wide singleton
# ══════════════════════════════════════════════════════════════════════════

_engine: ProcessEngine | None = None
_engine_lock = threading.Lock()


def get_engine(base_dir: str | None = None) -> ProcessEngine:
    """One engine per process, created on first use.

    Guarded because Flask serves requests on threads: two simultaneous hits on
    the tab at startup would otherwise build two engines, two stores and two
    coverage tables over one file, and the second would quietly take the first
    one's leases.
    """
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ProcessEngine(base_dir)
        return _engine


# ══════════════════════════════════════════════════════════════════════════
# Publishing on the way out
# ══════════════════════════════════════════════════════════════════════════
#
# A Kaggle session ends by having its container deleted. Everything on scratch
# disk goes with it, so an exported shard that never reached the channel is a
# shard that never existed — and the window between the last publish and the end
# is up to five minutes of work on four workers. Closing that window is the
# cheapest correctness win in the whole publish path.
#
# Two mechanisms, because neither covers the other:
#
#   atexit    fires on a normal return and on SIGINT — KeyboardInterrupt unwinds
#             the stack normally, so the handler runs. It does *not* fire when
#             the process is signalled dead.
#   SIGTERM   is what a container stop sends, and its default disposition kills
#             the process outright without running atexit. It needs a handler,
#             and the handler must chain to whatever was installed before it so
#             installing this does not make the process unkillable.
#
# `signal.signal` can only be called from the main thread, and this module is
# imported from a Flask worker as often as not — hence the `ValueError` arm and
# the rule that `install_exit_flush()` is called from `__main__`.

_exit_flush_installed = False


def _flush_engine(why: str) -> None:
    """Publish the running engine's tail, if there is one. Never raises."""
    eng = _engine
    if eng is None:
        return
    try:
        res = eng.flush_for_exit(why)
        if res.get("shard"):
            print(f"⚙️  [PROCESS] Exit flush published shard {res['shard']}",
                  flush=True)
    except Exception as exc:                    # noqa: BLE001
        print(f"⚙️  [PROCESS] Exit flush failed: {type(exc).__name__}: "
              f"{str(exc)[:200]}", flush=True)


def install_exit_flush() -> dict:
    """Arm the exit paths. Idempotent, and safe to call off the main thread.

    Returns what it managed to install, so the caller can log the truth rather
    than an assumption — a process that could not take the signals still has
    `atexit`, and that is worth knowing from the log rather than from a lost
    shard.
    """
    global _exit_flush_installed
    if _exit_flush_installed:
        return {"atexit": True, "signals": [], "already": True}
    _exit_flush_installed = True

    import atexit
    import signal

    atexit.register(_flush_engine, "atexit")

    took = []
    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            previous = signal.getsignal(sig)
        except Exception:                       # noqa: BLE001
            previous = None

        def handler(signum, frame, _prev=previous, _name=name):
            _flush_engine(_name.lower())
            # Chain, so the process still dies. A handler that swallows SIGTERM
            # turns a container stop into a hang and then a hard kill, which
            # loses strictly more than doing nothing would have.
            if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                _prev(signum, frame)
            elif _prev == signal.SIG_IGN:
                return
            else:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        try:
            signal.signal(sig, handler)
            took.append(name)
        except (ValueError, OSError, RuntimeError):
            # Off the main thread, or a platform without it. atexit still stands.
            pass
    return {"atexit": True, "signals": took, "already": False}
