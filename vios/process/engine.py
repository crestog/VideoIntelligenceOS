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

import json
import os
import re
import threading
import time
import traceback
from collections import deque

from . import intake, media, registry, resources
from .. import creds as _creds
from .coverage import Coverage, worker_id
from .runners import get as get_runner
from .runners import missing as missing_runners
from .runners.base import DeferPass, Job, ModelCache, SkipPass
from .store import Store
from .store import observer_id as store_observer_id

from ..capture.upload import Telegram, UploadError

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

# Extra VRAM to leave unclaimed, *on top of* the 1 GB `resources.probe` already
# holds back. The allocator fragments over a twelve-hour session in a way a
# single measurement at startup cannot see, and the reel that OOMs is always the
# one with forty shots, three hours in.
VRAM_HEADROOM_MB = 384

# Working-directory cache. Proxy plus frames plus wav is 8–20 MB per reel, so
# 12 GB holds roughly a thousand — a whole cohort's worth for most partitions.
CACHE_BUDGET_MB = 12_000
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


_ENV_UNKNOWN: dict = {}


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
        self.current: dict = {}
        self.session = {"videos": 0, "passes": 0, "claims": 0, "vectors": 0,
                        "skipped": 0, "failed": 0, "deferred": 0,
                        "elsewhere": 0,
                        "seconds": 0.0, "downloaded": 0, "shards": 0}
        self.last_publish: float | None = None
        self.since_publish = 0

        # Components deferred because this machine cannot run them, so their
        # dependents can be told the same thing instead of being retried every
        # five minutes, and so the log says it once rather than once per video.
        self._elsewhere: set = set()
        self._elsewhere_logged: set = set()

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
        """
        with self._lock:
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
        adopted = 0
        for d in self.video_dirs:
            if os.path.isdir(d):
                try:
                    adopted += intake.adopt_folder(self.store, d).get("added", 0)
                except Exception as exc:
                    self._log(f"folder {d}: {type(exc).__name__}: {exc}", "warn")
        have = len(self.store.video_keys())
        ok("Videos to process", have > 0,
           f"{have} in the evidence store, {synced.get('seen', 0)} uploaded in "
           f"the capture ledger"
           + (f", {adopted} adopted from disk" if adopted else "") if have else
           f"nothing yet — capture some reels first, point this at an existing "
           f"ledger ({self.ledger_path}), or give it a folder of videos",
           block=True)

        # ── hardware ─────────────────────────────────────────────────────
        res = resources.probe(self.cache_dir)
        self.resources = res
        ok("Hardware", True, resources.describe(res))
        ok("Scratch disk", res["disk_free_mb"] > self.disk_floor_mb,
           f"{res['disk_free_mb'] / 1024:.1f} GB free on {self.cache_dir}")

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
                            "vectors": 0, "skipped": 0, "failed": 0,
                            "deferred": 0, "elsewhere": 0, "seconds": 0.0,
                            "downloaded": 0, "shards": 0}
            self._elsewhere_logged = set()
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
            self._drain_publisher("sweep-end")
            self._teardown()
            with self._lock:
                if self.state not in (ERROR,):
                    self.state = IDLE
                    self.message = (self.message if self._stop.is_set()
                                    else self.message) or "Stopped."
                self.current = {}
                self.cohort_index = -1

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
                should_stop=self._stopping)
            self._shards_at = time.time()
            self._log(f"Shards: {got.get('imported', 0)} imported, "
                      f"{got.get('skipped', 0)} already held, "
                      f"{got.get('claims', 0)} claims recovered"
                      + (f" — {got['reason']}" if got.get("reason") else ""))
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
        """Run every pass in this cohort over every video that still needs one.

        `seen` is why a failure does not become a spin: a video whose pass
        failed is still claimable — that is the point of retries — and without
        remembering that it was attempted this cohort, `candidates` would hand
        it straight back and the loop would grind on the same reel until its
        attempts ran out. Retries belong to the next rotation, not this one.

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

        seen: set = set()
        worked = 0
        while not self._stopping():
            self._wait_if_paused()
            keys = [k for k in self.coverage.candidates(ids, limit=64)
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
                if self._process_video(key, ids, cohort):
                    worked += 1
        return worked

    # ── one video, every resident pass ───────────────────────────────────
    def _process_video(self, key: str, ids: list, cohort) -> bool:
        cov, store = self.coverage, self.store
        video = store.video(key)
        if video is None:
            return False

        mine = cov.claim_for(key, ids)
        if not mine:
            return False                      # another worker got there first
        order = registry.topo_sort(mine)
        workdir = os.path.join(self.cache_dir, _safe(key))
        started = time.time()

        with self._lock:
            self.current = {"video_key": key, "cohort": cohort.index,
                            "component": "", "title": "fetching",
                            "since": started, "passes": len(order)}

        try:
            source = self._source.ensure(video, workdir)
        except intake.SourceError as exc:
            for cid in order:
                cov.fail(key, cid, str(exc))
            self._log(f"{key}: {exc}", "warn")
            self.session["failed"] += len(order)
            return False
        except Exception as exc:
            for cid in order:
                cov.fail(key, cid, f"{type(exc).__name__}: {exc}")
            self._log(f"{key}: fetch failed — {type(exc).__name__}: {exc}",
                      "error")
            self.session["failed"] += len(order)
            return False
        intake.touch(workdir)

        states = {r["component"]: r["state"] for r in cov.for_video(key)}
        for cid in order:
            if self._stopping():
                cov.release(key, cid)
                continue
            self._wait_if_paused()
            states[cid] = self._run_pass(key, cid, video, source, workdir,
                                         states, cohort)

        self.session["videos"] += 1
        self.session["seconds"] += time.time() - started
        self.since_publish += 1
        # Not `if due: publish` any more. The whole point of the publisher thread
        # is that the decision and the 45 MB POST happen somewhere this loop is
        # not — so this says "there are rows" and goes back to the next video.
        # The publisher applies the same `_publish_due` gate on its own tick.
        self._rows_written(f"cohort-{cohort.index}")

        freed = intake.evict(self.cache_dir, self.cache_budget_mb,
                             keep={_safe(key)}, floor_mb=self.disk_floor_mb)
        if freed["removed"]:
            self._log(f"Cache: removed {freed['removed']} working directories, "
                      f"{freed['freed_mb']} MB")
        with self._lock:
            self.current = {}
        return True

    # ── one pass ─────────────────────────────────────────────────────────
    def _run_pass(self, key: str, cid: str, video: dict, source: str,
                  workdir: str, states: dict, cohort) -> str:
        cov, store = self.coverage, self.store
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
            self.session["elsewhere"] = self.session.get("elsewhere", 0) + 1
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
            why = (f"depends on {', '.join(broken)}, which produced nothing "
                   f"for this video")
            cov.skip(key, cid, why)
            self.session["skipped"] += 1
            return "skipped"

        degraded = [n for n in comp.soft
                    if states.get(n) in ("failed", "skipped")]
        if degraded:
            # Recorded, not vetoed: the claim this pass writes carries a note
            # saying which inputs were missing, so a thinner answer is legible
            # as thinner rather than passed off as complete.
            self._log(f"{key} {cid}: running without {', '.join(degraded)} "
                      f"— soft input, the pass reports what it had", "warn")

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
            self.session["deferred"] = self.session.get("deferred", 0) + 1
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
            self.session["deferred"] = self.session.get("deferred", 0) + 1
            return "deferred"

        with self._lock:
            self.current.update({"component": cid, "title": comp.title,
                                 "component_since": time.time(), "note": "",
                                 "detail": "", "detail_at": 0.0,
                                 "frames_done": 0, "frames_total": 0})
        # So a load or unload fault names the pass that provoked it rather than
        # whichever pass happened to be running when the log was read.
        self._cache.context = cid

        params = dict(comp.params)
        if self._hf_token:
            params.setdefault("hf_token", self._hf_token)
        job = Job(
            video=dict(video), component=comp, store=store, source=source,
            workdir=workdir, params=params, resources=self.resources,
            cache=self._cache,
            renew=lambda progress="", k=key, c=cid: cov.renew(k, c, progress),
            progress=self._progress,
            log=self._note)

        t0 = time.time()
        try:
            self._ensure_inputs(job, comp)
            fn = get_runner(cid)
            em = fn(job)
        except SkipPass as exc:
            cov.skip(key, cid, str(exc))
            self.session["skipped"] += 1
            self._log(f"{key} · {cid}: skipped — {exc}")
            return "skipped"
        except DeferPass as exc:
            # Not a failure and not a skip: the work is runnable, the moment
            # is wrong. No attempt is spent, so an archive cannot exhaust its
            # retries against a rate limit that clears on its own.
            wait = max(30.0, min(float(getattr(exc, "retry_after", 300.0)),
                                 3600.0))
            cov.defer(key, cid, str(exc), wait)
            self.session["deferred"] = self.session.get("deferred", 0) + 1
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
            blame = self._loading_blame(exc, t0)
            reason = f"{type(exc).__name__}: {exc}"
            if blame:
                reason = f"{reason} [{blame}]"
            state = cov.fail(key, cid, reason)
            self.session["failed"] += 1
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
                             f"{', '.join(self._cache.loaded()) or 'nothing'}"
                             if v else ""), "error")
                self._unload("out of VRAM — dropping every resident model")
            return state

        # ── accept the emission ──────────────────────────────────────────
        try:
            observer = store.observer(
                cid, comp.model or comp.family, comp.revision, comp.params,
                comp.device)
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
            for a in em.artifacts:
                size = 0
                try:
                    size = (os.path.getsize(a["path"])
                            if os.path.isfile(a["path"]) else 0)
                except OSError:
                    pass
                store.set_artifact(key, a["name"], None, "", size, a["meta"])
        except Exception as exc:
            state = cov.fail(key, cid, f"store rejected the emission: "
                                       f"{type(exc).__name__}: {exc}")
            self.session["failed"] += 1
            self._log(f"{key} · {cid}: store rejected the emission — "
                      f"{type(exc).__name__}: {exc}", "error")
            return state

        seconds = time.time() - t0
        cov.done(key, cid, seconds, n_claims, n_vectors, observer)
        self.session["passes"] += 1
        self.session["claims"] += n_claims
        self.session["vectors"] += n_vectors
        return "done"

    def _note(self, message: str) -> None:
        with self._lock:
            if self.current:
                self.current["note"] = str(message)[:200]
        self._log(message)

    def _progress(self, detail: str) -> None:
        """A pass's own report of where it is inside itself.

        Deliberately not logged. `job.heartbeat` fires once per batch, so a
        900-frame OCR pass produces about thirty of these per video and several
        thousand per sweep; putting them in the activity ring would push out
        every error the ring exists to keep. They go to `current` only, which is
        what the live panel reads.

        `frame 320/900` is parsed into counters here rather than in the browser
        so the shape stays in one place: the runners already emit that string,
        and a progress bar in the interface should not depend on a regex in
        JavaScript agreeing with a format string in Python.
        """
        detail = str(detail)[:200]
        with self._lock:
            if not self.current:
                return
            self.current["detail"] = detail
            done = total = 0
            hit = _PROGRESS_RE.search(detail)
            if hit:
                done, total = int(hit.group(1)), int(hit.group(2))
            self.current["frames_done"] = done
            self.current["frames_total"] = total
            self.current["detail_at"] = time.time()

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        name = type(exc).__name__
        return ("OutOfMemory" in name
                or "CUDA out of memory" in str(exc)
                or "CUBLAS_STATUS_ALLOC_FAILED" in str(exc))

    def _loading_blame(self, exc: Exception, since: float) -> str:
        """"loading <key>", when this failure came out of a model load.

        `ModelCache` records every load fault as it happens, with the key it
        was loading. Matching on the exception text rather than re-raising a
        wrapper keeps the original traceback intact — which is what someone
        reading the Kaggle log actually needs — while still putting the model
        name in the coverage row, where it is queryable.
        """
        try:
            msg = str(exc)[:300]
            for f in reversed(self._cache.recent_failures(8)):
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

    # ══════════════════════════════════════════════════════════════════════
    # Unloading and publishing
    # ══════════════════════════════════════════════════════════════════════

    def _unload(self, why: str = "") -> None:
        """Drop every resident model and say what actually came back.

        The reclaimed total is logged against the cohort that held the models,
        because that is the only moment the number is attributable. A leak
        found later — as an OOM in the next cohort — names the wrong pass.
        """
        loaded = self._cache.loaded()
        expected = sum(self._cache.footprints().get(k, 0) for k in loaded)
        before = ModelCache.vram()
        freed = self._cache.unload_all()
        after = ModelCache.vram()
        if not loaded:
            return

        line = (f"Unloaded {len(loaded)} models — {freed} MB reclaimed"
                + (f" of {expected} MB held" if expected else "")
                + (f" — {why}" if why else ""))
        self._log(line)
        if after:
            self._log(f"VRAM now {after.get('allocated', 0)} MB allocated, "
                      f"{after.get('free', 0)} MB free of "
                      f"{after.get('total', 0)}; resident: "
                      f"{', '.join(self._cache.loaded()) or 'nothing'}")
        # Still resident after an unload_all means a drop raised and the
        # reference survived. Loud, because the packer's next plan is wrong.
        if self._cache.loaded():
            self._log(f"Models still resident after unload: "
                      f"{', '.join(self._cache.loaded())} — the next cohort "
                      f"is planned against VRAM that is not free", "error")
        elif expected >= 512 and freed < expected // 2 and before:
            self._log(f"Only {freed} MB of {expected} MB came back after "
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
        with its watermark unmoved, so the next attempt includes it rather than
        losing it.

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
                                           lo_fmet, hi_fmet)
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
            # Only advance the watermark once the rows are safely in a file.
            # If the upload failed the shard is still on local disk and still
            # counted, which is the honest state: exported, not yet published.
            store.set_meta("shard_lo_id", str(stats["hi_id"]))
            store.set_meta("shard_lo_vec", str(stats["hi_vec"]))
            store.set_meta("shard_lo_fvec", str(stats["hi_fvec"]))
            store.set_meta("shard_lo_fmet", str(stats["hi_fmet"]))
            store.set_meta("shard_seq", str(seq))
            store.checkpoint()

            with self._lock:
                self.since_publish = 0
                self.last_publish = time.time()
                self.session["shards"] += 1
            self._log(f"Shard {sid}: {stats['claims']} claims, "
                      f"{stats['vectors']} vectors, "
                      f"{stats['frame_vectors']} frame-vector rows, "
                      f"{stats['frame_metrics']} frame-metric rows, "
                      f"{stats.get('coverage', 0)} coverage rows, "
                      f"{stats['bytes'] / 1024:.0f} KB"
                      + (f" → message {msg_id}" if msg_id else
                         " (held locally — no upload)"))
            if msg_id is None and self._tg and self._tg.token:
                return ""
            return sid

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
                       "pending": pending},
            "stages": stages,
            # What the last reconcile spared. Sits next to `shards` because the
            # two are one story: the channel gave the evidence back, and this is
            # how much of the work table that evidence already answered.
            "reconciled": dict(self._reconciled),
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
        return {
            "state": self.state,
            "message": self.message,
            "current": cur,
            "session": dict(self.session),
            "loaded": self._cache.loaded(),
            # What the GPU is actually doing, next to what the packer assumed
            # it would be doing. A leak is visible here as a resident set whose
            # footprints do not add up to the allocated total.
            "models": {
                "resident": self._cache.loaded(),
                "footprints_mb": self._cache.footprints(),
                "vram": ModelCache.vram(),
                "failures": self._cache.recent_failures(10),
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
                  f"{got.get('seen', 0)} uploaded")
        return got

    # ── reconcile: what the channel already answered ─────────────────────
    def expected_observers(self, components=None) -> dict:
        """`{component: observer_id}` for the build that is about to run.

        Derived, never registered: `Store.observer` bumps a run counter, and
        asking "has this already been done" must not leave a fingerprint saying
        it was done here.
        """
        out = {}
        for cid in (components if components is not None else self.selected):
            comp = registry.BY_ID.get(cid)
            if comp is None:
                continue
            out[cid] = store_observer_id(
                cid, comp.model or comp.family, comp.revision, comp.params)
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
            for oid, counts in seen.items():
                cid = by_observer.get(oid)
                if not cid:
                    continue                      # a superseded revision
                rows.append((video_key, cid, oid,
                             counts.get("claims", 0), counts.get("vectors", 0),
                             0.0))
                proved.update(implied.get(cid, ()))
            for cid in proved:
                # Zero seconds and zero counts: this row was inferred, not run,
                # and the coverage matrix should not claim a duration for work
                # that happened on another machine on another day.
                rows.append((video_key, cid, expected.get(cid, ""), 0, 0, 0.0))

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
        got = intake.adopt_folder(self.store, folder)
        with self._lock:
            if folder not in self.video_dirs:
                self.video_dirs.append(folder)
        self._cov = None                  # the partition counts just changed
        self._log(f"Folder: {got.get('added', 0)} new videos of "
                  f"{got.get('seen', 0)} files in {folder}")
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
