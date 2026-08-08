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

import os
import threading
import time
import traceback
from collections import deque

from . import intake, media, registry, resources
from .coverage import Coverage, worker_id
from .runners import get as get_runner
from .runners import missing as missing_runners
from .runners.base import Job, ModelCache, SkipPass
from .store import Store

from ..capture.upload import Telegram, UploadError

IDLE, RUNNING, PAUSED, STOPPING, ERROR = (
    "idle", "running", "paused", "stopping", "error")

# Videos between shard uploads. Fifty is about four minutes of work on a T4 and
# a few hundred kilobytes gzipped — small enough that a killed session loses
# almost nothing, large enough that the channel does not fill with noise.
PUBLISH_EVERY = 50

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
                        "skipped": 0, "failed": 0, "seconds": 0.0,
                        "downloaded": 0, "shards": 0}
        self.last_publish: float | None = None
        self.since_publish = 0

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.RLock()

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
        except Exception:
            self.cred_sources = {}

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
        self._activity.append({"at": time.time(), "level": level,
                               "text": str(text)[:400]})

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
                # alone would not be enough for pyannote.
                os.environ["HF_TOKEN"] = self._hf_token
                os.environ.setdefault("HUGGINGFACE_TOKEN", self._hf_token)

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
                self.publish_every = max(5, int(publish_every))
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
                            "seconds": 0.0, "downloaded": 0, "shards": 0}
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="vios-process")
            self._thread.start()
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
        self._teardown()
        with self._lock:
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
        if self.restore_on_start and self._channel and self._channel.ready:
            self.message = "Replaying evidence shards from the channel."
            self._log("Scanning the channel for evidence shards")
            got = intake.restore_shards(
                store, self._tg, self._channel,
                on_progress=lambda seen, head, n: setattr(
                    self, "message",
                    f"Replaying shards — {seen}/{head} messages, {n} imported"),
                should_stop=self._stopping)
            self._log(f"Shards: {got.get('imported', 0)} imported, "
                      f"{got.get('skipped', 0)} already held, "
                      f"{got.get('claims', 0)} claims recovered"
                      + (f" — {got['reason']}" if got.get("reason") else ""))
            for e in got.get("errors", [])[:5]:
                self._log(f"shard: {e}", "warn")

        self._source = intake.Source(self._tg, self._channel, self._log,
                                     local_dirs=self.video_dirs)

        while not self._stopping():
            res = resources.probe(self.cache_dir)
            with self._lock:
                self.resources = res
            self._log(resources.describe(res))

            sel = list(self.selected)
            cannot = registry.unrunnable(sel, res)
            for cid, why in cannot.items():
                self._log(f"{cid} cannot run here: {why}", "warn")
            runnable = [c for c in sel if c not in cannot]
            if not runnable:
                raise RuntimeError(
                    "no selected pass can run on this machine — "
                    + "; ".join(f"{k}: {v}" for k, v in cannot.items()))

            cov.plan(runnable)
            reclaimed = cov.reclaim_stale()
            if reclaimed:
                self._log(f"Reclaimed {reclaimed} stale leases from a session "
                          f"that did not shut down")

            revived = cov.revive_failed()
            if revived["revived"]:
                self._log(f"Auto-revived {revived['revived']} failed rows "
                          f"(exhausted: {revived['exhausted']})")

            budget = max(res.get("usable_vram_mb", 0) - self.vram_headroom_mb,
                         512)
            cohorts = registry.plan_cohorts(
                runnable, budget, max(res.get("gpu_count", 0), 1),
                res.get("disk_free_mb", 0))
            with self._lock:
                self.cohorts = [c.as_dict() for c in cohorts]
            self._log(f"Plan: {len(cohorts)} cohorts over {len(runnable)} "
                      f"passes, {budget} MB per card")

            worked = 0
            for cohort in cohorts:
                if self._stopping():
                    break
                worked += self._run_cohort(cohort)
                self._unload(f"cohort {cohort.index} complete")
                self._publish(f"cohort-{cohort.index}")

            if self._stopping():
                break
            if worked == 0:
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
        self._publish("final")
        self._log("Stopped cleanly")

    def _sleep(self, seconds: float) -> None:
        """Interruptible wait, in short slices, so stop is felt immediately."""
        end = time.time() + seconds
        while time.time() < end and not self._stopping():
            time.sleep(min(1.0, max(end - time.time(), 0)))

    # ── one cohort ───────────────────────────────────────────────────────
    def _run_cohort(self, cohort) -> int:
        """Run every pass in this cohort over every video that still needs one.

        `seen` is why a failure does not become a spin: a video whose pass
        failed is still claimable — that is the point of retries — and without
        remembering that it was attempted this cohort, `candidates` would hand
        it straight back and the loop would grind on the same reel until its
        attempts ran out. Retries belong to the next rotation, not this one.
        """
        ids = list(cohort.components)
        with self._lock:
            self.cohort_index = cohort.index
            self.message = (f"Cohort {cohort.index + 1}: "
                            f"{', '.join(registry.get(i).title for i in ids[:4])}"
                            + (f" +{len(ids) - 4} more" if len(ids) > 4 else ""))
        self._log(f"Cohort {cohort.index}: {len(ids)} passes, "
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
        if self.since_publish >= self.publish_every:
            self._publish(f"cohort-{cohort.index}")

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

        # Dependency gate. A pass whose input was never produced should not
        # spend a GPU second discovering that for itself.
        broken = [n for n in comp.needs
                  if states.get(n) in ("failed", "skipped")]
        if broken:
            why = (f"depends on {', '.join(broken)}, which produced nothing "
                   f"for this video")
            cov.skip(key, cid, why)
            self.session["skipped"] += 1
            return "skipped"

        with self._lock:
            self.current.update({"component": cid, "title": comp.title,
                                 "component_since": time.time(), "note": ""})

        params = dict(comp.params)
        if self._hf_token:
            params.setdefault("hf_token", self._hf_token)
        job = Job(
            video=dict(video), component=comp, store=store, source=source,
            workdir=workdir, params=params, resources=self.resources,
            cache=self._cache,
            renew=lambda progress="", k=key, c=cid: cov.renew(k, c, progress),
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
        except (KeyboardInterrupt, SystemExit):
            cov.release(key, cid)
            raise
        except Exception as exc:
            state = cov.fail(key, cid, f"{type(exc).__name__}: {exc}")
            self.session["failed"] += 1
            self._log(f"{key} · {cid}: {type(exc).__name__}: "
                      f"{str(exc)[:200]}", "error")
            if self._is_oom(exc):
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

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        name = type(exc).__name__
        return ("OutOfMemory" in name
                or "CUDA out of memory" in str(exc)
                or "CUBLAS_STATUS_ALLOC_FAILED" in str(exc))

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
        if not needs & {"artifacts", "keyframes"}:
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
        if "keyframes" in needs and not os.path.exists(
                job.path("frames/index.json")):
            self._rebuild(job, "keyframes", "frame index absent")

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
        loaded = self._cache.loaded()
        self._cache.unload_all()
        if loaded:
            self._log(f"Unloaded {len(loaded)} models"
                      + (f" — {why}" if why else ""))

    def _publish(self, note: str) -> str:
        """Export everything written since the last shard and send it up.

        Best effort by design. A session with no bot token still produces a
        complete local database, and a shard that fails to upload stays on disk
        with its watermark unmoved, so the next attempt includes it rather than
        losing it.
        """
        store = self.store
        with self._lock:
            lo_id = int(store.get_meta("shard_lo_id", "0") or 0)
            lo_vec = int(store.get_meta("shard_lo_vec", "0") or 0)
            hi_id = store.max_claim_id()
            hi_vec = store.max_vector_id()
            if hi_id <= lo_id and hi_vec <= lo_vec:
                self.since_publish = 0
                return ""

            seq = int(store.get_meta("shard_seq", "0") or 0) + 1
            sid = f"{intake.site_id(store)}-{seq:04d}"
            path = os.path.join(self.shard_dir, intake.shard_name(sid))
            try:
                stats = store.export_shard(path, lo_id, hi_id, lo_vec, hi_vec,
                                           note)
            except Exception as exc:
                self._log(f"Shard export failed: {type(exc).__name__}: {exc}",
                          "error")
                return ""

            msg_id = None
            if self._tg and self._tg.token:
                caption = (f"vios evidence · {sid}\n"
                           f"{stats['claims']} claims, {stats['vectors']} "
                           f"vectors\nworker {self.index + 1}/"
                           f"{self.partitions} · {note}")
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
            store.set_meta("shard_seq", str(seq))
            store.checkpoint()

            self.since_publish = 0
            self.last_publish = time.time()
            self.session["shards"] += 1
            self._log(f"Shard {sid}: {stats['claims']} claims, "
                      f"{stats['vectors']} vectors, "
                      f"{stats['bytes'] / 1024:.0f} KB"
                      + (f" → message {msg_id}" if msg_id else
                         " (held locally — no upload)"))
            if msg_id is None and self._tg and self._tg.token:
                return ""
            return sid

    def publish_now(self) -> dict:
        """The tab's manual button. Also the right thing to press before
        closing a Kaggle session early."""
        sid = self._publish("manual")
        return {"ok": bool(sid), "shard": sid}

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
        try:
            stats = self.store.stats()
            matrix = self.coverage.matrix()
            counts = self.coverage.counts()
            failures = self.coverage.failures(12)
            running = self.coverage.running()
            retry = self.coverage.retry_state()
            shards = self.store.shards()
            pending = max(self.store.max_claim_id() -
                          int(self.store.get_meta("shard_lo_id", "0") or 0), 0)
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
            "uptime": (round(time.time() - self.started_at, 1)
                       if self.started_at else 0),
        }

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
