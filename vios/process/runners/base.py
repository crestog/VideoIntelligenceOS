"""
vios.process.runners.base — the shape every pass has, and the model cache.

A runner is one function:

    def run(job: Job) -> Emission

It reads from `job`, returns claims and vectors, and writes nothing to the
database itself. The store stays the only writer, which is what makes a failed
pass leave no half-written evidence behind — an exception three quarters of the
way through a runner discards the whole emission, and the coverage row goes back
to queued with nothing to clean up.

Two ideas carry most of the weight here.

**The model cache is keyed by weights, not by component.** `describe`, `narrate`
and `style-read` are three passes over one 8 GB VLM. Loading it three times
would not fit and loading it per video would spend more time on `from_pretrained`
than on inference. The cache holds it for the whole cohort and the engine drops
it when the cohort ends.

**Shot indices are the only currency for time.** `Job.shot_at()` converts
seconds to an index; nothing converts an index back except the store, from the
shot table. A runner that wants to say "at 4.2 seconds" physically cannot — the
claim it builds has a shot index field and no time field, and the store derives
t0/t1 itself.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, field


class SkipPass(Exception):
    """This pass cannot apply to this video, and that is a correct outcome.

    A reel with no audio track is not a transcription failure. Raising this
    marks the coverage row `skipped` with the reason, which keeps the coverage
    matrix honest: 4,800 done and 200 skipped is a complete sweep, while 4,800
    done and 200 failed is an unsolved problem. Conflating the two hides real
    failures inside a number that looks fine.
    """


class Emission:
    """What a runner produces. Nothing is written until the engine accepts it."""

    __slots__ = ("claims", "vectors", "artifacts", "notes", "shots")

    def __init__(self):
        self.claims: list = []
        self.vectors: list = []
        self.artifacts: list = []
        self.notes: dict = {}
        self.shots: list = []

    def claim(self, channel: str, kind: str, value=None, *, shot_idx=None,
              confidence: float = 1.0, num=None, ordinal=None) -> None:
        """Record one observation.

        `value` is the searchable payload and `num` the sortable one; a claim
        may carry either or both. The separation matters at query time —
        "shots brighter than 0.7" is a range scan over `num`, and pushing that
        into text would make it a full-table string comparison. A dict or list
        passed as `value` is JSON-encoded by the store, which is how structured
        kinds like a colour palette travel.
        """
        c = {"channel": channel, "kind": kind, "value": value,
             "num": None if num is None else float(num),
             "shot_idx": shot_idx, "confidence": float(confidence)}
        if ordinal is not None:
            c["ordinal"] = int(ordinal)
        self.claims.append(c)

    def vector(self, space: str, values, *, shot_idx=None) -> None:
        self.vectors.append({"space": space, "values": values,
                             "shot_idx": shot_idx})

    def artifact(self, name: str, path: str, meta: dict | None = None) -> None:
        self.artifacts.append({"name": name, "path": path, "meta": meta or {}})

    def __len__(self) -> int:
        return len(self.claims) + len(self.vectors)

    def __repr__(self) -> str:
        return (f"<Emission {len(self.claims)} claims, "
                f"{len(self.vectors)} vectors>")


@dataclass
class Job:
    """Everything one pass over one video is allowed to see."""

    video: dict                       # the video row
    component: object                 # registry.Component
    store: object                     # process.store.Store
    source: str                       # the original mp4 on local disk
    workdir: str                      # this video's artifact directory
    params: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    cache: "ModelCache" = None
    renew: object = None              # callable: push the lease out
    log: object = None                # callable(str)

    _shots: list = field(default_factory=list, repr=False)
    _frames: list = field(default_factory=list, repr=False)

    # ── the video ───────────────────────────────────────────────────────
    @property
    def key(self) -> str:
        return self.video["video_key"]

    @property
    def duration(self) -> float:
        return float(self.video.get("duration") or 0.0)

    def note(self, message: str) -> None:
        if self.log:
            self.log(f"{self.key} · {self.component.id} · {message}")

    def heartbeat(self, progress: str = "") -> None:
        """Tell the ledger this is still alive. Called by long passes between
        units of work, so a VLM grinding through forty shots does not have its
        own lease expire and its row handed to another worker."""
        if self.renew:
            self.renew(progress)

    # ── artifacts ───────────────────────────────────────────────────────
    def path(self, name: str) -> str:
        return os.path.join(self.workdir, name)

    def artifact(self, name: str, required: bool = True) -> str:
        """A derived file, or a skip.

        Required-but-missing raises `SkipPass` rather than `FileNotFoundError`
        on purpose. The artifacts pass records *why* each file is absent, and
        "no audio stream" reaching the coverage matrix as a skip with a reason
        is far more useful than a traceback that says a path does not exist.
        """
        p = self.path(name)
        if os.path.exists(p):
            return p
        if required:
            raise SkipPass(f"artifact {name} is missing")
        return ""

    @property
    def media(self) -> str:
        """What vision passes should read: the proxy if it exists, else the
        original. Decoding 1080×1920 when a 480p copy is sitting next to it
        costs about four times as much for frames that get resized to 384 px
        anyway."""
        p = self.path("proxy.mp4")
        return p if os.path.exists(p) else self.source

    # ── shots ───────────────────────────────────────────────────────────
    def shots(self) -> list:
        if not self._shots:
            self._shots = self.store.shots(self.key)
        return self._shots

    def shot_at(self, t: float) -> int:
        """Seconds → shot index. The only direction that exists.

        Binary-search-free because reels have tens of shots, not thousands, and
        a linear scan over forty rows is not the bottleneck in a pass that just
        spent nine seconds in Whisper.
        """
        shots = self.shots()
        if not shots:
            return 0
        for s in shots:
            if s["t0"] <= t < s["t1"]:
                return int(s["idx"])
        return int(shots[-1]["idx"]) if t >= shots[-1]["t1"] else 0

    def shot_range(self, t0: float, t1: float) -> tuple:
        return self.shot_at(t0), self.shot_at(max(t1 - 1e-3, t0))

    def frames(self) -> list:
        """The keyframes, one per shot, as [{shot_idx, path, t}]."""
        if not self._frames:
            index = self.path("frames/index.json")
            if os.path.exists(index):
                try:
                    with open(index, "r", encoding="utf-8") as fh:
                        self._frames = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    self._frames = []
        return self._frames

    # ── evidence written by earlier passes ──────────────────────────────
    def claims(self, channel: str = "", kind: str = "") -> list:
        return self.store.claims(self.key, channel=channel, kind=kind)

    def text_bundle(self) -> dict:
        """Everything sayable about this video, assembled from claims.

        The language passes read this instead of re-reading files, which means
        they see exactly what the database contains — if a transcript is empty
        because the pass was skipped, the concept extractor sees an empty
        transcript rather than silently reaching around the database for the
        audio. What the database knows is what the models get.
        """
        def joined(channel, kind):
            rows = self.store.claims(self.key, channel=channel, kind=kind)
            return [r for r in rows if r.get("value")]

        return {
            "transcript": " ".join(
                r["value"] for r in joined("speech", "segment")),
            "on_screen": [r["value"] for r in joined("ocr", "text")],
            "caption": next((r["value"] for r in joined("caption", "caption")), ""),
            "hashtags": [r["value"] for r in joined("caption", "hashtag")],
            "comments": [r["value"] for r in joined("caption", "comment")],
            "descriptions": [(r.get("shot_idx"), r["value"])
                             for r in joined("visual", "shot_description")],
        }


# ══════════════════════════════════════════════════════════════════════════
# The model cache
# ══════════════════════════════════════════════════════════════════════════

class ModelCache:
    """Loaded weights, keyed by `Component.load_key`, held for a cohort.

    `unload_all` is not politeness — it is the mechanism the rotation loop
    depends on. Python dropping the last reference to a model does not return
    VRAM to CUDA; the caching allocator keeps it. Without the explicit
    `empty_cache` the second cohort inherits the first cohort's memory and OOMs
    on a plan the packer proved would fit.
    """

    def __init__(self, log=None):
        self._models: dict = {}
        self._loaded_at: dict = {}
        self.log = log or (lambda m: None)

    def get(self, key: str, loader) -> object:
        """Return the cached object, loading it once on first use.

        Loading is lazy rather than eager at cohort start so a cohort whose
        videos are all already done never pays for weights it will not use — a
        common case on a resumed session.
        """
        if key not in self._models:
            t0 = time.time()
            self.log(f"loading {key}")
            self._models[key] = loader()
            self._loaded_at[key] = time.time()
            self.log(f"loaded {key} in {time.time() - t0:.1f}s")
        return self._models[key]

    def has(self, key: str) -> bool:
        return key in self._models

    def drop(self, key: str) -> None:
        obj = self._models.pop(key, None)
        self._loaded_at.pop(key, None)
        if obj is None:
            return
        try:
            if hasattr(obj, "to"):
                obj.to("cpu")
        except Exception:
            pass
        del obj
        self._sweep()
        self.log(f"unloaded {key}")

    def unload_all(self) -> None:
        for key in list(self._models):
            self.drop(key)
        self._sweep()

    def loaded(self) -> list:
        return sorted(self._models)

    @staticmethod
    def _sweep() -> None:
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def device_and_dtype(resources: dict) -> tuple:
    """The device string and torch dtype this machine can actually use.

    Every GPU runner calls this instead of hardcoding `bfloat16`. On a T4 that
    would not merely be slow — Turing has no BF16 path at all, so it either
    errors at load or silently falls back to a software emulation that is
    several times slower than the FP16 it should have used.
    """
    if not resources.get("gpu_count"):
        return "cpu", "float32"
    return "cuda", resources.get("dtype", "float16")


def torch_dtype(name: str):
    import torch  # noqa: PLC0415
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.float16)
