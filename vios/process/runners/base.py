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

**Indices are the only currency for time.** `Job.shot_at()` converts seconds to
a shot index and `Job.all_frames()` hands out frame indices already dated from
the container's own presentation timestamps. Nothing converts an index back
except the store. A runner that wants to say "at 4.2 seconds" physically cannot
— the claim it builds carries an index, and the store derives t0/t1 itself.

**A pass sees every frame, or says why not.** `Job.frames()` is the old
one-keyframe-per-shot view and remains for the passes that genuinely summarise a
shot. `Job.all_frames()` / `Job.frame_batches()` are the complete set that
`structure.allframes` extracted and proved, and they are what the perception
passes read: a reel is judged on all nine hundred of its frames, not on thirty
of them. Missing frames are reported through `job.note`, never quietly dropped,
because a pass that reads 812 of 900 frames and reports success is the failure
this design exists to remove.
"""

from __future__ import annotations

import bisect
import contextlib
import gc
import json
import os
import time
from dataclasses import dataclass, field

# How often a long pass is allowed to write its lease renewal to the ledger.
# Well under `coverage.LEASE_SECONDS` (40 minutes) with room for a batch that
# takes far longer than expected, and long enough that a per-batch heartbeat
# over nine hundred frames is not nine hundred commits.
HEARTBEAT_SECONDS = 20.0


class SkipPass(Exception):
    """This pass cannot apply to this video, and that is a correct outcome.

    A reel with no audio track is not a transcription failure. Raising this
    marks the coverage row `skipped` with the reason, which keeps the coverage
    matrix honest: 4,800 done and 200 skipped is a complete sweep, while 4,800
    done and 200 failed is an unsolved problem. Conflating the two hides real
    failures inside a number that looks fine.
    """


class DeferPass(Exception):
    """Not now — but nothing is wrong, and no attempt should be spent.

    A rate-limited cloud call is the case this exists for. The work is
    perfectly runnable; the account is simply out of requests this minute. If
    that went through `fail()` it would burn one of three attempts each time
    and a busy hour would exhaust an entire archive's retries against a
    condition that clears by itself.

    So the row goes back to `queued` with `next_try_at` set, untouched
    otherwise. `retry_after` is seconds from now; the engine clamps it.
    """

    def __init__(self, reason: str, retry_after: float = 300.0):
        super().__init__(reason)
        self.retry_after = float(retry_after)


class Emission:
    """What a runner produces. Nothing is written until the engine accepts it."""

    __slots__ = ("claims", "vectors", "artifacts", "notes", "shots",
                 "frame_vectors", "frame_metrics")

    def __init__(self):
        self.claims: list = []
        self.vectors: list = []
        self.artifacts: list = []
        self.notes: dict = {}
        self.shots: list = []
        self.frame_vectors: list = []
        self.frame_metrics: list = []

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

    # ── per-frame ───────────────────────────────────────────────────────
    def frame_claim(self, frame_idx: int, frame_t: float, channel: str,
                    kind: str, value=None, *, frame_hi=None, frame_t1=None,
                    confidence: float = 1.0, num=None, ordinal=None) -> None:
        """Record one observation about a frame, or a run of frames.

        `frame_hi` makes it a run: *frames 100 through 142 all read
        "SUBSCRIBE"* is one row that still answers "what was on screen at frame
        117" exactly. That is not a summary or a sample — every frame in the
        span was looked at, and the span is only how the answer is stored.

        The time comes from the extractor's manifest, which read it out of the
        container's presentation timestamps. A model is never asked when
        something happened, here or anywhere else.
        """
        c = {"channel": channel, "kind": kind, "value": value,
             "num": None if num is None else float(num),
             "frame_idx": int(frame_idx), "frame_t": float(frame_t),
             "confidence": float(confidence)}
        if frame_hi is not None:
            c["frame_hi"] = int(frame_hi)
        if frame_t1 is not None:
            c["frame_t1"] = float(frame_t1)
        if ordinal is not None:
            c["ordinal"] = int(ordinal)
        self.claims.append(c)

    def frame_runs(self, channel: str, kind: str, readings, *,
                   confidence: float = 1.0) -> int:
        """Collapse a per-frame series into runs and emit them. Returns rows.

        `readings` is an iterable of `(frame_idx, frame_t, value)` in frame
        order, one entry per frame the pass actually read — including the empty
        ones. An empty value is not skipped silently: it ends the run before it
        and is itself dropped, so "nothing on screen from 143 to 200" is
        represented by the absence of a claim between two runs rather than by
        two hundred rows saying nothing.

        Consecutive identical values become one row. This is what makes total
        per-frame OCR affordable: a caption held on screen for two seconds is
        one row instead of sixty, and the row is still per-frame queryable.
        """
        rows = 0
        run = None                       # [lo, lo_t, hi, hi_t, value]
        ordinal = 0

        def flush():
            nonlocal run, rows, ordinal
            if run is None:
                return
            lo, lo_t, hi, hi_t, val = run
            self.frame_claim(lo, lo_t, channel, kind, val,
                             frame_hi=(hi if hi != lo else None),
                             frame_t1=(hi_t if hi != lo else None),
                             confidence=confidence, ordinal=ordinal)
            ordinal += 1
            rows += 1
            run = None

        for idx, t, value in readings:
            if value is None or value == "" or value == [] or value == {}:
                flush()
                continue
            # Compare on the serialised form so a dict of boxes rebuilt each
            # frame still matches the identical dict from the frame before.
            if run is not None and run[4] == value and idx == run[2] + 1:
                run[2], run[3] = int(idx), float(t)
            else:
                flush()
                run = [int(idx), float(t), int(idx), float(t), value]
        flush()
        return rows

    def window_runs(self, channel: str, kind: str, readings, *,
                    confidence: float = 1.0, frame_of=None) -> int:
        """Collapse a per-window audio series into runs. Returns rows written.

        `readings` is `(window_idx, t0, t1, value)` in window order, one entry
        per window the pass actually scored — including the silent ones, whose
        empty value ends the run before them and is itself dropped.

        The same bargain `frame_runs` makes, for the same reason: a music bed
        running the whole reel is one row rather than four hundred, and the row
        still answers "was there music at 7.3 s" exactly. Coverage is total; the
        run is only how it is stored.

        `frame_of` is a callable seconds → frame index (normally
        `job.frame_at`). Passing it is what puts audio on the *same* index as
        video, so a sound and a shot are joined on an integer. Without it the
        run is still written, with times only.
        """
        rows = 0
        run = None                     # [lo_i, lo_t, hi_i, hi_t, value]
        ordinal = 0

        def flush():
            nonlocal run, rows, ordinal
            if run is None:
                return
            _lo, lo_t, _hi, hi_t, val = run
            fi = frame_of(lo_t) if frame_of else None
            fhi = frame_of(hi_t) if frame_of else None
            if fi is not None:
                self.frame_claim(fi, lo_t, channel, kind, val,
                                 frame_hi=(fhi if fhi and fhi != fi else None),
                                 frame_t1=(hi_t if hi_t > lo_t else None),
                                 confidence=confidence, ordinal=ordinal)
            else:
                self.claim(channel, kind, val, num=round(lo_t, 3),
                           confidence=confidence, ordinal=ordinal)
            ordinal += 1
            rows += 1
            run = None

        for idx, t0, t1, value in readings:
            if value is None or value == "" or value == [] or value == {}:
                flush()
                continue
            if run is not None and run[4] == value and idx == run[2] + 1:
                run[2], run[3] = int(idx), float(t1)
            else:
                flush()
                run = [int(idx), float(t0), int(idx), float(t1), value]
        flush()
        return rows

    def frame_vector_set(self, space: str, frames, matrix) -> None:
        """One video's per-frame embeddings in one space, as a single row.

        `frames[i]` is the frame index whose embedding is `matrix[i]`. The
        pairing is explicit rather than assumed contiguous, so a pass that
        could not read one JPEG records the frames it did read at their real
        indices instead of shifting everything after the gap.
        """
        self.frame_vectors.append({"space": space, "frames": list(frames),
                                   "matrix": matrix})

    def frame_metric(self, name: str, frames, values) -> None:
        """One per-frame scalar series — brightness, motion, depth mean."""
        self.frame_metrics.append({"name": name, "frames": list(frames),
                                   "values": list(values)})

    def vector(self, space: str, values, *, shot_idx=None) -> None:
        self.vectors.append({"space": space, "values": values,
                             "shot_idx": shot_idx})

    def artifact(self, name: str, path: str, meta: dict | None = None) -> None:
        self.artifacts.append({"name": name, "path": path, "meta": meta or {}})

    def __len__(self) -> int:
        return (len(self.claims) + len(self.vectors)
                + len(self.frame_vectors) + len(self.frame_metrics))

    def __repr__(self) -> str:
        return (f"<Emission {len(self.claims)} claims, "
                f"{len(self.vectors)} vectors, "
                f"{len(self.frame_vectors)} frame-vector sets, "
                f"{len(self.frame_metrics)} frame metrics>")


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
    progress: object = None           # callable(str): the tab's live line
    log: object = None                # callable(str)

    last_progress: str = ""            # what the Process tab shows mid-pass

    _shots: list = field(default_factory=list, repr=False)
    _frames: list = field(default_factory=list, repr=False)
    _allframes: dict = field(default_factory=dict, repr=False)
    _frame_times: list = field(default=None, repr=False)
    _beat: float = field(default=0.0, repr=False)

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
        units of work, so a VLM grinding through forty shots — or an OCR pass
        grinding through nine hundred frames — does not have its own lease
        expire and its row handed to another worker.

        Throttled to one ledger write every `HEARTBEAT_SECONDS`, because a
        per-batch heartbeat over a large reel would otherwise be an UPDATE and
        a commit every fraction of a second. The progress string is kept
        immediately either way: the ledger is what stops the work being stolen,
        `last_progress` is what the Process tab reads, and only the first has a
        cost worth throttling.

        `progress` is the tab's live line and is therefore *not* throttled — it
        writes to a dict in memory, and a panel that updated once every twenty
        seconds during a nine-hundred-frame pass is the frozen line this whole
        change exists to remove. It is wrapped because a display callback must
        never be able to kill the pass it is reporting on.
        """
        if progress:
            self.last_progress = str(progress)
            if self.progress:
                try:
                    self.progress(str(progress))
                except Exception:                  # noqa: BLE001
                    pass
        now = time.time()
        if not self.renew:
            return
        if progress and (now - self._beat) < HEARTBEAT_SECONDS:
            return
        self._beat = now
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

    # ── every frame ─────────────────────────────────────────────────────
    def all_frames(self, tier: str = "") -> list:
        """Every frame the extractor wrote: [{i, t, path, exact}].

        This is the difference between judging a 900-frame reel on thirty
        keyframes and actually watching it. `structure.allframes` already
        extracted and *proved* the complete set; until now `perframe` was its
        only reader.

        `tier` selects resolution, and the choice is per model, not global:

          analysis  384 px — at or above what SigLIP, CLIP and Depth-Anything
                    ingest, so a larger image is resized back down and buys
                    nothing but decode time
          full      source resolution — what OCR, detection and face passes
                    need, because a 12-pixel caption or a face at the back of a
                    room is legible at 1080×1920 and gone at 384

        An empty `tier` means the component's `params["tier"]`, defaulting to
        analysis. Asking for `full` when the extractor did not write that tier
        falls back to analysis with a note rather than skipping the pass: a
        lower-resolution reading of every frame is worth far more than no
        reading at all, and the note says which one happened.
        """
        tier = tier or str(self.params.get("tier") or "analysis")
        cached = self._allframes.get(tier)
        if cached is not None:
            return cached

        root = self.path("allframes")
        mpath = os.path.join(root, "manifest.json")
        if not os.path.exists(mpath):
            raise SkipPass("no allframes manifest — run the allframes pass first")
        try:
            with open(mpath, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise SkipPass(f"allframes manifest unreadable: {exc}")

        entries = manifest.get("frames") or []
        if not entries:
            raise SkipPass("allframes manifest is empty")

        want = tier if tier in ("analysis", "full") else "analysis"
        if want == "full" and not manifest.get("full_tier"):
            self.note("full-resolution tier was not extracted — reading the "
                      "384 px analysis tier instead")
            want = "analysis"

        out = []
        missing = 0
        for e in entries:
            # manifest paths are relative to the allframes dir and already name
            # their tier ("analysis/f_000123.jpg"), so swapping the leading
            # segment is how the other tier is addressed.
            rel = str(e.get("file") or "")
            if want != "analysis" and rel.startswith("analysis/"):
                rel = want + rel[len("analysis"):]
            p = os.path.join(root, rel.replace("/", os.sep))
            if not os.path.exists(p):
                missing += 1
                continue
            t = e.get("t")
            out.append({"i": int(e.get("i", 0)),
                        "t": 0.0 if t is None else float(t),
                        "path": p, "exact": bool(e.get("exact"))})

        if not out:
            raise SkipPass(f"no readable frames in the allframes {want} tier")
        if missing:
            # Said out loud every time. A pass that quietly read 812 of 900
            # frames and reported success is the exact failure this whole
            # change exists to remove.
            self.note(f"{missing} of {len(entries)} frames missing from the "
                      f"{want} tier — reading the {len(out)} that are present")
        self._allframes[tier] = out
        return out

    def frame_count(self) -> int:
        """How many frames the manifest claims, without loading the paths."""
        try:
            with open(self.path("allframes/manifest.json"), "r",
                      encoding="utf-8") as fh:
                return int(json.load(fh).get("count") or 0)
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return 0

    def frame_batches(self, size: int = 32, tier: str = "", stride: int = 0):
        """Yield `(indices, times, paths)` over every frame, in order.

        Batching is what makes total coverage affordable: SigLIP on a T4 costs
        roughly the same for one image as for thirty-two, because the cost is
        dominated by the kernel launch and the weights already resident. A
        900-frame reel is ~28 batches.

        A heartbeat fires per batch, which renews the coverage lease. Without
        it a 900-frame OCR pass would outlive its own 40-minute lease and have
        its row handed to another worker mid-run.

        `stride` exists so an operator can sample deliberately, and it ships as
        1 everywhere in the registry — nothing in the default path skips a
        frame. It is honoured here rather than in each runner so that a pass
        cannot quietly disagree with what its own registry row advertises, and
        a stride above 1 is announced in the log because a sampled run must
        never be mistaken later for a complete one.
        """
        frames = self.all_frames(tier)
        step = int(stride or self.params.get("stride", 1) or 1)
        if step > 1:
            frames = frames[::step]
            self.note(f"stride {step} — sampling {len(frames)} frames, "
                      f"not the full set")
        size = max(1, int(size))
        total = len(frames)
        for start in range(0, total, size):
            chunk = frames[start:start + size]
            self.heartbeat(f"frame {min(start + size, total)}/{total}")
            yield ([f["i"] for f in chunk], [f["t"] for f in chunk],
                   [f["path"] for f in chunk])

    def frame_at(self, t: float) -> int | None:
        """Seconds → the index of the video frame that covers that instant.

        This is what makes audio and video one index instead of two. A Whisper
        segment, a CLAP window and an OCR run all end up addressed by the same
        integer, so "what was said, shown and heard at frame 417" is a join on a
        key rather than a comparison of floats with three different rounding
        conventions.

        The manifest is the authority because its times came out of the
        container's presentation timestamps, which is the only correct answer
        for the variable-frame-rate reels this archive is full of. `fps` is the
        fallback for a video whose frames were never extracted, and it is a
        fallback precisely because `t * fps` is wrong on exactly those reels.

        Returns None when neither is available — the caller then writes a
        shot-level claim, which is worse but still true.
        """
        t = max(float(t), 0.0)
        try:
            frames = self.all_frames("analysis")
        except SkipPass:
            frames = []
        if frames:
            times = self._frame_times
            if times is None:
                times = [f["t"] for f in frames]
                self._frame_times = times
            # bisect_right - 1: the last frame whose time is at or before t.
            pos = bisect.bisect_right(times, t) - 1
            return int(frames[max(pos, 0)]["i"])
        fps = float(self.video.get("fps") or 0.0)
        if fps > 0:
            return int(t * fps)
        return None

    # ── every audio window ──────────────────────────────────────────────
    def audio_grid(self, window: float = 0.0, hop: float = 0.0) -> list:
        """The fixed analysis grid for audio: [{i, t0, t1}] covering the video.

        The audio counterpart of `all_frames`, and it exists for the same
        reason. A pass that reports at shot granularity has answered "there was
        applause somewhere in this eleven-second shot", which cannot place the
        moment; a pass that reports on this grid has answered "applause from
        4.32 s to 5.28 s", which can.

        Defaults of 0.96 s with a 0.48 s hop are not arbitrary. 0.96 s is the
        clip length AudioSet-family taggers (PANNs, AST, CLAP) were trained to
        score, so a window is one whole model input rather than a slice of one;
        the half-window hop means every instant is covered by two windows, so an
        event landing on a boundary is still centred in one of them.

        Coverage is total by construction: windows are laid from 0 to
        `duration` with no gaps, and the last one is kept even when it is short,
        because the end of a reel is where the payoff is and dropping a partial
        window would leave the final half-second unobserved.
        """
        win = float(window or self.params.get("window_seconds", 0.96) or 0.96)
        step = float(hop or self.params.get("hop_seconds", 0.0) or win / 2.0)
        win = max(win, 0.05)
        step = max(min(step, win), 0.01)
        dur = float(self.duration or 0.0)
        if dur <= 0:
            return []
        out, i, t = [], 0, 0.0
        while t < dur - 1e-6:
            out.append({"i": i, "t0": round(t, 4),
                        "t1": round(min(t + win, dur), 4)})
            i += 1
            t += step
        return out

    def window_at(self, t: float, window: float = 0.0, hop: float = 0.0) -> int:
        """Seconds → the index of the grid window that *starts* nearest below.

        This is what puts a Whisper segment, whose timings come from its own
        decoder, onto the same index as every tagger window — so "what was said"
        and "what it sounded like" can be joined on one integer rather than
        compared as two sets of floats.
        """
        win = float(window or self.params.get("window_seconds", 0.96) or 0.96)
        step = float(hop or self.params.get("hop_seconds", 0.0) or win / 2.0)
        step = max(min(step, win), 0.01)
        return max(0, int(float(t) / step))

    def audio_batches(self, size: int = 16, window: float = 0.0,
                      hop: float = 0.0):
        """Yield `(indices, spans)` over the whole grid, heartbeat per batch.

        Same bargain as `frame_batches`: batching amortises the kernel launch,
        and the heartbeat renews the lease so a long tagging pass over a grid of
        several hundred windows cannot have its row stolen mid-run.
        """
        grid = self.audio_grid(window, hop)
        size = max(1, int(size))
        total = len(grid)
        for start in range(0, total, size):
            chunk = grid[start:start + size]
            self.heartbeat(f"window {min(start + size, total)}/{total}")
            yield ([w["i"] for w in chunk],
                   [(w["t0"], w["t1"]) for w in chunk])

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

    Which is exactly why nothing in here is allowed to fail quietly. Every
    load, unload and sweep reports what it did and how much memory moved, and
    every fault is both logged and kept in `failures` so the Process tab can
    show it. The number that matters is MB reclaimed against MB loaded: when
    those disagree, the next cohort is already doomed and this is the only
    place that knows it.
    """

    def __init__(self, log=None):
        self._models: dict = {}
        self._loaded_at: dict = {}
        self._footprint: dict = {}       # load_key → MB the load actually cost
        self.context = ""                # component id the engine is running
        self.failures: list = []         # every load/unload fault, in order
        self._log = log
        self._quiet = log is None

    # ── saying what happened ─────────────────────────────────────────────
    def log(self, message: str, level: str = "info") -> None:
        """One line, to whoever is listening.

        The engine's logger puts it in the activity ring the Process tab reads
        *and* on stdout for warnings and errors, so a load fault survives in
        the Kaggle log to be read afterwards. When no logger was supplied — a
        cache built in isolation — faults are printed here instead, because a
        silent failure in this class is the exact bug this phase exists to
        remove.
        """
        if self._quiet:
            if level in ("warn", "error"):
                print(f"[models] {level}: {message}", flush=True)
            return
        try:
            self._log(message, level)
        except TypeError:                # a one-argument logger
            self._log(message)

    @staticmethod
    def vram() -> dict:
        """Allocated / reserved / free VRAM in MB, or `{}` without CUDA.

        Deliberately never raises. It is called from inside failure handlers,
        and a probe that throws while reporting a fault destroys the report
        that was the whole point of running it.
        """
        try:
            import torch  # noqa: PLC0415
            if not torch.cuda.is_available():
                return {}
            mb = 1024 ** 2
            free, total = torch.cuda.mem_get_info()
            return {"allocated": torch.cuda.memory_allocated() // mb,
                    "reserved": torch.cuda.memory_reserved() // mb,
                    "free": free // mb, "total": total // mb}
        except Exception:
            return {}

    @staticmethod
    def _vram_line(v: dict) -> str:
        if not v:
            return "no CUDA"
        return (f"{v['allocated']} MB allocated, {v['reserved']} MB reserved, "
                f"{v['free']} MB free of {v['total']}")

    def get(self, key: str, loader, component: str = "") -> object:
        """Return the cached object, loading it once on first use.

        Loading is lazy rather than eager at cohort start so a cohort whose
        videos are all already done never pays for weights it will not use — a
        common case on a resumed session.

        A loader that raises is reported in full and re-raised. In full because
        a download that timed out and an architecture the GPU cannot run are
        the same three words in a traceback and completely different problems:
        the first is worth retrying, the second never is. The VRAM readings on
        either side of the attempt are what separate "the weights would not
        fit" from "the weights would not load".
        """
        if key in self._models:
            return self._models[key]

        who = component or self.context or "?"
        before = self.vram()
        t0 = time.time()
        self.log(f"loading {key} for {who} — {self._vram_line(before)}")
        try:
            obj = loader()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            after = self.vram()
            detail = (f"{who} · {key} · load failed · "
                      f"{type(exc).__name__}: {str(exc)[:300]} · "
                      f"VRAM before {self._vram_line(before)} · "
                      f"after {self._vram_line(after)}")
            self.failures.append({"at": time.time(), "key": key,
                                  "component": who, "phase": "load",
                                  "error": f"{type(exc).__name__}: {exc}"[:300],
                                  "vram_before": before, "vram_after": after})
            self.log(detail, "error")
            raise

        self._models[key] = obj
        self._loaded_at[key] = time.time()
        after = self.vram()
        cost = max((after.get("allocated", 0) - before.get("allocated", 0)), 0)
        self._footprint[key] = cost
        self.log(f"loaded {key} in {time.time() - t0:.1f}s — "
                 f"{cost} MB, {self._vram_line(after)}")
        return obj

    def has(self, key: str) -> bool:
        return key in self._models

    def drop(self, key: str) -> int:
        """Release one model. Returns the MB actually reclaimed.

        The move to CPU used to be wrapped in a bare `except: pass`, which
        defeated this class's entire purpose. A model that refuses to release
        its VRAM does not fail here — it fails two cohorts later as an OOM on a
        plan the packer proved would fit, naming a model that was innocent. So
        the failure is reported where it happens, with the number that proves
        it: how much came back against how much the load cost.
        """
        obj = self._models.pop(key, None)
        cost = self._footprint.pop(key, 0)
        self._loaded_at.pop(key, None)
        if obj is None:
            return 0

        before = self.vram()
        if hasattr(obj, "to"):
            try:
                obj.to("cpu")
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                # Not fatal — the reference is dropped either way and the sweep
                # below may still reclaim the memory. But it is said out loud,
                # because a model stuck on the GPU is the leak that OOMs the
                # next cohort.
                self.failures.append(
                    {"at": time.time(), "key": key, "component": self.context,
                     "phase": "unload",
                     "error": f"{type(exc).__name__}: {exc}"[:300]})
                self.log(f"{key} would not move to CPU — "
                         f"{type(exc).__name__}: {str(exc)[:200]}", "warn")
        del obj
        self._sweep(f"drop {key}")
        after = self.vram()
        freed = max(before.get("allocated", 0) - after.get("allocated", 0), 0)

        line = f"unloaded {key} — {freed} MB reclaimed"
        if cost:
            line += f" of {cost} MB loaded"
        if before:
            line += f", {after.get('free', 0)} MB free"
        # A drop that returns materially less than the load cost is the leak.
        # Half is the threshold because fp16 activations and a few cached
        # buffers can legitimately hold on to a little; losing most of it
        # cannot be explained that way.
        if cost >= 256 and freed < cost // 2:
            self.failures.append(
                {"at": time.time(), "key": key, "component": self.context,
                 "phase": "leak",
                 "error": f"reclaimed {freed} MB of {cost} MB"})
            self.log(line + " — VRAM was not released; the next cohort will "
                            "be short by the difference", "warn")
        else:
            self.log(line)
        return freed

    def unload_all(self) -> int:
        """Drop everything resident. Returns total MB reclaimed."""
        freed = 0
        for key in list(self._models):
            try:
                freed += self.drop(key)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                # One model that cannot be dropped must not strand the rest of
                # the cohort on the GPU.
                self.failures.append(
                    {"at": time.time(), "key": key, "component": self.context,
                     "phase": "unload",
                     "error": f"{type(exc).__name__}: {exc}"[:300]})
                self.log(f"dropping {key} raised {type(exc).__name__}: "
                         f"{str(exc)[:200]} — continuing with the rest",
                         "error")
                self._models.pop(key, None)
        self._sweep("unload_all")
        return freed

    def loaded(self) -> list:
        return sorted(self._models)

    def footprints(self) -> dict:
        """load_key → MB the load cost, for the status surface."""
        return dict(self._footprint)

    def recent_failures(self, limit: int = 20) -> list:
        return self.failures[-limit:]

    def _sweep(self, why: str = "") -> None:
        """Return freed blocks to the driver. Reports what it cannot do.

        `empty_cache` failing is not cosmetic: it means the caching allocator
        still holds the last cohort's memory, and the packer's plan for the
        next one is wrong. It used to be swallowed.
        """
        gc.collect()
        try:
            import torch  # noqa: PLC0415
        except Exception:
            return                       # no torch at all — nothing to sweep
        try:
            if not torch.cuda.is_available():
                return
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            self.failures.append(
                {"at": time.time(), "key": "", "component": self.context,
                 "phase": "sweep",
                 "error": f"{type(exc).__name__}: {exc}"[:300]})
            self.log(f"CUDA sweep failed{(' after ' + why) if why else ''} — "
                     f"{type(exc).__name__}: {str(exc)[:200]}; VRAM may not "
                     f"have been returned", "error")


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


def cuda_ordinal() -> int:
    """Which card this thread is on, for the few libraries that demand a number.

    Almost nothing needs this. `device_and_dtype` returns a bare `"cuda"` and
    every `.to("cuda")` in the runners resolves against CUDA's *current device*,
    which is per host thread — so a worker that calls `torch.cuda.set_device(i)`
    once puts its whole cohort on card `i` with no runner edits at all.

    The exceptions are libraries that take a device *string* of their own and
    parse it themselves: PaddleOCR's `"gpu:0"` and onnxruntime's `ctx_id`. A
    literal `0` there sends every worker to card 0 no matter which thread it is,
    which is how one card ends up holding six gigabytes while the other holds
    a hundred megabytes and neither is busy. Asking is one call; guessing is a
    silent half-machine.
    """
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:                                      # noqa: BLE001
        pass
    return 0


@contextlib.contextmanager
def no_flash_attn():
    """Let `trust_remote_code` models load without flash-attention installed.

    `transformers.dynamic_module_utils.get_imports` regex-scans a remote module
    for every import it contains — *including the ones inside
    `if is_flash_attn_2_available():` guards* — and `check_imports` then refuses
    to run the file until each one is installed. Florence-2's remote code has
    such a guard, so on a machine without flash-attn the load dies before a
    single weight is read, and it dies with an exception that names neither
    flash-attn nor the guard.

    flash-attn cannot be installed here at all: it requires Ampere or newer and
    Kaggle's T4s are Turing. The guard would have evaluated to False and the
    model would have used SDPA, which `resources.capabilities()` already reports
    as this machine's attention path. So the import list is filtered rather than
    satisfied — this is the documented workaround for
    huggingface/transformers#31793 — and restored on the way out, because the
    patch is global and a later pass may legitimately want the real answer.
    """
    try:
        from transformers import dynamic_module_utils  # noqa: PLC0415
    except Exception:                                  # noqa: BLE001
        yield False
        return

    original = getattr(dynamic_module_utils, "get_imports", None)
    if original is None:
        yield False
        return

    def filtered(filename):
        return [m for m in original(filename) if m != "flash_attn"]

    dynamic_module_utils.get_imports = filtered
    try:
        yield True
    finally:
        dynamic_module_utils.get_imports = original
