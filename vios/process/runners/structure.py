"""
vios.process.runners.structure — probe, artifacts, shots, keyframes, allframes.

The five passes that turn a file into something the rest of the pipeline can
address. None of them produces an opinion; all of them produce geometry. If
these are wrong every claim downstream inherits the error, which is why the
shot detector gets two algorithms and the keyframe picker gets a sharpness
criterion instead of taking the midpoint.

TransNetV2 is the only model here, and it is 90 MB.
"""

from __future__ import annotations

import json
import os

from .. import frames as frames_module
from .. import media
from .base import Emission, Job, SkipPass


# ══════════════════════════════════════════════════════════════════════════
# probe
# ══════════════════════════════════════════════════════════════════════════

def probe(job: Job) -> Emission:
    """ffprobe → the video row. Writes no claims.

    Duration, resolution and frame rate are container facts. Storing them as
    claims would invite the interface to show a confidence next to "1080×1920",
    which is meaningless, and would let two observers disagree about a number
    that has exactly one correct value.
    """
    if not os.path.exists(job.source):
        raise SkipPass(f"source file is missing: {job.source}")
    meta = media.probe(job.source)
    if not meta["duration"]:
        raise SkipPass("container reports zero duration — file is truncated")

    job.store.update_video(
        job.key, duration=meta["duration"], width=meta["width"],
        height=meta["height"], fps=meta["fps"],
        has_audio=1 if meta["has_audio"] else 0,
        bytes=meta["size_bytes"] or None)

    em = Emission()
    em.notes = meta
    # Refresh the cached row so passes later in this cohort see the real
    # duration instead of whatever the capture record guessed.
    job.video.update({k: meta[k] for k in
                      ("duration", "width", "height", "fps")})
    job.video["has_audio"] = 1 if meta["has_audio"] else 0
    return em


# ══════════════════════════════════════════════════════════════════════════
# artifacts
# ══════════════════════════════════════════════════════════════════════════

def artifacts(job: Job) -> Emission:
    """Proxy, poster, sprite, loop, wav, waveform.

    Partial success is success. A reel with no audio gets five of the six
    artifacts and an error note for the sixth, because the alternative — fail
    the pass, retry it three times, retire it — would also lose the proxy, and
    the proxy is what the site plays.
    """
    if not media.have_ffmpeg():
        raise SkipPass("ffmpeg is not on PATH")
    meta = {k: job.video.get(k) for k in
            ("duration", "width", "height", "fps")}
    meta["has_audio"] = bool(job.video.get("has_audio"))
    if not meta.get("duration"):
        meta = media.probe(job.source)

    out = media.build_artifacts(
        job.source, job.workdir, meta,
        proxy_height=int(job.params.get("proxy_height", 480)),
        sprite_cols=int(job.params.get("sprite_cols", 10)),
        loop_seconds=float(job.params.get("loop_seconds", 4)))

    em = Emission()
    for name, info in out["artifacts"].items():
        em.artifact(name, info["path"],
                    {k: v for k, v in info.items() if k != "path"})
    em.notes = {"errors": out["errors"], "made": sorted(out["artifacts"])}
    if not out["artifacts"]:
        raise SkipPass("ffmpeg produced nothing: " +
                       "; ".join(f"{k}: {v}" for k, v in out["errors"].items()))
    if out["errors"]:
        job.note("partial: " + ", ".join(out["errors"]))
    return em


# ══════════════════════════════════════════════════════════════════════════
# shots
# ══════════════════════════════════════════════════════════════════════════

_TRANSNET_REPO = "Soulter/TransNetV2"


def _load_transnet(job: Job):
    """TransNetV2 weights, or None.

    Returns None rather than raising when the model is unavailable, because the
    PySceneDetect prefilter alone is a usable shot list — worse on dissolves,
    fine on hard cuts, and re-runnable later under a new observer once the
    weights are present. A session that cannot reach Hugging Face should still
    produce a database.
    """
    def loader():
        import torch  # noqa: PLC0415
        from transnetv2_pytorch import TransNetV2  # noqa: PLC0415
        model = TransNetV2()
        state = torch.hub.load_state_dict_from_url(
            "https://github.com/soCzech/TransNetV2/raw/master/inference-pytorch/"
            "transnetv2-pytorch-weights.pth", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        if job.resources.get("gpu_count"):
            model = model.cuda()
        return model

    try:
        return job.cache.get("transnetv2", loader)
    except Exception as exc:
        job.note(f"TransNetV2 unavailable ({type(exc).__name__}); "
                 f"using the PySceneDetect prefilter alone")
        return None


def _transnet_boundaries(model, path: str, threshold: float) -> tuple:
    """Frame indices where TransNetV2 says a shot ends, and the raw curve.

    The model wants 48×27 RGB at the video's own frame rate and predicts a
    per-frame probability that the frame is inside a transition. Everything
    here is arithmetic on that curve — no interpretation, and no model asked
    for a timestamp.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (48, 27))[:, :, ::-1])
    cap.release()
    if len(frames) < 3:
        return [], []

    arr = np.asarray(frames, dtype=np.uint8)
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    probs = []
    # 100-frame windows with 25 frames of context each side, which is what the
    # architecture expects; a shorter window degrades exactly at the boundaries
    # it is meant to find.
    window, pad = 100, 25
    padded = np.concatenate([
        np.repeat(arr[:1], pad, axis=0), arr,
        np.repeat(arr[-1:], pad + window, axis=0)])
    for start in range(0, len(arr), window):
        chunk = padded[start:start + window + 2 * pad]
        if len(chunk) < window + 2 * pad:
            break
        with torch.no_grad():
            t = torch.from_numpy(chunk.copy()).unsqueeze(0).to(device)
            single, _ = model(t)
            single = torch.sigmoid(single).cpu().numpy().reshape(-1)
        probs.extend(single[pad:pad + window].tolist())
    probs = probs[:len(arr)]
    return [i for i, p in enumerate(probs) if p >= threshold], probs


def shots(job: Job) -> Emission:
    """The shot list. Prefilter, then refine, then enforce a minimum length.

    PySceneDetect proposes and TransNetV2 disposes: the neural boundaries win
    where they exist, and the threshold detector's cuts survive only where the
    model agrees within a few frames. Boundaries closer together than
    `min_shot_frames` are merged, because a two-frame "shot" is a flash frame
    and giving it an index would let every later pass emit claims about it.
    """
    fps = float(job.video.get("fps") or 0) or 30.0
    duration = job.duration or 0.0
    if duration <= 0:
        raise SkipPass("no duration — run probe first")

    candidates = media.scene_candidates(
        job.media, threshold=float(job.params.get("threshold_content", 27.0)))
    detector = "pyscenedetect"
    cuts = sorted({round(a, 3) for a, _ in candidates} |
                  {round(b, 3) for _, b in candidates})

    model = _load_transnet(job)
    if model is not None:
        try:
            job.heartbeat("transnetv2")
            indices, _probs = _transnet_boundaries(
                model, job.media, float(job.params.get("threshold", 0.5)))
            neural = sorted({round(i / fps, 3) for i in indices})
            if neural:
                cuts = sorted(set(neural) | {0.0, round(duration, 3)})
                detector = "pyscenedetect+transnetv2"
        except Exception as exc:
            job.note(f"TransNetV2 failed ({type(exc).__name__}: {exc}); "
                     f"keeping the prefilter's boundaries")

    min_len = max(float(job.params.get("min_shot_frames", 8)) / fps, 0.05)
    merged = [0.0]
    for c in cuts:
        if c <= 0.0 or c >= duration:
            continue
        if c - merged[-1] >= min_len:
            merged.append(round(c, 3))
    if duration - merged[-1] < min_len and len(merged) > 1:
        merged.pop()
    merged.append(round(duration, 3))

    rows = [{"t0": merged[i], "t1": merged[i + 1],
             "keyframe": round((merged[i] + merged[i + 1]) / 2.0, 3)}
            for i in range(len(merged) - 1)]
    n = job.store.set_shots(job.key, rows, detector)
    job._shots = []                      # force a reload from the store

    em = Emission()
    em.shots = rows
    em.notes = {"shots": n, "detector": detector,
                "asl": round(duration / max(n, 1), 3)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# keyframes
# ══════════════════════════════════════════════════════════════════════════

def _sharpness(path: str) -> float:
    """Variance of the Laplacian — the standard blur measure.

    Higher is sharper. It is scale-dependent, so it may only be compared
    between frames of the same size, which is exactly how it is used here:
    candidates within one shot, all extracted at the same width.
    """
    try:
        import cv2  # noqa: PLC0415
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


def keyframes(job: Job) -> Emission:
    """One representative frame per shot, chosen for sharpness.

    Five candidates spread across the middle 70% of each shot, and the
    sharpest wins. The midpoint would be the obvious choice and is often the
    worst one — a shot with a whip pan is blurriest in the middle, and every
    vision pass reading that frame reports a smear.
    """
    rows = job.shots()
    if not rows:
        raise SkipPass("no shots — run the shots pass first")

    per_shot = max(int(job.params.get("candidates_per_shot", 5)), 1)
    outdir = os.path.join(job.workdir, "frames")
    os.makedirs(outdir, exist_ok=True)

    index, em = [], Emission()
    for n, s in enumerate(rows):
        if n % 20 == 0:
            job.heartbeat(f"shot {n}/{len(rows)}")
        span = float(s["t1"]) - float(s["t0"])
        times = ([float(s["t0"]) + span * (0.15 + 0.70 * k / max(per_shot - 1, 1))
                  for k in range(per_shot)] if per_shot > 1
                 else [float(s["t0"]) + span * 0.5])
        cands = media.frames_at(job.media, times, outdir, width=640,
                                prefix=f"s{int(s['idx']):04d}_")
        if not cands:
            continue
        best = max(cands, key=lambda c: _sharpness(c["path"]))
        final = os.path.join(outdir, f"shot{int(s['idx']):04d}.jpg")
        try:
            if os.path.exists(final):
                os.remove(final)
            os.replace(best["path"], final)
        except OSError:
            final = best["path"]
        for c in cands:
            if c["path"] != final and os.path.exists(c["path"]):
                try:
                    os.remove(c["path"])
                except OSError:
                    pass
        index.append({"shot_idx": int(s["idx"]), "t": best["t"],
                      "path": final})

    if not index:
        raise SkipPass("could not extract a frame from any shot")

    with open(os.path.join(outdir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    job._frames = index

    em.artifact("frames", outdir, {"count": len(index)})
    em.notes = {"frames": len(index), "shots": len(rows)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# allframes
# ══════════════════════════════════════════════════════════════════════════

def allframes(job: Job) -> Emission:
    """Every frame of the video, dated from the stream, with the proof beside it.

    Distinct from `keyframes`, which picks one frame per shot for the passes
    that read a handful of images. This writes the complete set, and the two
    coexist deliberately: a shot-level summary wants a representative frame, and
    a claim to have seen the whole video wants all of them.

    The pass does not fail when coverage is incomplete, and that is on purpose.
    A reel with a genuine hole in it — dropped frames from a bad upload, an
    encoder that gave up — is still a reel worth analysing, and raising here
    would send it round the retry ladder for a defect no retry can repair. The
    verdict is recorded, surfaced in the notes, and left for a person to read.
    What *does* fail is producing no frames at all, which means the stream was
    never readable and the video row is wrong.

    Reads `job.source`, not `job.media`. Every other vision pass prefers the
    480p proxy and is right to, but the proxy is a re-encode written at a
    constant rate: its frames have already been resampled, so extracting from
    it would produce a frame set whose count and timestamps describe the proxy
    rather than the video. The one pass that certifies completeness has to read
    the file the claim is about.
    """
    outdir = os.path.join(job.workdir, "allframes")
    p = job.params
    job.heartbeat("extracting every frame")

    manifest, verdict = frames_module.extract_and_verify(
        job.source, outdir,
        analysis_width=int(p.get("analysis_width", 384)),
        full=bool(p.get("full_tier", False)),
        full_width=int(p.get("full_width", 0)))

    rate = manifest["rates"]
    if job.log:
        for line in frames_module.report(verdict,
                                         os.path.basename(job.source)).splitlines():
            job.log(line)

    em = Emission()
    em.artifact("allframes", outdir, {
        "count": manifest["count"],
        "complete": verdict["complete"],
        "analysis_width": manifest["analysis_width"],
        "bytes": verdict.get("bytes", {}),
    })
    em.notes = {
        "frames": manifest["count"],
        "fps": rate["fps"],
        "vfps": rate["vfps"],
        "variable_rate": rate["variable"],
        "duration": rate["duration"],
        "complete": verdict["complete"],
        "coverage": verdict["checks"]["coverage"]["detail"],
        "count_check": verdict["checks"]["count"]["detail"],
        "timing_check": verdict["checks"]["timing"]["detail"],
    }
    if not verdict["complete"]:
        failed = [k for k, c in verdict["checks"].items() if not c["ok"]]
        em.notes["incomplete"] = ", ".join(failed)
        job.note(f"frame coverage incomplete ({', '.join(failed)}) "
                 f"— see allframes/coverage.json")
    return em
