"""
vios.process.runners.signal — the passes that compute rather than infer.

Nine of the thirty components in the registry never load a neural network, and
between them they carry the parts of the database that have to be exactly
right: how fast it cuts, how loud it gets, what colour it is, where the camera
moves, where the beat falls, what the creator wrote, what the audience wrote
back, and what happens in the first three seconds.

This is the user's rule made literal:

    "dont force llms or models to do things that they cant do by prompting,
     use mathmatical and proven systems for specific things"

A vision-language model asked "how fast does this cut?" answers "quickly, with
an energetic rhythm". `numpy.diff` over the shot table answers "2.1 seconds
average, coefficient of variation 0.44, accelerating after shot 9". Only one of
those can be sorted, filtered, compared across five thousand reels, or shown to
be wrong.

Every function here is deterministic. Run it twice, get the same claims, with
the same uid, which `INSERT OR IGNORE` then collapses to one row — so a
half-finished pass that is retried costs nothing and duplicates nothing.
"""

from __future__ import annotations

import json
import math
import os
import re

from .. import media
from .base import Emission, Job, SkipPass


def _np():
    import numpy  # noqa: PLC0415
    return numpy


# ══════════════════════════════════════════════════════════════════════════
# caption — the creator's words and the audience's
# ══════════════════════════════════════════════════════════════════════════

_HASHTAG = re.compile(r"#([\wऀ-ॿ஀-௿ఀ-౿]+)")
_MENTION = re.compile(r"@([A-Za-z0-9._]+)")


def _unwrap(rec: dict) -> dict:
    """A vios capture record → the flat shape this pass reads.

    The record nests its content under `post`, `engagement` and `comments`, but
    it also carries `raw` — the untouched info JSON — precisely so that nothing
    downstream has to guess at a normalisation done two years earlier. So `raw`
    is the base, and the normalised fields are laid over it where they are
    genuinely better: engagement because `_first_number` already reconciled
    `view_count` against `play_count`, and comments because they are deduped,
    length-capped and carry `is_creator`.
    """
    post = rec.get("post") or {}
    eng = rec.get("engagement") or {}
    raw = rec.get("raw")
    info = dict(raw) if isinstance(raw, dict) else {}

    if not info.get("description"):
        info["description"] = post.get("description") or post.get("title") or ""
    if not info.get("uploader"):
        info["uploader"] = post.get("uploader") or ""
    if info.get("duration") is None:
        info["duration"] = post.get("duration")

    for src, dst in (("views", "view_count"), ("likes", "like_count"),
                     ("comments", "comment_count"),
                     ("reposts", "repost_count")):
        if isinstance(eng.get(src), (int, float)):
            info[dst] = eng[src]

    norm = rec.get("comments")
    if isinstance(norm, list) and norm:
        info["comments"] = [
            {"text": c.get("text"), "like_count": c.get("likes"),
             "author": c.get("author"), "is_creator": c.get("is_creator"),
             "parent": c.get("parent")}
            for c in norm if isinstance(c, dict)]
    if isinstance(rec.get("comments_captured"), int):
        info["comments_captured"] = rec["comments_captured"]
    return info


def _from_head(head: dict) -> dict:
    """The capture head, when the record document itself could not be fetched.

    Engagement and the uploader only. The ledger also holds a `title`, and it
    would be easy to pass it off as the caption — it is often the caption's
    first line — but "often" is not a basis for a claim that later passes will
    read as the creator's own words. A thin, true record beats a full, guessed
    one.
    """
    out = {}
    for src, dst in (("views", "view_count"), ("likes", "like_count"),
                     ("comment_count", "comment_count")):
        if isinstance(head.get(src), (int, float)):
            out[dst] = head[src]
    if head.get("uploader"):
        out["uploader"] = head["uploader"]
    return out


def _info_json(job: Job) -> dict:
    """The capture record, from wherever this session can reach it.

    Four sources, richest first: the `record.json` intake pulled down beside
    the mp4, yt-dlp's own sidecar if the file was fetched directly here, a
    record stored whole in `video.meta`, and finally the capture head — the
    pointers and engagement figures `sync` copied out of the capture ledger,
    which is all that survives when the record document has gone missing from
    the channel.
    """
    candidates = (os.path.join(job.workdir, "record.json"),
                  os.path.splitext(job.source)[0] + ".info.json",
                  job.source + ".info.json")
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or not data:
            continue
        return _unwrap(data) if "post" in data else data

    meta = job.video.get("meta")
    if isinstance(meta, str) and meta.strip():
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = None
    if isinstance(meta, dict) and meta:
        if "post" in meta:
            return _unwrap(meta)
        if isinstance(meta.get("capture"), dict):
            return _from_head(meta["capture"])
        if any(k in meta for k in ("description", "comments", "like_count")):
            return meta
    return {}


def caption(job: Job) -> Emission:
    """Caption, hashtags, mentions, comments, engagement — parsed, not read.

    Hashtags come out with a regular expression that includes Devanagari and
    the southern scripts, because `\\w` under Python's Unicode rules already
    matches them but the ASCII-only pattern people reach for first does not,
    and a silently empty hashtag list looks exactly like a reel with no
    hashtags.

    Comments are stored verbatim with their like counts. They are the only
    direct evidence in the archive of what an audience actually thought, and
    no summary of them is worth as much as the sentences themselves.
    """
    info = _info_json(job)
    if not info:
        raise SkipPass("no capture record for this video")

    em = Emission()
    text = (info.get("description") or info.get("caption") or "").strip()
    if text:
        em.claim("caption", "caption", text)
        em.claim("caption", "caption_length", num=len(text),
                 value=f"{len(text)} characters")
        words = [w for w in re.split(r"\s+", text) if w]
        em.claim("caption", "caption_words", num=len(words))

    for i, tag in enumerate(dict.fromkeys(_HASHTAG.findall(text))):
        em.claim("caption", "hashtag", tag.lower(), ordinal=i)
    for i, who in enumerate(dict.fromkeys(_MENTION.findall(text))):
        em.claim("caption", "mention", who.lower(), ordinal=i)

    for key, kind in (("like_count", "likes"), ("comment_count", "comments"),
                      ("view_count", "views"), ("repost_count", "reposts"),
                      ("duration", "declared_duration")):
        val = info.get(key)
        if isinstance(val, (int, float)):
            em.claim("caption", kind, num=val, value=str(val))

    comments = info.get("comments") or []
    for i, c in enumerate(comments[:200]):
        body = (c.get("text") or "").strip() if isinstance(c, dict) else str(c)
        if not body:
            continue
        em.claim("caption", "comment", body, ordinal=i,
                 num=float(c.get("like_count") or 0) if isinstance(c, dict) else None)

    # The creator answering their own audience is a different kind of evidence
    # from the audience talking, and it is buried in the same list. Pulling it
    # out here means a later question — how does this creator handle a hostile
    # comment — is a filter rather than a re-read of every comment.
    replies = [c for c in comments
               if isinstance(c, dict) and c.get("is_creator")]
    for i, c in enumerate(replies[:60]):
        body = (c.get("text") or "").strip()
        if body:
            em.claim("caption", "creator_reply", body, ordinal=i,
                     num=float(c.get("like_count") or 0))

    got = info.get("comments_captured")
    if isinstance(got, int):
        em.claim("caption", "comments_captured", num=got,
                 value=f"{got} of {info.get('comment_count') or '?'} held")

    if info.get("uploader") or info.get("channel"):
        em.claim("caption", "uploader",
                 info.get("uploader") or info.get("channel"))

    em.notes = {"hashtags": len(_HASHTAG.findall(text)),
                "comments": len(comments), "creator_replies": len(replies),
                "caption_chars": len(text)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# cuts — editing rhythm
# ══════════════════════════════════════════════════════════════════════════

def cuts(job: Job) -> Emission:
    """Average shot length, cut rate, regularity, and where it accelerates.

    Cut rhythm is the most transferable craft variable in short-form video and
    it is fully determined by the shot table. The coefficient of variation is
    the interesting one: two reels can share an ASL of 1.8 s while one cuts
    metronomically and the other alternates four-second holds with half-second
    bursts, and only the CV tells them apart.
    """
    rows = job.shots()
    if not rows:
        raise SkipPass("no shots")
    np = _np()

    lengths = np.array([float(s["t1"]) - float(s["t0"]) for s in rows],
                       dtype=float)
    duration = float(job.duration or lengths.sum())
    asl = float(lengths.mean())
    cv = float(lengths.std() / asl) if asl > 0 else 0.0

    em = Emission()
    em.claim("style", "asl", f"{asl:.2f}s average shot length", num=asl)
    em.claim("style", "cut_rate",
             f"{len(rows) / max(duration, 0.001) * 60:.1f} cuts per minute",
             num=len(rows) / max(duration, 0.001) * 60)
    em.claim("style", "shot_count", f"{len(rows)} shots", num=len(rows))
    em.claim("style", "rhythm",
             "metronomic" if cv < 0.35 else "varied" if cv < 0.9 else "erratic",
             num=cv)
    em.claim("style", "longest_shot", num=float(lengths.max()),
             shot_idx=int(np.argmax(lengths)))
    em.claim("style", "shortest_shot", num=float(lengths.min()),
             shot_idx=int(np.argmin(lengths)))

    # Acceleration: compare the second half's cut rate to the first half's.
    # A reel that speeds up towards its payoff and one that slows into a
    # punchline are opposite structures with identical averages.
    if len(rows) >= 6:
        mid = duration / 2.0
        first = sum(1 for s in rows if float(s["t0"]) < mid)
        second = len(rows) - first
        ratio = second / max(first, 1)
        em.claim("style", "acceleration",
                 "accelerates" if ratio > 1.3 else
                 "decelerates" if ratio < 0.77 else "steady", num=ratio)

    em.notes = {"shots": len(rows), "asl": round(asl, 3), "cv": round(cv, 3)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# colour
# ══════════════════════════════════════════════════════════════════════════

def _palette(img, k: int = 5):
    """k-means in CIELAB, returned as hex plus each swatch's share of frame.

    Lab because Euclidean distance in it approximates how different two colours
    look, and RGB distance does not — clustering in RGB reliably merges a deep
    navy with a black and splits two greens a person would call identical.
    """
    import cv2  # noqa: PLC0415
    np = _np()

    small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype("float32")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Fixed seed: the palette of a frame must not change between runs, or the
    # claim's uid changes and the store grows a duplicate every sweep.
    cv2.setRNGSeed(12345)
    _compact, labels, centres = cv2.kmeans(
        lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k).astype(float)
    share = counts / counts.sum()
    order = np.argsort(-share)

    swatches = []
    for i in order:
        c = centres[i].reshape(1, 1, 3).astype("uint8")
        bgr = cv2.cvtColor(c, cv2.COLOR_LAB2BGR).reshape(3)
        swatches.append({"hex": "#%02X%02X%02X" % (bgr[2], bgr[1], bgr[0]),
                         "share": round(float(share[i]), 4)})
    return swatches


_COLOUR_NAMES = (
    # (name, hue lower, hue upper) in OpenCV's 0–179 hue scale.
    ("red", 0, 9), ("orange", 10, 22), ("yellow", 23, 33),
    ("green", 34, 77), ("cyan", 78, 96), ("blue", 97, 126),
    ("purple", 127, 148), ("pink", 149, 168), ("red", 169, 179),
)


def _colour_name(hue: float, sat: float, val: float) -> str:
    """A frame's dominant colour as a word, from HSV means.

    Achromatic first: a frame with almost no saturation has a hue, and it is
    meaningless — reporting "green" for a grey frame because its hue landed at
    60 is the classic way a colour index becomes untrustworthy.
    """
    if val < 0.12:
        return "black"
    if sat < 0.12:
        return "white" if val > 0.75 else "grey"
    for name, lo, hi in _COLOUR_NAMES:
        if lo <= hue <= hi:
            return name
    return "neutral"


def colour(job: Job) -> Emission:
    """Palette per shot, and the colour of every single frame.

    Two granularities on purpose. A five-swatch k-means palette is the right
    description of a *shot* and costs too much to run 900 times; the dominant
    colour of a *frame* is a hue histogram, costs nothing, and is what makes
    "find the moment the screen goes red" answerable. The palette comes from the
    keyframes, the per-frame curves and colour runs from the complete set — which
    is why this component declares both as inputs.
    """
    import cv2  # noqa: PLC0415
    np = _np()

    frames = job.frames()
    if not frames:
        raise SkipPass("no keyframes")

    k = int(job.params.get("swatches", 5))
    em = Emission()
    brights, sats, temps = [], [], []

    for n, f in enumerate(frames):
        if n % 20 == 0:
            job.heartbeat(f"palette {n}/{len(frames)}")
        img = cv2.imread(f["path"])
        if img is None:
            continue
        idx = int(f["shot_idx"])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(float) / 255.0
        s = hsv[:, :, 1].astype(float) / 255.0
        b, g, r = (img[:, :, i].astype(float).mean() for i in range(3))
        # Warm/cool as the red-blue difference, normalised. Not a colour
        # temperature in kelvin — that would need a white-point assumption the
        # footage does not supply — but monotonic with one, which is all a
        # sort or a filter needs.
        temp = (r - b) / 255.0

        em.claim("style", "palette", _palette(img, k), shot_idx=idx)
        em.claim("style", "brightness", num=round(float(v.mean()), 4),
                 shot_idx=idx)
        em.claim("style", "contrast", num=round(float(v.std()), 4),
                 shot_idx=idx)
        em.claim("style", "saturation", num=round(float(s.mean()), 4),
                 shot_idx=idx)
        em.claim("style", "temperature", num=round(float(temp), 4),
                 shot_idx=idx)
        brights.append(float(v.mean()))
        sats.append(float(s.mean()))
        temps.append(temp)

    if not brights:
        raise SkipPass("no readable keyframes")

    em.claim("style", "brightness",
             "dark" if np.mean(brights) < 0.35 else
             "bright" if np.mean(brights) > 0.65 else "mid",
             num=round(float(np.mean(brights)), 4))
    em.claim("style", "saturation",
             "desaturated" if np.mean(sats) < 0.25 else
             "saturated" if np.mean(sats) > 0.55 else "natural",
             num=round(float(np.mean(sats)), 4))
    em.claim("style", "temperature",
             "cool" if np.mean(temps) < -0.02 else
             "warm" if np.mean(temps) > 0.02 else "neutral",
             num=round(float(np.mean(temps)), 4))

    # ── every frame ─────────────────────────────────────────────────────
    idxs, hues, dominant = [], [], []
    read = 0
    for b_idx, b_t, b_paths in job.frame_batches(64):
        for i, t, path in zip(b_idx, b_t, b_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            read += 1
            hsv = cv2.cvtColor(cv2.resize(img, (64, 64),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0].astype(np.float32)
            s = hsv[:, :, 1].astype(np.float32) / 255.0
            v = hsv[:, :, 2].astype(np.float32) / 255.0
            # Hue averaged over the saturated pixels only, and by histogram
            # peak rather than mean: hue is circular, so the mean of red at 178
            # and red at 2 is cyan.
            mask = (s > 0.15) & (v > 0.15)
            if mask.any():
                hist = np.bincount(h[mask].astype("int64"), minlength=180)
                peak = float(int(hist.argmax()))
            else:
                peak = 0.0
            idxs.append(int(i))
            hues.append(peak)
            dominant.append((int(i), float(t),
                             _colour_name(peak, float(s.mean()),
                                          float(v.mean()))))

    if read:
        em.frame_metric("hue", idxs, hues)
        runs = em.frame_runs("style", "dominant_colour", dominant,
                             confidence=0.8)
        em.notes = {"shots": len(brights), "frames": read,
                    "frames_available": job.frame_count() or read,
                    "colour_runs": runs}
    else:
        job.note("no frames in the allframes set were readable — the "
                 "per-shot palette was written, the per-frame colour was not")
        em.notes = {"shots": len(brights), "frames": 0}
    return em


# ══════════════════════════════════════════════════════════════════════════
# motion — what the camera is doing
# ══════════════════════════════════════════════════════════════════════════

def motion(job: Job) -> Emission:
    """Camera movement from an affine fit, not from a model's impression.

    Track corners between consecutive frames, fit a partial affine transform,
    and read the move out of its parameters: translation dominates in a pan or
    tilt, uniform scale in a push or pull, and a high inlier residual with low
    net translation is handheld shake. This is a solved problem in computer
    vision with a closed-form answer, and asking a VLM the same question gets
    a fluent guess with no number attached.
    """
    import cv2  # noqa: PLC0415
    np = _np()

    rows = job.shots()
    if not rows:
        raise SkipPass("no shots")

    # Read the extracted frames rather than re-decoding the container. Two
    # reasons, and the second is the one that matters: decoding again would
    # sample at ~10 fps and label the motion of frames nothing else measured,
    # so a camera move could never be joined to the OCR or the objects at the
    # same instant. Reading `allframes` puts every measurement on one index.
    frames = job.all_frames()
    if len(frames) < 2:
        raise SkipPass("fewer than two frames in the allframes set")

    samples = []                       # (frame_idx, t, prev_grey, grey)
    prev = None
    read = 0
    for b_idx, b_t, b_paths in job.frame_batches(64):
        for i, t, path in zip(b_idx, b_t, b_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            read += 1
            g = cv2.cvtColor(cv2.resize(img, (320, 180)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                samples.append((int(i), float(t), prev, g))
            prev = g

    if not samples:
        raise SkipPass("could not read frames for motion")

    per_shot: dict = {}
    f_idx, f_energy, f_drift, f_zoom = [], [], [], []
    f_label = []
    for idx_f, t, a, b in samples:
        idx = job.shot_at(t)
        pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01,
                                      minDistance=8)
        entry = per_shot.setdefault(idx, {"dx": [], "dy": [], "scale": [],
                                          "resid": [], "energy": []})
        energy = float(np.abs(
            b.astype(float) - a.astype(float)).mean() / 255.0)
        entry["energy"].append(energy)
        f_idx.append(idx_f)
        f_energy.append(energy)

        dx = dy = 0.0
        scale = 1.0
        resid = 1.0
        ok = False
        if pts is not None and len(pts) >= 8:
            nxt, status, _err = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
            if nxt is not None:
                good_a = pts[status.flatten() == 1]
                good_b = nxt[status.flatten() == 1]
                if len(good_a) >= 8:
                    matrix, inliers = cv2.estimateAffinePartial2D(
                        good_a, good_b, method=cv2.RANSAC,
                        ransacReprojThreshold=3)
                    if matrix is not None:
                        dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
                        scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
                        share = (float(inliers.mean())
                                 if inliers is not None else 0.0)
                        resid = 1.0 - share
                        ok = True
        f_drift.append(math.hypot(dx, dy))
        f_zoom.append(scale - 1.0)
        f_label.append((idx_f, t, _move_label(dx, dy, scale - 1.0, 0.0, resid)
                        if ok else None))
        if ok:
            entry["dx"].append(dx)
            entry["dy"].append(dy)
            entry["scale"].append(scale)
            entry["resid"].append(resid)

    em = Emission()
    em.frame_metric("motion_energy", f_idx, f_energy)
    em.frame_metric("camera_drift", f_idx, f_drift)
    em.frame_metric("camera_zoom", f_idx, f_zoom)
    em.frame_runs("style", "camera_move", _smooth_labels(f_label),
                  confidence=0.6)
    labels = []
    for idx in sorted(per_shot):
        e = per_shot[idx]
        energy = float(np.mean(e["energy"])) if e["energy"] else 0.0
        em.claim("style", "motion_energy", num=round(energy, 4), shot_idx=idx)
        if not e["dx"]:
            continue
        dx, dy = float(np.mean(e["dx"])), float(np.mean(e["dy"]))
        drift = math.hypot(dx, dy)
        zoom = float(np.mean(e["scale"])) - 1.0
        jitter = float(np.std(e["dx"]) + np.std(e["dy"]))
        resid = float(np.mean(e["resid"]))

        label = _move_label(dx, dy, zoom, jitter, resid)
        labels.append(label)
        em.claim("style", "camera_move", label, shot_idx=idx,
                 num=round(drift, 3),
                 confidence=round(max(0.4, 1.0 - resid), 3))
        em.claim("style", "stability", num=round(1.0 / (1.0 + jitter), 4),
                 shot_idx=idx)

    if labels:
        common = max(set(labels), key=labels.count)
        em.claim("style", "camera_move", common,
                 num=labels.count(common) / len(labels))
    em.notes = {"shots_measured": len(per_shot), "samples": len(samples),
                "frames": read, "frames_available": job.frame_count() or read}
    return em


def _move_label(dx: float, dy: float, zoom: float, jitter: float,
                resid: float) -> str:
    """Camera move from affine parameters, in one place for both granularities.

    Shared so the per-frame run and the per-shot claim cannot drift apart: two
    copies of these thresholds would eventually disagree, and a database where
    the frame says "pan left" while the shot says "static" is worse than one
    that only had the shot.

    The order of the tests is the meaning. Zoom is checked first because a push
    in also translates slightly and would otherwise read as a pan; jitter is
    checked last because handheld is the residual category, what is left when
    there is movement without direction.
    """
    drift = math.hypot(dx, dy)
    if abs(zoom) > 0.004 and abs(zoom) > drift / 160.0:
        return "push in" if zoom > 0 else "pull out"
    if drift > 1.2 and jitter < drift * 1.6:
        return ("pan right" if dx < -0.8 else "pan left" if dx > 0.8 else
                "tilt down" if dy < 0 else "tilt up")
    if jitter > 2.2 or resid > 0.45:
        return "handheld"
    return "static"


def _smooth_labels(readings: list, window: int = 5) -> list:
    """Majority-vote a per-frame label series over a small window.

    Frame-to-frame affine fits are noisy: a single frame of "handheld" inside a
    steady pan is a fitting artefact, not a camera move, and left alone it would
    split one run-length claim into three. The window is odd and short, so a
    genuine move that lasts a fifth of a second still survives.
    """
    n = len(readings)
    if n < window:
        return readings
    half = window // 2
    out = []
    for i, (idx, t, _value) in enumerate(readings):
        near = [v for _, _, v in readings[max(0, i - half): i + half + 1]
                if v is not None]
        out.append((idx, t, max(set(near), key=near.count) if near else None))
    return out


# ══════════════════════════════════════════════════════════════════════════
# loudness
# ══════════════════════════════════════════════════════════════════════════

def loudness(job: Job) -> Emission:
    """R128 from ffmpeg, plus an RMS curve this pipeline can reason over.

    The integrated figure is what a platform normalises against; the curve at
    100 ms resolution is what tells you the reel opens on a beat drop, holds a
    beat of silence before the punchline, or ducks the music under a voice.
    The hook pass reads this curve, so its resolution is a design decision, not
    a default.
    """
    wav_path = job.artifact("audio.wav")
    np = _np()

    try:
        r128 = media.loudness(wav_path)
    except media.MediaError as exc:
        raise SkipPass(str(exc)) from None

    em = Emission()
    for key, kind, label in (
            ("integrated_lufs", "lufs", "LUFS"),
            ("range_lu", "dynamic_range", "LU"),
            ("true_peak_dbfs", "true_peak", "dBFS")):
        if r128.get(key) is not None:
            em.claim("audio", kind, f"{r128[key]:.1f} {label}", num=r128[key])

    try:
        import soundfile as sf  # noqa: PLC0415
        data, rate = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception:
        # soundfile is the fast path; a wav we wrote ourselves is plain PCM and
        # the stdlib can read it, so a missing dependency costs speed, not the
        # pass.
        import wave  # noqa: PLC0415
        with wave.open(wav_path, "rb") as fh:
            rate = fh.getframerate()
            raw = fh.readframes(fh.getnframes())
        data = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0

    if data.ndim > 1:
        data = data.mean(axis=1)
    hop = max(int(rate * 0.1), 1)
    usable = len(data) - (len(data) % hop)
    if usable < hop:
        raise SkipPass("audio too short to measure")
    windows = data[:usable].reshape(-1, hop)
    rms = np.sqrt((windows ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)

    floor = float(np.percentile(db, 15)) - 6.0
    silent = float((db < max(floor, -60.0)).mean())
    em.claim("audio", "silence_ratio",
             f"{silent * 100:.0f}% near-silent", num=round(silent, 4))
    em.claim("audio", "loudness_curve",
             [round(float(x), 2) for x in db], num=float(db.mean()))

    # The same curve again, as numbers on the shared time index. The JSON claim
    # above stays because the hook pass reads it as a whole; this is what makes
    # the curve *queryable* — "find the frame where the audio drops 12 dB" is a
    # comparison over this array, and asking it of a JSON blob would mean
    # parsing every video's claim to answer it once. Written only when frames
    # exist: a metric keyed by 100 ms slot and one keyed by frame are different
    # namespaces, and silently mixing them would corrupt every later reader.
    frames = [job.frame_at(i * 0.1) for i in range(len(db))]
    if all(f is not None for f in frames):
        em.frame_metric("loudness_db", frames,
                        [round(float(x), 2) for x in db])
        # Silence as a run rather than a ratio: where the reel actually holds a
        # beat before the punchline, addressable to the frame.
        quiet = max(floor, -60.0)
        em.window_runs(
            "audio", "silence",
            [(i, i * 0.1, i * 0.1 + 0.1,
              "near-silent" if float(db[i]) < quiet else None)
             for i in range(len(db))],
            confidence=0.9, frame_of=job.frame_at)

    # Per-shot mean level, so "the loudest shot" is a query and not a listen.
    for s in job.shots():
        a = int(float(s["t0"]) * 10)
        b = max(int(float(s["t1"]) * 10), a + 1)
        seg = db[a:b]
        if len(seg):
            em.claim("audio", "shot_level", num=round(float(seg.mean()), 2),
                     shot_idx=int(s["idx"]))

    em.notes = {**r128, "silence_ratio": round(silent, 4),
                "windows": int(len(db))}
    return em


# ══════════════════════════════════════════════════════════════════════════
# music — tempo, beats, key
# ══════════════════════════════════════════════════════════════════════════

_PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl–Kessler profiles: how strongly each scale degree is expected in a
# major and a minor key. Correlating a chroma vector against all 24 rotations
# is the standard key-finding method and it is a dot product, not a model.
_KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def music(job: Job) -> Emission:
    """Tempo, beat grid, key, and the harmonic/percussive balance.

    Knowing where the beats are turns "the cut lands on the downbeat" from a
    stylistic impression into a checkable statement — the cuts pass has the cut
    times, this pass has the beat times, and the comparison is subtraction.
    """
    wav_path = job.artifact("audio.wav")
    np = _np()
    try:
        import librosa  # noqa: PLC0415
    except ImportError:
        raise SkipPass("librosa is not installed") from None

    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    if len(y) < sr:
        raise SkipPass("audio shorter than one second")

    job.heartbeat("beat tracking")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.sum(harmonic ** 2))
    p_energy = float(np.sum(percussive ** 2))

    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr).mean(axis=1)
    chroma = chroma / (chroma.sum() + 1e-9)
    best, best_score = ("", ""), -2.0
    for root in range(12):
        rolled = np.roll(chroma, -root)
        for name, profile in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
            p = np.asarray(profile, dtype=float)
            score = float(np.corrcoef(rolled, p / p.sum())[0, 1])
            if score > best_score:
                best_score, best = score, (_PITCHES[root], name)

    em = Emission()
    em.claim("audio", "tempo", f"{tempo:.0f} BPM", num=round(tempo, 2))
    em.claim("audio", "key", f"{best[0]} {best[1]}",
             confidence=round(max(min((best_score + 1) / 2, 1.0), 0.0), 3))
    em.claim("audio", "harmonic_ratio",
             num=round(h_energy / (h_energy + p_energy + 1e-9), 4))
    em.claim("audio", "beat_grid", [round(float(b), 3) for b in beats],
             num=len(beats))

    # Do the cuts land on the beat? A number, computed here, once.
    shots = job.shots()
    if len(shots) > 2 and len(beats) > 2:
        beat_times = np.asarray(beats, dtype=float)
        period = float(np.median(np.diff(beat_times))) if len(beats) > 2 else 0.5
        offsets = [float(np.min(np.abs(beat_times - float(s["t0"]))))
                   for s in shots[1:]]
        on_beat = float(np.mean([o < period * 0.18 for o in offsets]))
        em.claim("style", "cut_on_beat",
                 f"{on_beat * 100:.0f}% of cuts land on a beat",
                 num=round(on_beat, 4))

    em.notes = {"tempo": round(tempo, 2), "key": " ".join(best),
                "beats": int(len(beats))}
    return em


# ══════════════════════════════════════════════════════════════════════════
# tag — zero-shot labels by cosine, not by prompt
# ══════════════════════════════════════════════════════════════════════════

# The vocabulary. Deliberately concrete and deliberately extensible: adding a
# label re-tags the whole archive in seconds, because the expensive half —
# embedding five thousand videos — is already done and cached.
LABELS = (
    # setting
    "indoors", "outdoors", "a kitchen", "an office", "a bedroom", "a gym",
    "a street", "a car interior", "a stage", "a beach", "a studio backdrop",
    # format
    "a person talking to camera", "a screen recording", "text on a plain "
    "background", "a product close-up", "a whiteboard explanation",
    "a split screen", "an interview", "b-roll footage", "a phone screen",
    "a slideshow of photos", "an animation", "a chart or graph",
    # subject
    "food being prepared", "a workout", "a computer or laptop", "a smartphone",
    "handwriting", "a crowd of people", "a pet", "a vehicle", "money or cash",
    "a book", "a camera or lens", "plants or nature",
    # treatment
    "high contrast lighting", "soft natural light", "neon lighting",
    "a night scene", "a close-up of a face", "a wide establishing shot",
    "a top-down flat lay", "a moving vehicle shot",
)


def tag(job: Job) -> Emission:
    """Label the frames by cosine similarity against embedded label text.

    Nothing is prompted and nothing is generated, so nothing can be
    hallucinated: the answer is a matrix multiply between vectors that already
    exist and a label matrix computed once. The cost of being wrong is bounded
    too — a bad label is a bad threshold, visible and fixable, not a sentence
    that has to be read to be doubted.
    """
    np = _np()
    rows = job.store.vectors_for(job.key, "siglip2")
    if not rows:
        raise SkipPass("no visual vectors — run visual-embed first")

    from .vision import label_matrix  # noqa: PLC0415 — needs the text tower
    matrix, names = label_matrix(job, LABELS)
    if matrix is None:
        raise SkipPass("the visual model is not loaded, so labels cannot be "
                       "embedded")

    top_k = int(job.params.get("top_k", 12))
    floor = float(job.params.get("min_similarity", 0.22))
    em = Emission()
    totals = np.zeros(len(names), dtype="float32")

    for r in rows:
        v = np.asarray(r["values"], dtype="float32")
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = matrix @ v
        totals += sims
        if r.get("shot_idx") is None:
            continue
        order = np.argsort(-sims)[:max(top_k // 3, 3)]
        for rank, i in enumerate(order):
            if float(sims[i]) < floor:
                break
            em.claim("visual", "tag", names[i], shot_idx=int(r["shot_idx"]),
                     num=round(float(sims[i]), 4),
                     confidence=round(float(sims[i]), 4), ordinal=rank)

    totals /= max(len(rows), 1)
    order = np.argsort(-totals)[:top_k]
    for rank, i in enumerate(order):
        if float(totals[i]) < floor:
            break
        em.claim("visual", "tag", names[i], num=round(float(totals[i]), 4),
                 confidence=round(float(totals[i]), 4), ordinal=rank)

    em.notes = {"vectors": len(rows), "labels": len(names)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# keyphrase — the control group for concept extraction
# ══════════════════════════════════════════════════════════════════════════

_STOP = set("""a an the and or but if then than that this these those of in on
at to for with from by as is are was were be been being it its it's i you he
she they we me him her them us my your his their our so just very really about
into over under out up down not no yes do does did done can could would should
will shall may might must have has had here there what when where who whom how
why which while all any each more most other some such only own same too s t
don now ll re ve m d""".split())


def _ngrams(words, n):
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def keyphrase(job: Job) -> Emission:
    """YAKE where available, and a transparent fallback where it is not.

    Both are statistical and neither can produce a phrase absent from the
    text. That is the entire point: they are the control against which the
    language model's concept list is judged, and a control that could
    hallucinate would be no control at all.
    """
    bundle = job.text_bundle()
    text = " ".join(filter(None, [
        bundle["transcript"], bundle["caption"],
        " ".join(bundle["on_screen"])])).strip()
    if len(text) < 40:
        raise SkipPass("not enough text to extract phrases from")

    em = Emission()
    got = []
    try:
        import yake  # noqa: PLC0415
        extractor = yake.KeywordExtractor(n=3, dedupLim=0.75, top=25)
        # YAKE scores are distances: lower is better. Inverted here so the
        # claim's `num` sorts the same way as every other confidence in the
        # database — a mixed convention is a bug waiting for whoever writes the
        # first ORDER BY.
        got = [(kw, 1.0 / (1.0 + score)) for kw, score in extractor.extract_keywords(text)]
    except Exception:
        words = [w for w in re.findall(r"[\w'ऀ-ॿ]+", text.lower())
                 if w not in _STOP and len(w) > 2]
        freq: dict = {}
        for n in (1, 2, 3):
            for g in _ngrams(words, n):
                freq[g] = freq.get(g, 0) + n            # longer phrases weigh more
        total = max(sum(freq.values()), 1)
        got = sorted(((g, c / total) for g, c in freq.items() if c > 1),
                     key=lambda x: -x[1])[:25]

    for rank, (phrase, score) in enumerate(got):
        em.claim("concept", "keyphrase", phrase, num=round(float(score), 5),
                 confidence=round(min(float(score) * 4, 1.0), 3), ordinal=rank)
    em.notes = {"phrases": len(got), "chars": len(text)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# hook — the first three seconds, assembled
# ══════════════════════════════════════════════════════════════════════════

def hook(job: Job) -> Emission:
    """What happens in the window where retention is decided.

    Assembly, not inference. Every number here already exists somewhere in the
    database; this pass gathers the ones that fall inside the window and writes
    them as a single addressable object, so "show me every reel that opens on
    a question with two cuts in the first three seconds" is one query instead
    of a join nobody will write twice.

    Because it is assembly, it is nearly free to recompute — which matters,
    since the right length for "the hook" is an empirical question this archive
    exists to answer, and the answer will change the window.
    """
    window = float(job.params.get("window_seconds", 3.0))
    shots = job.shots()
    if not shots:
        raise SkipPass("no shots")

    opening = [s for s in shots if float(s["t0"]) < window]
    idx = {int(s["idx"]) for s in opening}

    said = [c["value"] for c in job.claims("speech", "segment")
            if c.get("t0") is not None and float(c["t0"]) < window and c.get("value")]
    written = [c["value"] for c in job.claims("ocr", "text")
               if c.get("shot_idx") in idx and c.get("value")]
    seen = [c["value"] for c in job.claims("visual", "shot_description")
            if c.get("shot_idx") in idx and c.get("value")]
    levels = [c["num"] for c in job.claims("audio", "shot_level")
              if c.get("shot_idx") in idx and c.get("num") is not None]

    words = " ".join(said).split()
    em = Emission()
    em.claim("narrative", "hook_cuts", f"{len(opening)} shots in the first "
             f"{window:g}s", num=len(opening))
    if words:
        em.claim("narrative", "hook_words", " ".join(words))
        em.claim("narrative", "hook_word_rate",
                 num=round(len(words) / window, 2))
        first = " ".join(words[:14])
        em.claim("narrative", "hook_opening_line", first)
        em.claim("narrative", "hook_form",
                 "question" if "?" in " ".join(said) else
                 "command" if words and words[0].lower() in {
                     "stop", "watch", "look", "listen", "try", "do", "never",
                     "always", "don't", "dont"} else "statement")
    if written:
        em.claim("narrative", "hook_text", " · ".join(written[:5]))
    if seen:
        em.claim("narrative", "hook_visual", seen[0])
    if levels:
        em.claim("narrative", "hook_loudness",
                 num=round(sum(levels) / len(levels), 2))

    em.notes = {"window": window, "shots": len(opening),
                "words": len(words), "on_screen": len(written)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# perframe
# ══════════════════════════════════════════════════════════════════════════

def perframe(job: Job) -> Emission:
    """Measure every extracted frame, not one per shot.

    The other pixel passes in this module read `job.frames()`, which is the
    keyframe set: one image per shot, chosen for sharpness. That is the correct
    input for a shot-level claim and it is a sample. This pass reads the
    `allframes` manifest instead and measures all of it, so the archive can say
    what was happening at any moment rather than at forty moments.

    **The output does not go in the claims table, and that is deliberate.** A
    900-frame reel measured on six metrics is 5,400 claims; across five
    thousand videos it is twenty-seven million rows carrying numbers nobody
    queries individually. What people query is the summary — "reels that get
    dark in the last third", "shots with a freeze frame" — and what people plot
    is the whole curve at once. So the per-frame numbers go in as columnar
    arrays, one packed row per metric, which is two orders of magnitude smaller
    than a row per frame and the shape a plot actually wants.

    They now go into the **database** as `frame_metric` rows as well as into
    `perframe.json`. The JSON was written next to frames that are deleted with
    the workdir, so every number this pass computed died with the session and
    none of it reached v17, Atlas or a shard. The columnar shape is what made
    that fixable without the 27 million rows: same arrays, durable table.

    `phash` earns its place by answering a question nothing else here can: a
    run of identical hashes is a freeze frame, and a freeze frame is the
    difference between a video that ended and a video that broke. It also makes
    near-duplicate reels findable across the archive without a model.
    """
    import cv2  # noqa: PLC0415
    np = _np()

    mpath = job.path("allframes/manifest.json")
    if not os.path.exists(mpath):
        raise SkipPass("no allframes manifest — run the allframes pass first")
    with open(mpath, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    entries = manifest.get("frames") or []
    if not entries:
        raise SkipPass("allframes manifest is empty")

    root = job.path("allframes")
    cols = {k: [] for k in ("t", "brightness", "contrast", "saturation",
                            "temperature", "sharpness", "motion")}
    idxs, hashes, black, unreadable = [], [], 0, 0
    prev_small = None

    for n, e in enumerate(entries):
        if n % 200 == 0:
            job.heartbeat(f"frame {n}/{len(entries)}")
        img = cv2.imread(os.path.join(root, e["file"]))
        if img is None:
            unreadable += 1
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        b, g, r = (img[:, :, i].astype(np.float32).mean() for i in range(3))
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bright = float(v.mean())
        if bright < 0.02:
            black += 1

        # Motion as mean absolute difference against the previous frame, on a
        # 64x64 reduction. At full size this is the single most expensive line
        # in the pass and the extra pixels change the number in the third
        # decimal — the curve is what matters, not its precision.
        small = cv2.resize(grey, (64, 64), interpolation=cv2.INTER_AREA)
        if prev_small is None:
            motion_v = 0.0
        else:
            motion_v = float(np.abs(small.astype(np.float32)
                                    - prev_small.astype(np.float32)).mean() / 255.0)
        prev_small = small

        cols["t"].append(e["t"])
        idxs.append(int(e.get("i", n)))
        cols["brightness"].append(round(bright, 4))
        cols["contrast"].append(round(float(v.std()), 4))
        cols["saturation"].append(round(float(s.mean()), 4))
        cols["temperature"].append(round(float((r - b) / 255.0), 4))
        cols["sharpness"].append(round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 3))
        cols["motion"].append(round(motion_v, 5))
        hashes.append(_phash(grey, cv2, np))

    measured = len(cols["t"])
    if not measured:
        raise SkipPass("no readable frames in the allframes set")

    # Freeze runs: consecutive frames whose hash is identical. One repeat is a
    # static shot, which is ordinary; a run past a second is a stall.
    freezes, run_start, longest = [], 0, 0
    for i in range(1, len(hashes)):
        if hashes[i] != hashes[i - 1]:
            span = i - run_start
            if span > 1:
                secs = cols["t"][i - 1] - cols["t"][run_start]
                if secs >= 1.0:
                    freezes.append({"from": cols["t"][run_start],
                                    "to": cols["t"][i - 1],
                                    "seconds": round(secs, 3),
                                    "frames": span})
                longest = max(longest, span)
            run_start = i

    with open(job.path("allframes/perframe.json"), "w", encoding="utf-8") as fh:
        json.dump({"count": measured, "columns": cols, "phash": hashes,
                   "freezes": freezes}, fh, separators=(",", ":"))

    em = Emission()
    em.artifact("perframe", job.path("allframes/perframe.json"),
                {"frames": measured, "metrics": len(cols) - 1})

    # The curves, into the database. `t` is not among them: time is derived
    # from the frame index everywhere else in this system and storing a second
    # copy here would be the one place it could disagree.
    for name in ("brightness", "contrast", "saturation", "temperature",
                 "sharpness", "motion"):
        em.frame_metric(name, idxs, cols[name])

    # Freeze spans as run-length claims, so "where did it stall" is a query
    # rather than a file that no longer exists.
    if hashes:
        em.frame_runs("style", "freeze",
                      [(idxs[i], cols["t"][i],
                        "freeze" if (i and hashes[i] == hashes[i - 1]) else None)
                       for i in range(len(hashes))], confidence=0.9)

    for name in ("brightness", "contrast", "saturation", "temperature",
                 "sharpness", "motion"):
        vals = cols[name]
        em.claim("style", f"{name}_mean", num=round(sum(vals) / len(vals), 4))
        em.claim("style", f"{name}_min", num=round(min(vals), 4))
        em.claim("style", f"{name}_max", num=round(max(vals), 4))

    em.claim("style", "black_frames", num=black)
    em.claim("style", "freeze_spans", num=len(freezes))
    if freezes:
        em.claim("style", "longest_freeze",
                 num=round(max(f["seconds"] for f in freezes), 3))
    em.claim("style", "unique_frames",
             num=round(len(set(hashes)) / len(hashes), 4))

    em.notes = {
        "measured": measured,
        "of": len(entries),
        "unreadable": unreadable,
        "complete": measured == len(entries),
        "black_frames": black,
        "freeze_spans": len(freezes),
        "distinct_ratio": round(len(set(hashes)) / len(hashes), 4),
    }
    return em


def _phash(grey, cv2, np) -> str:
    """A 64-bit perceptual hash, as 16 hex characters.

    DCT of a 32x32 reduction, low-frequency 8x8 block, thresholded at its own
    median with the DC term excluded — the DC term is overall brightness, and
    leaving it in makes every hash track exposure instead of content.
    """
    small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(small.astype(np.float32))[:8, :8].flatten()
    med = np.median(d[1:])
    bits = 0
    for i, val in enumerate(d):
        if val > med:
            bits |= (1 << i)
    return f"{bits:016x}"
