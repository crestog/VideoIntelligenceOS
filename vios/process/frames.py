"""
vios.process.frames — every frame, and the proof that it is every frame.

The keyframes pass in `runners/structure.py` extracts one frame per shot. That
is a *sample*, and a sample is the right input for a shot-level summary. It is
the wrong input for the claim "this archive has seen every frame of every
video", which is the claim this module exists to make true and, more
importantly, to make checkable.

Three things have to be right, and only the first is obvious.

**Extract without resampling.** ffmpeg's default `-fps_mode` is `cfr`: it
duplicates and drops frames to force a constant output rate. Point it at a
variable-rate reel and it will silently emit a different number of frames than
the file contains — usually more, occasionally fewer — and every one of them
carries a timestamp that was invented rather than read. `passthrough` turns
that off.

Measured on a 110-frame test file with a deliberate 1.4-second hole: the
default emitted 150 frames, forty of them duplicates manufactured to fill the
gap. That is not merely a wrong count. The fabricated frames *cover the hole*,
so the coverage check below would have passed the file as complete. The flag is
what stops this module from certifying its own blind spot.

**Timestamp from the file, not from arithmetic.** `index / fps` is the natural
way to date a frame and it is wrong on most Instagram reels. Phone cameras,
screen recordings and anything that has been through a social pipeline are
variable frame rate: the container declares 30 fps, the real gaps between
frames run from 20 ms to 90 ms, and by the end of a 60-second clip the
arithmetic timestamp has drifted seconds away from the real one. Every frame is
still extracted, but each is labelled with a moment it did not come from, and
any model output keyed to those labels is misaligned with the audio. So the
presentation timestamp of every frame is read out of the stream with ffprobe
and used as the truth.

**Distinguish "all the frames" from "all the video".** Those are different
claims and only one of them is usually in doubt. A file can hand over every
frame it contains and still leave a four-second hole where the encoder dropped
everything, and a frame count that matches the header proves nothing about that
hole. So the report answers both separately: whether the extracted count matches
what the stream declares, and whether every wall-clock second of the duration
contains at least one frame. The second question is the one that catches real
corruption.

`vfps` is the measured frame rate — extracted frames over real duration — as
opposed to `fps`, which is whatever the container claims. When they disagree by
more than a rounding error the file is variable-rate, and the report says so
rather than quietly preferring one.

Nothing here decides *what* to analyse. It produces the frame set and a
manifest with an honest timestamp per frame; the runners consume that.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import subprocess

from . import media

# The analysis tier's width. 384 px covers what the vision stack actually
# ingests — CLIP and SigLIP want 224, Florence-2 and the captioners want 384 —
# so anything larger is decoded, resized and thrown away by every consumer.
ANALYSIS_WIDTH = 384

# JPEG quality, ffmpeg's scale where 2 is near-lossless and 31 is unusable.
# 6 on a 384 px frame is ~8 KB and visually clean; 3 doubles the bytes for
# detail no model at this resolution can see.
ANALYSIS_QUALITY = 6
FULL_QUALITY = 3

# A second holding no frame at all. Anything longer than this between two
# consecutive frames is a hole in the video, not a slow patch.
GAP_SECONDS = 1.0

_TS = re.compile(r"^([0-9]*\.?[0-9]+)")


class FrameError(RuntimeError):
    """Extraction failed, or produced something that cannot be trusted."""


# ══════════════════════════════════════════════════════════════════════════
# Rate facts
# ══════════════════════════════════════════════════════════════════════════

def _fps_mode_flag() -> list:
    """`-fps_mode passthrough`, or `-vsync 0` on ffmpeg older than 5.1.

    Both mean "give me exactly the frames that are in the file". The old spelling
    still works in ffmpeg 7 but warns; the new one does not exist before 5.1 and
    is a hard error there. Kaggle's image and a 2019 laptop are both in scope,
    so the version is asked rather than assumed.
    """
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                             text=True, timeout=30).stdout
        m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", out)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            if (major, minor) >= (5, 1):
                return ["-fps_mode", "passthrough"]
    except (OSError, subprocess.SubprocessError):
        pass
    return ["-vsync", "0"]


def frame_times(src: str, timeout: int = 900) -> list:
    """The presentation timestamp of every video frame, in order.

    This is the ground truth the whole module rests on. `pts_time` was called
    `pkt_pts_time` before ffmpeg 5 and both spellings are requested, because a
    manifest built on the wrong one comes back as a list of empty strings and
    the failure looks like a video with no frames.

    Frames whose timestamp is unreadable are kept in the list as `None` rather
    than dropped: losing them would make the count disagree with the extracted
    files, and a count mismatch is the signal this module is built to raise.
    """
    out = media._run(
        ["ffprobe", "-hide_banner", "-loglevel", "error",
         "-select_streams", "v:0", "-show_entries",
         "frame=pts_time,pkt_pts_time", "-of", "csv=p=0", src],
        timeout=timeout)
    times: list = []
    for line in out.splitlines():
        line = line.strip().strip(",")
        if not line:
            continue
        m = _TS.match(line)
        times.append(round(float(m.group(1)), 6) if m else None)
    return times


def declared_count(src: str) -> int:
    """`nb_frames` from the stream header, or 0 when the container omits it.

    Reels remuxed by Instagram frequently have no `nb_frames` at all, so a zero
    here is normal and means "the file declines to say", not "the file is
    empty". The report treats it that way.
    """
    try:
        out = media._run(
            ["ffprobe", "-hide_banner", "-loglevel", "error",
             "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
             "-of", "default=nk=1:nw=1", src], timeout=120)
        return int((out or "0").strip() or 0)
    except (media.MediaError, ValueError):
        return 0


def rates(src: str, meta: dict | None = None) -> dict:
    """fps as declared, vfps as measured, and whether they agree.

    `vfps` is computed from the real timestamp list, so it costs a full stream
    parse. That is the point: the declared rate is exactly the number that
    cannot be trusted on the files this system ingests.
    """
    meta = meta or media.probe(src)
    times = [t for t in frame_times(src) if t is not None]
    duration = float(meta.get("duration") or 0)

    # Span between first and last frame, not the container duration: a file
    # with three seconds of trailing audio would otherwise report a frame rate
    # a tenth below its real one.
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    vfps = (len(times) - 1) / span if span > 0 else 0.0

    fps = float(meta.get("fps") or 0)
    drift = abs(vfps - fps) / fps if fps and vfps else 0.0

    # Real gaps, for the variable-rate call. A CFR file's deltas are identical
    # to the microsecond; anything above a few percent of spread is variable.
    deltas = [round(b - a, 6) for a, b in zip(times, times[1:])] if len(times) > 1 else []
    spread = 0.0
    if deltas:
        lo, hi = min(deltas), max(deltas)
        spread = (hi - lo) / hi if hi else 0.0

    return {
        "fps": round(fps, 4),
        "vfps": round(vfps, 4),
        "frames_seen": len(times),
        "frames_declared": declared_count(src),
        "duration": round(duration, 3),
        "span": round(span, 3),
        "drift": round(drift, 4),
        "variable": bool(spread > 0.02 or drift > 0.02),
        "spread": round(spread, 4),
        "min_delta": round(min(deltas), 6) if deltas else 0.0,
        "max_delta": round(max(deltas), 6) if deltas else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════
# Extraction
# ══════════════════════════════════════════════════════════════════════════

def extract_all(src: str, outdir: str, analysis_width: int = ANALYSIS_WIDTH,
                full: bool = False, full_width: int = 0,
                timeout: int = 3600) -> dict:
    """Every frame of `src`, written to `outdir`. Returns the manifest.

    One ffmpeg invocation. When the full tier is requested the filter graph
    splits after a single decode, so the second tier costs an encode and not
    another pass over the file — on a 60-second 1080p reel that is the
    difference between 25 seconds and 45.

    The full tier is off by default and that is a storage decision, not a
    quality one: 1800 frames at 1080p is ~200 MB per reel, and across a
    five-thousand-reel archive it is a terabyte that no model reads. The
    analysis tier is sized for what the vision stack actually ingests. Turn
    `full` on for the handful of videos worth keeping at source resolution.
    """
    os.makedirs(outdir, exist_ok=True)
    adir = os.path.join(outdir, "analysis")
    os.makedirs(adir, exist_ok=True)

    passthrough = _fps_mode_flag()
    apat = os.path.join(adir, "f_%06d.jpg")

    if full:
        fdir = os.path.join(outdir, "full")
        os.makedirs(fdir, exist_ok=True)
        scale = (f"scale={int(full_width)}:-2," if full_width else "")
        cmd = ["ffmpeg", *media._FF, "-y", "-i", src,
               "-filter_complex",
               f"[0:v]split=2[a][b];"
               f"[a]{scale}null[fullout];"
               f"[b]scale={int(analysis_width)}:-2[anaout]",
               "-map", "[fullout]", *passthrough,
               "-q:v", str(FULL_QUALITY), "-start_number", "0",
               os.path.join(fdir, "f_%06d.jpg"),
               "-map", "[anaout]", *passthrough,
               "-q:v", str(ANALYSIS_QUALITY), "-start_number", "0", apat]
    else:
        cmd = ["ffmpeg", *media._FF, "-y", "-i", src,
               "-vf", f"scale={int(analysis_width)}:-2", *passthrough,
               "-q:v", str(ANALYSIS_QUALITY), "-start_number", "0", apat]

    media._run(cmd, timeout=timeout)

    written = sorted(glob.glob(os.path.join(adir, "f_*.jpg")))
    if not written:
        raise FrameError("ffmpeg wrote no frames — the video stream is "
                         "unreadable or the codec is unsupported")

    meta = media.probe(src)
    times = frame_times(src)
    rate = rates(src, meta)

    # ffmpeg numbers output files from the frames it emitted; `passthrough`
    # makes that one file per real frame, so position n in the sorted file list
    # is position n in the timestamp list. When the two lengths disagree the
    # pairing is no longer safe, and `verify` is what reports it — the manifest
    # still pairs what it can rather than throwing away a good 99%.
    entries = []
    for i, path in enumerate(written):
        t = times[i] if i < len(times) else None
        if t is None and rate["vfps"]:
            t = round(i / rate["vfps"], 6)      # last resort, and flagged below
        entries.append({
            "i": i,
            "t": t,
            "file": os.path.relpath(path, outdir).replace(os.sep, "/"),
            "exact": i < len(times) and times[i] is not None,
        })

    manifest = {
        "source": os.path.basename(src),
        "outdir": outdir,
        "analysis_width": analysis_width,
        "full_tier": bool(full),
        "rates": rate,
        "meta": {k: meta.get(k) for k in
                 ("duration", "width", "height", "rotation", "fps",
                  "video_codec", "vertical")},
        "count": len(entries),
        "frames": entries,
    }

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, separators=(",", ":"))
    return manifest


# ══════════════════════════════════════════════════════════════════════════
# The proof
# ══════════════════════════════════════════════════════════════════════════

def verify(manifest: dict, outdir: str | None = None) -> dict:
    """Check the claim, and say exactly which part fails when it does.

    Four independent checks, because they fail for different reasons and a
    single pass/fail would hide which one tripped:

      files      every frame in the manifest exists and is non-empty on disk
      count      the extracted count agrees with what the stream declared
      timing     every frame carries a timestamp read from the file
      coverage   every wall-clock second of the duration contains a frame

    `coverage` is the one that matters and the one a frame count cannot
    substitute for. A file can hand over exactly the number of frames its
    header promises and still have a hole in the middle where the encoder gave
    up; only walking the timeline second by second finds it.
    """
    outdir = outdir or manifest.get("outdir") or "."
    frames = manifest.get("frames") or []
    rate = manifest.get("rates") or {}
    duration = float(rate.get("duration") or 0)

    missing, empty = [], []
    for f in frames:
        p = os.path.join(outdir, f["file"])
        try:
            if os.path.getsize(p) <= 0:
                empty.append(f["i"])
        except OSError:
            missing.append(f["i"])

    declared = int(rate.get("frames_declared") or 0)
    seen = len(frames)
    count_ok = (declared == 0) or (declared == seen)

    inexact = [f["i"] for f in frames if not f.get("exact")]

    # Second-by-second occupancy. `buckets[n]` is how many frames landed in
    # second n; a zero is a hole with a name and a timestamp.
    stamped = sorted(f["t"] for f in frames if f.get("t") is not None)
    span_end = duration or (stamped[-1] if stamped else 0.0)
    nbuckets = max(int(math.ceil(span_end)), 0)
    buckets = [0] * nbuckets
    for t in stamped:
        b = int(t)
        if 0 <= b < nbuckets:
            buckets[b] += 1
    empty_seconds = [i for i, n in enumerate(buckets) if n == 0]

    # The largest hole between two consecutive frames, which is the same
    # information as `empty_seconds` but survives a video shorter than a second.
    gaps = []
    for a, b in zip(stamped, stamped[1:]):
        d = b - a
        if d > GAP_SECONDS:
            gaps.append({"from": round(a, 3), "to": round(b, 3),
                         "seconds": round(d, 3)})
    lead = round(stamped[0], 3) if stamped else 0.0
    tail = round(span_end - stamped[-1], 3) if stamped else 0.0

    files_ok = not missing and not empty
    timing_ok = not inexact
    coverage_ok = (not empty_seconds and not gaps
                   and lead <= GAP_SECONDS and tail <= GAP_SECONDS)

    checks = {
        "files": {"ok": files_ok, "missing": missing[:20],
                  "empty": empty[:20],
                  "detail": (f"all {seen} frame files present and non-empty"
                             if files_ok else
                             f"{len(missing)} missing, {len(empty)} zero-byte")},
        "count": {"ok": count_ok, "declared": declared, "extracted": seen,
                  "detail": ("stream declares no frame count; "
                             f"{seen} extracted" if declared == 0 else
                             f"{seen} extracted, {declared} declared"
                             + ("" if count_ok else
                                f" — {abs(seen - declared)} adrift"))},
        "timing": {"ok": timing_ok, "interpolated": len(inexact),
                   "detail": (f"all {seen} timestamps read from the stream"
                              if timing_ok else
                              f"{len(inexact)} timestamps interpolated from "
                              f"vfps because the stream did not carry one")},
        "coverage": {"ok": coverage_ok, "seconds": nbuckets,
                     "empty_seconds": empty_seconds[:20],
                     "gaps": gaps[:10], "lead_in": lead, "tail": tail,
                     "detail": (f"every one of {nbuckets} seconds holds at "
                                f"least one frame" if coverage_ok else
                                f"{len(empty_seconds)} empty seconds, "
                                f"{len(gaps)} gaps over {GAP_SECONDS}s")},
    }

    complete = all(c["ok"] for c in checks.values())
    return {
        "complete": complete,
        "checks": checks,
        "frames": seen,
        "duration": round(duration, 3),
        "fps": rate.get("fps"),
        "vfps": rate.get("vfps"),
        "variable": rate.get("variable"),
        "density": round(seen / duration, 2) if duration else 0.0,
        "bytes": _tier_bytes(outdir),
    }


def _tier_bytes(outdir: str) -> dict:
    out = {}
    for tier in ("analysis", "full"):
        d = os.path.join(outdir, tier)
        if not os.path.isdir(d):
            continue
        total = n = 0
        for name in os.listdir(d):
            try:
                total += os.path.getsize(os.path.join(d, name))
                n += 1
            except OSError:
                pass
        out[tier] = {"files": n, "bytes": total,
                     "mb": round(total / 1048576, 2)}
    return out


def report(verdict: dict, name: str = "") -> str:
    """The verdict as text a person can read in a log.

    This is the proof the archive is asked for, so it states the numbers rather
    than a checkmark: a line saying "complete" that cannot be argued with from
    its own contents is not evidence of anything.

    Deliberately ASCII. This goes to a Kaggle log and to a Windows console,
    and `cp1252` cannot encode an em dash — a report that raises
    `UnicodeEncodeError` while printing proof of completeness is worse than no
    report, because the traceback replaces the result.
    """
    v = verdict
    head = "COMPLETE" if v["complete"] else "INCOMPLETE"
    rate = (f"{v['fps']:.3f} fps declared, {v['vfps']:.3f} measured"
            if v.get("fps") else "frame rate unknown")
    if v.get("variable"):
        rate += "  (variable rate - timestamps read per frame)"

    lines = [
        f"[{head}] {name or v.get('source', 'video')}",
        f"  {v['frames']} frames over {v['duration']:.3f}s "
        f"- {v['density']:.2f} frames/second",
        f"  {rate}",
    ]
    for key in ("files", "count", "timing", "coverage"):
        c = v["checks"][key]
        lines.append(f"  {'PASS' if c['ok'] else 'FAIL'}  {key:<9} {c['detail']}")

    cov = v["checks"]["coverage"]
    if cov["empty_seconds"]:
        lines.append(f"        empty seconds: {cov['empty_seconds']}")
    for g in cov["gaps"]:
        lines.append(f"        gap {g['from']:.3f}s -> {g['to']:.3f}s "
                     f"({g['seconds']:.3f}s with no frame)")

    for tier, b in (v.get("bytes") or {}).items():
        lines.append(f"  {tier:<9} {b['files']} files, {b['mb']} MB")
    return "\n".join(lines)


def extract_and_verify(src: str, outdir: str, **kw) -> tuple:
    """Extract every frame and immediately check the claim. Returns
    (manifest, verdict).

    The verdict is written next to the frames as `coverage.json` so the proof
    travels with the evidence — a manifest that says 1800 frames is worth less
    than one accompanied by the check that walked the timeline and found no
    hole. Both are uploaded, so the claim is auditable from the archive alone
    without re-downloading the original.
    """
    manifest = extract_all(src, outdir, **kw)
    verdict = verify(manifest, outdir)
    verdict["source"] = manifest.get("source", "")
    with open(os.path.join(outdir, "coverage.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1)
    return manifest, verdict
