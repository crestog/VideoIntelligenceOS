"""
vios.process.media — ffmpeg and ffprobe, wrapped honestly.

Everything in this module is derived and disposable. If the whole artifact
directory is deleted, one CPU pass regenerates it; if the original mp4 is lost,
nothing regenerates that. So this layer is allowed to be aggressive — overwrite,
re-encode, throw away — in a way the capture plane never is.

No torch, no GPU, no model. The web host imports this to make a poster on
demand, and the web host has neither. Shot *detection* lives in the vision
runner; what lives here is the cheap PySceneDetect prefilter that gives it a
candidate list, because that runs on the proxy at CPU speed.

The artifact set, and why each one exists:

  proxy.mp4       480p H.264 faststart — what the site actually streams. The
                  original is 1080×1920 at 8 Mbps and playing it over a range
                  request from Telegram is the reason playback felt slow.
  poster.jpg      first non-black frame, not frame zero, which on reels is
                  usually a fade from black and looks broken as a thumbnail.
  sprite.jpg      a grid of small frames for scrub preview; one request buys
                  the entire timeline.
  loop.mp4        4 seconds around the most active moment, muted, for hover.
  audio.wav       16 kHz mono — decoded once here so six audio passes do not
                  each decode the original.
  waveform.png    the shape of the sound, for the player's scrubber.
  frames/         one keyframe per shot, written by the keyframes pass.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

# ffmpeg writes progress and warnings to stderr and always has; treating a
# non-empty stderr as failure would fail on every successful run.
_QUIET = ["-hide_banner", "-loglevel", "error"]

# `-nostdin` is an ffmpeg option and ffprobe rejects it outright, which is why
# the two do not share a flag list. It matters more than it looks: without it,
# an ffmpeg launched from a notebook cell inherits the cell's stdin and can sit
# waiting on it forever instead of failing.
_FF = [*_QUIET, "-nostdin"]


class MediaError(RuntimeError):
    """ffmpeg or ffprobe failed. Carries the tail of stderr, which is the only
    part that ever says anything useful."""


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _run(cmd: list, timeout: int = 900) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        raise MediaError(f"{cmd[0]} timed out after {timeout}s") from None
    except OSError as exc:
        raise MediaError(f"{cmd[0]} not runnable: {exc}") from None
    if res.returncode != 0:
        tail = (res.stderr or "").strip().splitlines()[-4:]
        raise MediaError(f"{cmd[0]} exit {res.returncode}: " + " | ".join(tail))
    return res.stdout


# ══════════════════════════════════════════════════════════════════════════
# Probing
# ══════════════════════════════════════════════════════════════════════════

def ffprobe(path: str) -> dict:
    out = _run(["ffprobe", *_QUIET, "-print_format", "json",
                "-show_format", "-show_streams", path], timeout=120)
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned unparseable JSON: {exc}") from None


def _fraction(text: str) -> float:
    """'30000/1001' → 29.97. Reels are full of these and int() on them is 0."""
    try:
        if "/" in str(text):
            num, den = str(text).split("/", 1)
            den = float(den)
            return float(num) / den if den else 0.0
        return float(text)
    except (ValueError, TypeError):
        return 0.0


def probe(path: str) -> dict:
    """Container facts, normalised.

    Two subtleties that matter on phone-shot vertical video:

    **Rotation.** A 1920×1080 stream with a 90° display matrix is a portrait
    video. Reading width and height without applying the matrix labels every
    vertical reel as landscape, and every downstream aspect-ratio claim is then
    wrong. ffmpeg's decoder applies the rotation; ffprobe's stream fields do
    not, so the correction happens here.

    **Duration.** The format duration and the video stream duration disagree on
    files with trailing audio. The video stream is the one that matters for
    shot timing, so it wins when present.
    """
    info = ffprobe(path)
    fmt = info.get("format") or {}
    streams = info.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(v.get("width") or 0) if v else 0
    height = int(v.get("height") or 0) if v else 0
    rotation = 0
    if v:
        for sd in v.get("side_data_list") or []:
            if "rotation" in sd:
                try:
                    rotation = int(float(sd["rotation"])) % 360
                except (ValueError, TypeError):
                    rotation = 0
        if rotation in (90, 270):
            width, height = height, width

    duration = 0.0
    if v and v.get("duration"):
        duration = _fraction(v["duration"])
    if not duration:
        duration = _fraction(fmt.get("duration"))

    fps = _fraction(v.get("avg_frame_rate") if v else 0) or \
        _fraction(v.get("r_frame_rate") if v else 0)

    try:
        size = int(fmt.get("size") or 0)
    except (ValueError, TypeError):
        size = 0

    return {
        "duration": round(duration, 3),
        "width": width,
        "height": height,
        "rotation": rotation,
        "fps": round(fps, 3),
        "frames": int(round(fps * duration)) if fps and duration else 0,
        "video_codec": (v or {}).get("codec_name", ""),
        "audio_codec": (a or {}).get("codec_name", ""),
        "has_audio": a is not None,
        "audio_channels": int((a or {}).get("channels") or 0),
        "sample_rate": int(_fraction((a or {}).get("sample_rate") or 0)),
        "bitrate": int(_fraction(fmt.get("bit_rate") or 0)),
        "size_bytes": size,
        "aspect": round(width / height, 4) if width and height else 0.0,
        "vertical": bool(height and width and height > width),
        "format": (fmt.get("format_name") or "").split(",")[0],
    }


# ══════════════════════════════════════════════════════════════════════════
# Artifacts
# ══════════════════════════════════════════════════════════════════════════

def proxy(src: str, dst: str, height: int = 480, crf: int = 26) -> str:
    """480p faststart H.264 — the file the player actually loads.

    `+faststart` moves the moov atom to the front, which is the difference
    between a video that starts on the first chunk and one that must be fully
    downloaded before the first frame appears. On a 30 s reel served over a
    range request that is a two-second wait versus a fifteen-second one.

    `-vf scale=-2:H` keeps width even, because H.264 4:2:0 requires it and an
    odd width fails the encode with a message that does not mention parity.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    _run(["ffmpeg", *_FF, "-y", "-i", src,
          "-vf", f"scale=-2:{int(height)}:flags=bicubic",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
          "-pix_fmt", "yuv420p", "-profile:v", "main",
          "-c:a", "aac", "-b:a", "96k", "-ac", "1",
          "-movflags", "+faststart", dst])
    return dst


def poster(src: str, dst: str, at: float = 0.0, width: int = 720) -> str:
    """A frame worth showing.

    When `at` is 0 the frame is chosen by ffmpeg's `thumbnail` filter over the
    first three seconds rather than taken at t=0. Reels open on a fade, a black
    frame, or a one-frame flash; frame zero is a poster that makes the whole
    grid look broken, and the archive is five thousand posters.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if at > 0:
        cmd = ["ffmpeg", *_FF, "-y", "-ss", f"{at:.3f}", "-i", src,
               "-frames:v", "1", "-vf", f"scale={int(width)}:-2", "-q:v", "3", dst]
    else:
        cmd = ["ffmpeg", *_FF, "-y", "-i", src, "-vf",
               f"thumbnail=90,scale={int(width)}:-2", "-frames:v", "1",
               "-q:v", "3", dst]
    _run(cmd, timeout=300)
    return dst


def sprite(src: str, dst: str, duration: float, cols: int = 10,
           max_rows: int = 10, tile_width: int = 160,
           target_interval: float = 1.0) -> dict:
    """A single image holding the whole timeline as a grid.

    One HTTP request instead of a hundred. The player positions a background
    offset as the pointer moves, so scrub preview costs no network at all after
    the first load. Returns the geometry the player needs — without it the
    sprite is an unreadable collage.

    The grid is sized from the duration, not fixed. A fixed 10×10 gave a
    six-second clip a hundred tiles sixty milliseconds apart — a 500 KB sprite
    for a 127 KB proxy, four times the pixels of the video it previews. One
    tile per second, floor of two tiles, ceiling of `cols × max_rows`, and rows
    only as deep as the tiles require.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    duration = max(float(duration or 0), 0.1)
    cap = max(cols * max_rows, 1)
    n = max(2, min(int(duration / max(target_interval, 0.05)) + 1, cap))
    cols = min(cols, n)
    rows = max(1, -(-n // cols))          # ceil, so the last row is the ragged one
    n = min(n, cols * rows)
    fps = max(n / duration, 0.01)
    _run(["ffmpeg", *_FF, "-y", "-i", src,
          "-vf", (f"fps={fps:.6f},scale={int(tile_width)}:-2,"
                  f"tile={int(cols)}x{int(rows)}"),
          "-frames:v", "1", "-q:v", "5", dst], timeout=600)
    return {"path": dst, "cols": cols, "rows": rows, "count": n,
            "tile_width": tile_width, "interval": round(duration / n, 4)}


def loop(src: str, dst: str, start: float, seconds: float = 4.0,
         height: int = 360) -> str:
    """A short muted loop for hover preview. Silent on purpose: a grid of
    twenty autoplaying reels that all make noise is unusable."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    _run(["ffmpeg", *_FF, "-y", "-ss", f"{max(start, 0):.3f}", "-i", src,
          "-t", f"{seconds:.3f}", "-an",
          "-vf", f"scale=-2:{int(height)}",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], timeout=300)
    return dst


def wav(src: str, dst: str, rate: int = 16000) -> str:
    """16 kHz mono PCM — what every speech and audio model wants.

    Decoded once. Whisper, the second reader, diarisation, sound tagging,
    loudness and tempo are six passes; six independent decodes of the original
    would cost more CPU than any of them spend on their actual work.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    _run(["ffmpeg", *_FF, "-y", "-i", src, "-vn",
          "-acodec", "pcm_s16le", "-ar", str(int(rate)), "-ac", "1", dst],
         timeout=600)
    return dst


def waveform(src: str, dst: str, width: int = 1200, height: int = 96,
             colour: str = "0x7DD3FC") -> str:
    """The scrubber's background. Colour defaults to the speech channel's blue
    so the player's timeline and the evidence spectrum agree."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    _run(["ffmpeg", *_FF, "-y", "-i", src,
          "-filter_complex",
          (f"showwavespic=s={int(width)}x{int(height)}:colors={colour}"
           ":split_channels=0"),
          "-frames:v", "1", dst], timeout=300)
    return dst


def frames_at(src: str, times, outdir: str, width: int = 640,
              prefix: str = "f") -> list:
    """Extract specific timestamps as JPEGs.

    One ffmpeg call per frame with `-ss` *before* `-i`, which seeks by keyframe
    index rather than decoding from zero. On a 60-second reel that is the
    difference between 40 ms and 3 s per frame, and the keyframes pass asks for
    a hundred of them.

    A zero exit code is not proof of a file. `-ss` past the last decodable frame
    of a container whose declared duration is optimistic makes ffmpeg exit 0
    having written nothing, and the caller then holds a path to a file that does
    not exist — which is how `imread_('.../frames/s0010_0003.jpg'): can't
    open/read file` reached the log from a workdir nothing had evicted. Existence
    and a non-trivial size are what count as extracted.
    """
    os.makedirs(outdir, exist_ok=True)
    out = []
    for i, t in enumerate(times):
        dst = os.path.join(outdir, f"{prefix}{i:04d}.jpg")
        try:
            _run(["ffmpeg", *_FF, "-y", "-ss", f"{max(float(t), 0):.3f}",
                  "-i", src, "-frames:v", "1",
                  "-vf", f"scale={int(width)}:-2", "-q:v", "3", dst],
                 timeout=120)
        except MediaError:
            # A seek past the end of a file whose declared duration was
            # optimistic. One missing frame is not a failed pass.
            continue
        try:
            if os.path.getsize(dst) < 512:      # a JPEG header and nothing else
                os.remove(dst)
                continue
        except OSError:
            continue                            # ffmpeg said fine, wrote nothing
        out.append({"index": i, "t": round(float(t), 3), "path": dst})
    return out


def build_artifacts(src: str, outdir: str, meta: dict | None = None,
                    proxy_height: int = 480, sprite_cols: int = 10,
                    loop_seconds: float = 4.0) -> dict:
    """Everything the player and the vision passes need, in one call.

    Each artifact is attempted independently and its failure recorded rather
    than raised. A reel with no audio stream cannot produce a wav or a waveform,
    and that is not a reason to lose its proxy and its poster — the audio passes
    will see the missing artifact and skip themselves, which is the coverage
    ledger's `skipped` state doing its job.
    """
    os.makedirs(outdir, exist_ok=True)
    meta = meta or probe(src)
    duration = float(meta.get("duration") or 0)
    out: dict = {"artifacts": {}, "errors": {}}

    def attempt(name, fn):
        try:
            out["artifacts"][name] = fn()
        except MediaError as exc:
            out["errors"][name] = str(exc)[:300]

    px = os.path.join(outdir, "proxy.mp4")
    attempt("proxy", lambda: proxy(src, px, proxy_height))

    # Later artifacts read the proxy when it exists: it is a tenth the size and
    # decodes several times faster, and a 160 px sprite tile cannot tell the
    # difference between a 480p source and a 1080p one.
    small = px if os.path.exists(px) else src

    attempt("poster", lambda: poster(small, os.path.join(outdir, "poster.jpg")))
    attempt("sprite", lambda: sprite(
        small, os.path.join(outdir, "sprite.jpg"), duration, cols=sprite_cols))

    # The loop starts a third of the way in — past the intro, before the payoff
    # is given away. Short reels start at zero rather than at a fraction of
    # nothing.
    start = max(duration / 3.0 - loop_seconds / 2.0, 0.0) if duration > 8 else 0.0
    attempt("loop", lambda: loop(small, os.path.join(outdir, "loop.mp4"),
                                 start, loop_seconds))

    if meta.get("has_audio"):
        w = os.path.join(outdir, "audio.wav")
        attempt("wav", lambda: wav(src, w))
        if os.path.exists(w):
            attempt("waveform", lambda: waveform(
                w, os.path.join(outdir, "waveform.png")))
    else:
        out["errors"]["wav"] = "no audio stream"

    for name, val in list(out["artifacts"].items()):
        path = val["path"] if isinstance(val, dict) else val
        try:
            out["artifacts"][name] = (
                {**val, "bytes": os.path.getsize(path)} if isinstance(val, dict)
                else {"path": path, "bytes": os.path.getsize(path)})
        except OSError:
            out["errors"][name] = "written but unreadable"
            out["artifacts"].pop(name, None)

    out["meta"] = meta
    return out


def loudness(path: str) -> dict:
    """EBU R128 integrated loudness, range and true peak, from ffmpeg.

    The standard is intricate — K-weighting, gated 400 ms blocks, a relative
    gate at −10 LU — and libebur128 is the reference implementation that
    broadcasters actually certify against. Reimplementing it in numpy would be
    a week of work to produce numbers slightly different from everyone else's.
    ffmpeg ships it; this parses its summary.

    The log level is raised for this one call, deliberately. The ebur128 filter
    writes its summary with `AV_LOG_INFO`, so the module-wide `-loglevel error`
    that keeps every other call quiet also throws away the entire result — the
    filter runs, the numbers are computed, and the block that would print them
    is discarded before it reaches stderr. That produced "ebur128 produced no
    summary — is there an audio stream?" on files that had perfectly good audio,
    which is a wrong answer rather than a missing one.
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info",
           "-nostats", "-i", path, "-af", "ebur128=peak=true",
           "-f", "null", "-"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise MediaError(f"ebur128 failed: {exc}") from None
    text = res.stderr or ""
    if "Summary:" not in text:
        # Two very different failures used to share one message. Which one it is
        # decides whether the pass should be re-run after a fix or marked as
        # correctly declined forever, so they are separated here.
        low = text.lower()
        if ("does not contain any stream" in low
                or "audio stream" in low and "not" in low
                or "invalid data found" in low):
            raise MediaError("no audio stream to measure")
        tail = " ".join(text.strip().splitlines()[-3:])[:300]
        raise MediaError(
            "ebur128 ran but printed no summary"
            + (f" — ffmpeg said: {tail}" if tail else ""))
    tail = text.rsplit("Summary:", 1)[1]

    def grab(label, unit):
        m = re.search(rf"{label}:\s*(-?\d+(?:\.\d+)?)\s*{unit}", tail)
        return float(m.group(1)) if m else None

    return {
        "integrated_lufs": grab("I", "LUFS"),
        "range_lu": grab("LRA", "LU"),
        "true_peak_dbfs": grab("Peak", "dBFS"),
        "lra_low": grab("LRA low", "LUFS"),
        "lra_high": grab("LRA high", "LUFS"),
    }


# ══════════════════════════════════════════════════════════════════════════
# The shot prefilter
# ══════════════════════════════════════════════════════════════════════════

def scene_candidates(path: str, threshold: float = 27.0,
                     min_seconds: float = 0.25) -> list:
    """PySceneDetect's content detector — cheap, CPU, and roughly right.

    Returns [(start, end)] in seconds. This is a *prefilter*, not the answer:
    a threshold on HSV content difference finds hard cuts reliably and misses
    dissolves, match cuts and whip pans, which edited reels use constantly.
    TransNetV2 resolves those, and giving it a candidate list means it only has
    to adjudicate the boundaries instead of scanning everything.

    If PySceneDetect is not installed the function degrades to a fixed grid
    rather than failing. A grid is a bad shot list, but a bad shot list still
    lets every downstream pass produce something, and the shots pass can be
    re-run later under its own observer without touching anything else.
    """
    try:
        from scenedetect import open_video, SceneManager  # noqa: PLC0415
        from scenedetect.detectors import ContentDetector  # noqa: PLC0415
    except Exception:
        return _grid(path, seconds=2.5)

    try:
        video = open_video(path)
        mgr = SceneManager()
        fps = video.frame_rate or 30.0
        mgr.add_detector(ContentDetector(
            threshold=threshold,
            min_scene_len=max(int(min_seconds * fps), 1)))
        mgr.detect_scenes(video, show_progress=False)
        scenes = mgr.get_scene_list()
    except Exception:
        return _grid(path, seconds=2.5)

    out = [(round(s.get_seconds(), 3), round(e.get_seconds(), 3))
           for s, e in scenes]
    if not out:
        # One continuous take — a real and common answer for a talking-head
        # reel, not a failure.
        meta = probe(path)
        return [(0.0, round(float(meta.get("duration") or 0), 3))]
    return out


def _grid(path: str, seconds: float = 2.5) -> list:
    try:
        duration = float(probe(path).get("duration") or 0)
    except MediaError:
        return []
    if duration <= 0:
        return []
    out, t = [], 0.0
    while t < duration:
        out.append((round(t, 3), round(min(t + seconds, duration), 3)))
        t += seconds
    return out


def sample_times(shots, per_shot: int = 1) -> list:
    """Timestamps to extract for a shot list.

    Samples are pulled in from the boundaries by 15% because the first and last
    frames of a shot are the ones most likely to be mid-transition — a
    dissolve's midpoint is two images at once, and every vision pass reading it
    reports a scene that never existed.
    """
    out = []
    for a, b in shots:
        span = max(b - a, 0.0)
        if per_shot <= 1:
            out.append(round(a + span * 0.5, 3))
            continue
        for k in range(per_shot):
            frac = 0.15 + (0.70 * k / max(per_shot - 1, 1))
            out.append(round(a + span * frac, 3))
    return out
