"""
Per-video asset sets: what gets uploaded alongside the mp4, and why.

The channel already holds one message per video. That is enough to *have* the
archive and not nearly enough to *use* it: Atlas can only read a video's bytes
through an MTProto media session, which needs a login, costs an auth handshake,
and is metered by a handful of permits. Playing a moment somebody searched for
means opening one of those sessions, seeking a 1 MiB chunk boundary, and hoping
the mp4's moov atom is at the front. Hovering a search result to preview it
means doing that again, for every result on the page.

So each video gets an *asset set* threaded under its message:

    <key>.mp4                 the video            (already there)
    <key>.json                metadata, merged with the Instagram export slice
    <key>-chunk-0000.mp4 …    2-second standalone clips, in order
    <key>-manifest.json       the index over all of the above

The clips are the point. `ffmpeg -f segment -c copy` is a demux/remux, not a
transcode — it costs I/O, not GPU, and it cuts on keyframes so every clip is
independently decodable. A 2-second vertical reel clip is ~150–400 KB, which is
two orders of magnitude under the Bot API's 20 MB `getFile` ceiling. That is the
whole optimisation: **a clip is reachable over plain HTTPS with nothing but the
bot token.** No session, no login, no permit. Atlas asking "show me 12.4 s of
this video" becomes one HTTP GET of a small file.

A correction to the earlier design, recorded here because it is the kind of
thing that reads as true until it is tested: a segment is **not** a byte slice
of the original file. `-c copy` writes a fresh mp4 container per segment, with
its own `ftyp` and `moov`, so the sample payload matches but nothing else does
and the byte offsets do not line up. A manifest mapping original byte ranges to
segments would be wrong at every entry. Clips are therefore indexed by **time**,
the way HLS does it, and byte-range playback of the whole file keeps using the
existing sparse-chunk path (with a new whole-file HTTP shortcut, which covers
nearly every reel — see `atlas/media.http_pull`).

What the clips buy, concretely:

  * hovering a search result plays the matched moment, immediately, without
    downloading the video
  * a poster frame can be extracted at *any* timestamp for a video that has
    never been downloaded — fetch the one clip that covers it, decode a frame
  * seeking to a moment costs one small file instead of a media session

`<key>-frames.tar.zst` is built only when `VIOS_UPLOAD_FRAMES=1`. Nothing in
Atlas reads raw frames — it reads video, clips, evidence, manifest and
metadata — and the analysis tier for one reel is tens of MB compressed. Across
hundreds of videos that would dominate every capture run to feed a consumer
that does not exist. The builder is here and works; the default is off, and
that is a deliberate choice rather than an omission.
"""

import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time

# ── knobs ─────────────────────────────────────────────────────────────────
# 2 seconds is the shortest clip that reliably contains a keyframe in a reel
# encoded at the usual 2 s GOP, and it is also about the length of a hover
# preview. Shorter and ffmpeg starts emitting clips that begin with a
# non-keyframe; longer and a preview costs bytes nobody watches.
CHUNK_SECONDS = float(os.environ.get("VIOS_CHUNK_SECONDS", "2.0"))

# A cap with a stated reason, not a silent truncation. 240 clips is 8 minutes
# at 2 s, which covers every reel and most long-form uploads; past that the
# per-message cost of the upload stops being worth it and the manifest records
# how many were dropped so the gap is visible rather than assumed away.
MAX_CHUNKS = int(os.environ.get("VIOS_MAX_CHUNKS", "240"))

# Telegram tolerates bursts but a channel has a per-minute message budget, and
# 15 documents back-to-back per video is exactly the shape that trips it. The
# uploader already retries a 429 with the server's own `retry_after`; this just
# keeps it from getting there.
CHUNK_PAUSE = float(os.environ.get("VIOS_CHUNK_PAUSE", "0.35"))

UPLOAD_FRAMES = os.environ.get("VIOS_UPLOAD_FRAMES", "0") == "1"

MANIFEST_VERSION = 1
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


class AssetError(RuntimeError):
    """Building an asset failed. Never fatal to a capture — assets are extra."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _run(argv: list, timeout: float) -> tuple:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    except OSError as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stderr or "")[-2000:]


# ══════════════════════════════════════════════════════════════════════════
# CLIPS
# ══════════════════════════════════════════════════════════════════════════
def segment(video_path: str, out_dir: str, key: str,
            seconds: float = None, timeout: float = 300.0) -> dict:
    """Cut `video_path` into standalone clips. Returns {clips, truncated, note}.

    `-c copy` so no frame is re-encoded: the sample bytes that land in a clip
    are the ones that were in the source, which means a clip is exactly as good
    as the original and costs a disk copy to produce.

    The clip *times* come from ffmpeg's own segment list rather than from
    `index * seconds`. Segmenting cuts on keyframes, so a reel with a 2.4 s GOP
    produces 2.4 s clips from a `-segment_time 2` request — computing the times
    would put every clip's boundary in the wrong place, and every seek with it.
    `-segment_list_type csv` writes `name,start,end` per clip, which is the
    truth from the muxer.

    `-reset_timestamps 1` makes each clip start at t=0 in its own timeline;
    without it a player handed clip 7 seeks to 14 s inside a 2 s file and shows
    nothing.
    """
    seconds = float(seconds or CHUNK_SECONDS)
    os.makedirs(out_dir, exist_ok=True)
    listing = os.path.join(out_dir, "segments.csv")
    pattern = os.path.join(out_dir, f"{key}-chunk-%04d.mp4")

    code, err = _run([
        FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-c", "copy", "-map", "0",
        "-f", "segment", "-segment_time", f"{seconds:g}",
        "-reset_timestamps", "1",
        "-segment_list", listing, "-segment_list_type", "csv",
        "-movflags", "+faststart",
        pattern,
    ], timeout=timeout)

    if code != 0 and not os.path.exists(listing):
        raise AssetError(f"ffmpeg segment failed ({code}): {err[-300:]}")

    clips = []
    try:
        with open(listing, "r", encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                path = os.path.join(out_dir, os.path.basename(row[0]))
                if not os.path.exists(path):
                    continue
                try:
                    t0, t1 = float(row[1]), float(row[2])
                except ValueError:
                    continue
                clips.append({"i": len(clips), "t0": round(t0, 3),
                              "t1": round(t1, 3), "path": path,
                              "name": os.path.basename(path),
                              "bytes": os.path.getsize(path)})
    except OSError as exc:
        raise AssetError(f"could not read the segment list: {exc}")

    if not clips:
        # The muxer wrote files but no usable list, or wrote neither. Fall back
        # to the files on disk with computed times: worse than the muxer's
        # answer, better than losing the clips entirely.
        for n, path in enumerate(sorted(glob.glob(
                os.path.join(out_dir, f"{key}-chunk-*.mp4")))):
            clips.append({"i": n, "t0": round(n * seconds, 3),
                          "t1": round((n + 1) * seconds, 3), "path": path,
                          "name": os.path.basename(path),
                          "bytes": os.path.getsize(path)})

    truncated = 0
    if len(clips) > MAX_CHUNKS:
        truncated = len(clips) - MAX_CHUNKS
        for extra in clips[MAX_CHUNKS:]:
            try:
                os.remove(extra["path"])
            except OSError:
                pass
        clips = clips[:MAX_CHUNKS]

    note = ""
    if truncated:
        note = (f"{truncated} clip(s) past {MAX_CHUNKS} "
                f"({MAX_CHUNKS * seconds:.0f}s) were not built")
    elif code != 0:
        note = f"ffmpeg exited {code} but produced {len(clips)} clip(s)"
    return {"clips": clips, "truncated": truncated, "note": note}


# ══════════════════════════════════════════════════════════════════════════
# FRAMES (off by default — see the module docstring)
# ══════════════════════════════════════════════════════════════════════════
def pack_frames(frames_dir: str, out_path: str) -> str:
    """Tar the analysis frames, zstd if available, gzip if not.

    zstd beats gzip by roughly 2× on JPEG-of-JPEG at a fraction of the CPU, but
    it is not in the standard library and Kaggle's image does not always carry
    the CLI. Falling back to gzip and *renaming the output accordingly* matters:
    a `.tar.zst` that is secretly gzip is a file nothing can open.
    """
    if not os.path.isdir(frames_dir):
        raise AssetError(f"no frames directory at {frames_dir}")
    jpegs = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not jpegs:
        raise AssetError("frames directory is empty")

    tar_path = out_path
    if out_path.endswith(".zst"):
        raw = out_path[:-4]
    else:
        raw = out_path
    with tarfile.open(raw, "w") as tar:
        for p in jpegs:
            tar.add(p, arcname=os.path.basename(p))

    if out_path.endswith(".zst"):
        zstd = shutil.which("zstd")
        if zstd:
            code, err = _run([zstd, "-q", "-19", "--rm", "-o", out_path, raw],
                             timeout=1800.0)
            if code == 0 and os.path.exists(out_path):
                return out_path
        try:
            import zstandard  # noqa: F401  (present in some images)
            import zstandard as zstd_mod
            cctx = zstd_mod.ZstdCompressor(level=10)
            with open(raw, "rb") as fin, open(out_path, "wb") as fout:
                cctx.copy_stream(fin, fout)
            os.remove(raw)
            return out_path
        except Exception:
            pass
        # Neither route worked — gzip it and say so in the name.
        import gzip
        gz_path = raw + ".gz"
        with open(raw, "rb") as fin, gzip.open(gz_path, "wb",
                                               compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout, 1024 * 1024)
        os.remove(raw)
        return gz_path
    return tar_path


# ══════════════════════════════════════════════════════════════════════════
# THE MANIFEST
# ══════════════════════════════════════════════════════════════════════════
def manifest_name(key: str) -> str:
    return f"{key}-manifest.json"


def build_manifest(key: str, video_part: dict, clips: list, assets: list,
                   duration: float = None, truncated: int = 0,
                   note: str = "") -> dict:
    """The index Atlas reads instead of scanning the channel.

    Every entry carries both `msg_id` and `file_id` on purpose. `file_id` is the
    cheap route — plain HTTPS, bot token, no session — but Telegram may expire
    one, and a `file_id` minted by a different bot is meaningless. `msg_id` is
    permanent and always resolvable over MTProto, so it is the fallback that
    keeps an asset reachable for as long as the channel exists.
    """
    return {
        "v": MANIFEST_VERSION,
        "key": key,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration": round(float(duration), 3) if duration else None,
        "chunk_seconds": CHUNK_SECONDS,
        "video": video_part,
        "chunks": [{"i": c["i"], "t0": c["t0"], "t1": c["t1"],
                    "name": c["name"], "msg_id": c.get("msg_id", 0),
                    "file_id": c.get("file_id", ""), "bytes": c.get("bytes", 0),
                    "sha256": c.get("sha256", "")}
                   for c in clips],
        "chunks_truncated": int(truncated),
        "assets": list(assets),
        "note": note,
    }


# ══════════════════════════════════════════════════════════════════════════
# BUILD + UPLOAD, THE ONE CALL THE ENGINE MAKES
# ══════════════════════════════════════════════════════════════════════════
def publish_assets(tg, result: dict, sent: dict, key: str, work: str,
                   ig_slice: dict = None, frames_dir: str = None,
                   on_note=None) -> dict:
    """Build and upload the asset set for one captured video.

    Returns {clips, uploaded, manifest_msg_id, manifest_file_id, notes}.

    **This never raises.** The video and its record are already in the channel
    by the time this runs; assets are an optimisation over data that is already
    safe, and a reel that plays slightly slower is not worth failing a capture
    for. Every failure becomes a note the ledger logs and the run continues.
    """
    notes = []
    out = {"clips": 0, "uploaded": 0, "manifest_msg_id": 0,
           "manifest_file_id": "", "notes": notes, "truncated": 0}

    def say(msg: str):
        notes.append(msg)
        if on_note:
            try:
                on_note(msg)
            except Exception:
                pass

    video_path = result.get("video")
    if not video_path or not os.path.exists(video_path):
        return out                      # a photo post has no clips to cut

    anchor = int(sent.get("msg_id") or sent.get("message_id") or 0)
    if not anchor:
        say("no anchor message id — assets not uploaded")
        return out

    # ── 1. metadata, merged with the Instagram export slice ───────────────
    assets = []
    meta_path = os.path.join(work, f"{key}.json")
    try:
        merged = dict(result.get("record") or {})
        if ig_slice:
            merged["instagram_export"] = ig_slice
        merged["telegram"] = {"msg_id": anchor,
                              "record_msg_id": sent.get("record_msg_id", 0),
                              "file_id": sent.get("file_id", "")}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, separators=(",", ":"))
        got = tg.send_document(meta_path, caption=f"meta · vios:{key}",
                               reply_to=anchor,
                               file_name=f"{key}.json")
        assets.append({"kind": "meta", "name": f"{key}.json",
                       "msg_id": got["message_id"], "file_id": got["file_id"],
                       "bytes": os.path.getsize(meta_path),
                       "sha256": _sha256(meta_path)})
        out["uploaded"] += 1
    except Exception as exc:
        say(f"metadata asset failed: {type(exc).__name__}: {exc}")

    # ── 2. clips ──────────────────────────────────────────────────────────
    clips = []
    try:
        cut = segment(video_path, os.path.join(work, "chunks"), key)
        clips = cut["clips"]
        out["truncated"] = cut["truncated"]
        if cut["note"]:
            say(cut["note"])
    except Exception as exc:
        say(f"clip build failed: {type(exc).__name__}: {exc}")

    kept = []
    for clip in clips:
        try:
            got = tg.send_document(clip["path"],
                                   caption=f"clip {clip['i']} · vios:{key}",
                                   reply_to=anchor, file_name=clip["name"])
            clip["msg_id"] = got["message_id"]
            clip["file_id"] = got["file_id"]
            clip["sha256"] = _sha256(clip["path"])
            kept.append(clip)
            out["uploaded"] += 1
        except Exception as exc:
            # A clip that will not upload is a hole in the preview index, not a
            # broken video. Record which one and keep going — the manifest will
            # simply not list it, and Atlas falls back to byte-range playback
            # for that stretch.
            say(f"clip {clip['i']} failed: {str(exc)[:120]}")
        if CHUNK_PAUSE:
            time.sleep(CHUNK_PAUSE)
    out["clips"] = len(kept)

    # ── 3. frames, only when asked for ────────────────────────────────────
    if UPLOAD_FRAMES and frames_dir and os.path.isdir(frames_dir):
        try:
            packed = pack_frames(frames_dir,
                                 os.path.join(work, f"{key}-frames.tar.zst"))
            got = tg.send_document(packed, caption=f"frames · vios:{key}",
                                   reply_to=anchor,
                                   file_name=os.path.basename(packed))
            assets.append({"kind": "frames", "name": os.path.basename(packed),
                           "msg_id": got["message_id"],
                           "file_id": got["file_id"],
                           "bytes": os.path.getsize(packed),
                           "sha256": _sha256(packed)})
            out["uploaded"] += 1
        except Exception as exc:
            say(f"frame pack failed: {type(exc).__name__}: {exc}")

    # ── 4. the manifest, last, because it names everything above ──────────
    video_part = {"kind": "video", "name": f"{key}.mp4", "msg_id": anchor,
                  "file_id": sent.get("file_id", ""),
                  "bytes": int(result.get("bytes") or 0),
                  "sha256": result.get("sha256", ""),
                  "width": sent.get("width"), "height": sent.get("height"),
                  "duration": sent.get("duration")}
    try:
        man = build_manifest(key, video_part, kept, assets,
                             duration=sent.get("duration"),
                             truncated=out["truncated"],
                             note="; ".join(notes)[:500])
        man_path = os.path.join(work, manifest_name(key))
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, separators=(",", ":"))
        got = tg.send_document(man_path, caption=f"manifest · vios:{key}",
                               reply_to=anchor, file_name=manifest_name(key))
        out["manifest_msg_id"] = got["message_id"]
        out["manifest_file_id"] = got["file_id"]
        out["uploaded"] += 1
    except Exception as exc:
        say(f"manifest failed: {type(exc).__name__}: {exc}")

    return out
