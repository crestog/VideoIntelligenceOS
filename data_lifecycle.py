"""
VIOS Data Lifecycle — full-res frame tier purge + on-demand recovery

Full-res frame JPEGs are a *processing artifact*: YOLO/OCR (QUEUE_ANALYZE)
and SigLIP/CLIP/Depth/RAFT (QUEUE_VISION_EMBED) read them once, then they
just sit on disk (~100 MB per minute of video) until Kaggle's 20 GB volume
fills and the whole pipeline stalls.

Lifecycle:
  1. Each frame-consuming worker calls mark_stage_done(msg_id, stage) on
     success ("analyzed" | "embedded").
  2. When BOTH stages are done, maybe_purge_frames() deletes the full-res
     tier — but first writes frames_index.json (the sorted filename list)
     so the workstation UI keeps its frame count/timestamps, served from
     the ~8 KB preview tier.
  3. ensure_full_frame() re-extracts a single full-res frame from the
     source video on demand (spatial proof, full-tier viewing).

The Oracle worker reads the *video file*, not the frames, so it does not
participate in the purge decision.
"""

import glob
import json
import os
import shutil
import sqlite3
import subprocess
import time

from config import (VIDEO_DIR, DB_PATH, SQLITE_TIMEOUT, PREVIEW_DIR_NAME,
                    PURGE_FULL_FRAMES, FRAME_INDEX_NAME)
from logger import vios_log

# Stages that must complete before the full-res tier can be dropped
FRAME_CONSUMER_STAGES = {"analyzed", "embedded"}
_STAGE_KEY_TTL = 7 * 24 * 3600   # Redis key lifetime; sessions never last this long


def log(msg, level="INFO"):
    vios_log(msg, "LIFECYCLE", level)


def _stage_key(msg_id):
    return f"VIOS_STAGES_DONE:{msg_id}"


def mark_stage_done(msg_id, stage):
    """
    Record that a frame-consuming stage finished for this video, and purge the
    full-res tier once every consumer is done. Best-effort — never raises.
    """
    try:
        from queue_manager import get_redis
        r = get_redis()
        key = _stage_key(msg_id)
        r.sadd(key, stage)
        r.expire(key, _STAGE_KEY_TTL)
        done = {s for s in r.smembers(key)}
    except Exception as e:
        log(f"⚠️ stage tracking unavailable ({type(e).__name__}) — purge deferred", "WARN")
        return

    if FRAME_CONSUMER_STAGES.issubset(done):
        maybe_purge_frames(msg_id)


def maybe_purge_frames(msg_id):
    """
    Delete the full-res frame tier for one video (preview tier + thumbnail
    stay). Writes frames_index.json first so the UI keeps its frame list.
    Returns MB freed (0.0 if purge disabled or nothing to do).
    """
    if not PURGE_FULL_FRAMES:
        return 0.0

    frames_dir = os.path.join(VIDEO_DIR, f"frames_{msg_id}")
    if not os.path.isdir(frames_dir):
        return 0.0

    full_frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not full_frames:
        return 0.0

    # Preserve the filename list (carries frame_idx + timestamp) for the UI
    index_path = os.path.join(frames_dir, FRAME_INDEX_NAME)
    try:
        with open(index_path, "w") as f:
            json.dump({"names": [os.path.basename(p) for p in full_frames],
                       "purged_at": time.time()}, f)
    except Exception as e:
        log(f"⚠️ #{msg_id}: frame index write failed ({e}) — keeping full tier", "WARN")
        return 0.0

    freed = 0
    for p in full_frames:
        try:
            freed += os.path.getsize(p)
            os.remove(p)
        except OSError:
            pass

    freed_mb = freed / (1024 * 1024)
    log(f"🧹 #{msg_id}: purged {len(full_frames)} full-res frames "
        f"({freed_mb:.0f} MB freed) — preview tier retained", "SUCCESS")
    return freed_mb


def load_frame_index(frames_dir):
    """Return the saved full-tier filename list, or [] if no index exists."""
    try:
        with open(os.path.join(frames_dir, FRAME_INDEX_NAME)) as f:
            return json.load(f).get("names", [])
    except Exception:
        return []


def ensure_full_frame(msg_id, frame_idx):
    """
    Return the path of one full-res frame, re-extracting it from the source
    video if the full tier was purged. Returns None if unrecoverable.
    """
    frames_dir = os.path.join(VIDEO_DIR, f"frames_{msg_id}")
    matches = glob.glob(os.path.join(frames_dir, f"frame_{frame_idx:05d}_ts_*.jpg"))
    if matches:
        return matches[0]

    # Need fps + source path to seek to the exact frame
    try:
        conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        row = conn.execute("SELECT fps, abs_path FROM videos WHERE msg_id = ?",
                           (msg_id,)).fetchone()
        conn.close()
    except Exception:
        row = None
    fps = (row[0] if row and row[0] else 30.0)

    video_path = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
    if not os.path.exists(video_path):
        return None

    ts = frame_idx / fps if fps > 0 else 0.0
    os.makedirs(frames_dir, exist_ok=True)
    out_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}_ts_{ts:.3f}s.jpg")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{ts:.3f}", "-i", video_path,
         "-frames:v", "1", "-q:v", "2", out_path],
        capture_output=True, text=True, timeout=60)
    if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    log(f"⚠️ #{msg_id}: on-demand frame {frame_idx} re-extraction failed: "
        f"{proc.stderr.strip()[:150]}", "WARN")
    return None


def emergency_disk_sweep(min_free_gb=1.0):
    """
    Last-resort space recovery: purge full-res tiers for ALL videos that have
    a preview tier, oldest first, until free space exceeds min_free_gb.
    Only used by the CV engine when it would otherwise stall on disk.
    Returns total MB freed.
    """
    freed_mb = 0.0
    try:
        free_gb = shutil.disk_usage(VIDEO_DIR).free / (1024 ** 3)
        if free_gb >= min_free_gb:
            return 0.0

        candidates = []   # (still_needed, mtime, msg_id)
        try:
            from queue_manager import get_redis
            r = get_redis()
        except Exception:
            r = None
        for folder in glob.glob(os.path.join(VIDEO_DIR, "frames_*")):
            if not os.path.isdir(os.path.join(folder, PREVIEW_DIR_NAME)):
                continue   # no preview tier → purging would blank the UI
            try:
                msg_id = int(os.path.basename(folder).split("_")[1])
            except (IndexError, ValueError):
                continue
            done = set()
            if r is not None:
                try:
                    done = set(r.smembers(_stage_key(msg_id)))
                except Exception:
                    pass
            still_needed = not FRAME_CONSUMER_STAGES.issubset(done)
            candidates.append((still_needed, os.path.getmtime(folder), msg_id))

        # Fully-processed videos first, then oldest — frames still awaiting
        # analyze/embed are sacrificed only if nothing else frees enough space.
        for still_needed, _mtime, msg_id in sorted(candidates):
            if still_needed:
                log(f"🚨 purging NOT-yet-analyzed frames for #{msg_id} (disk critical) — "
                    "its analyze/embed results will be empty", "WARN")
            freed_mb += maybe_purge_frames(msg_id)
            if shutil.disk_usage(VIDEO_DIR).free / (1024 ** 3) >= min_free_gb:
                break
        if freed_mb:
            log(f"🚨 Emergency sweep freed {freed_mb:.0f} MB", "WARN")
    except Exception as e:
        log(f"⚠️ emergency sweep failed: {e}", "WARN")
    return freed_mb
