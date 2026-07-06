"""
VIOS CV Frame Engine v2 — ffmpeg Dual-Tier Frame Extraction Worker

Replaces the OpenCV read/imwrite loop (single-threaded decode + sequential
JPEG encode) with ONE ffmpeg pass that produces BOTH tiers simultaneously:

  tier FULL    frames_<id>/frame_%05d_ts_%.3fs.jpg      (native resolution)
  tier PREVIEW frames_<id>/.preview/frame_%05d.jpg      (320px wide, ~8 KB)

ffmpeg decodes once and fans out through a split filter, using all CPU cores
for JPEG encoding — typically 4-8x faster than the OpenCV loop. The preview
tier is what makes <2s workspace opening possible: ~8 KB × N frames streams
to the browser in one or two round trips.

Job flow (unchanged lifecycle): claim → extract → DB write → ack
Plus NEW: on success, pushes an analysis job to QUEUE_ANALYZE so the GPU
worker can transcribe + tag the video without re-touching the file logic.
"""

import glob
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time

import redis as redis_lib

from queue_manager import claim_job, ack_job, fail_job, push_job
from config import (VIDEO_DIR, THUMB_DIR, DB_PATH, QUEUE_VISION, QUEUE_ANALYZE,
                    DISK_PAUSE_THRESHOLD_GB, DISK_WARN_THRESHOLD_GB,
                    PREVIEW_DIR_NAME, PREVIEW_WIDTH, PREVIEW_QUALITY, FULL_QUALITY,
                    SQLITE_TIMEOUT)
from logger import vios_log

QUEUE_NAME = QUEUE_VISION


def log(msg):
    vios_log(msg, "CV", "INFO")


# ═══════════════════════════════════════════════════════════
# PROBING
# ═══════════════════════════════════════════════════════════
def probe_video(video_path):
    """ffprobe → dict(fps, width, height, duration, nb_frames-estimate)."""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate,avg_frame_rate,width,height,nb_frames',
         '-show_entries', 'format=duration',
         '-of', 'json', video_path],
        capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout or '{}')
    stream = (info.get('streams') or [{}])[0]
    fmt = info.get('format') or {}

    def _rate(s):
        try:
            num, den = s.split('/')
            return float(num) / float(den) if float(den) else 0.0
        except Exception:
            return 0.0

    fps = _rate(stream.get('avg_frame_rate', '0/1')) or _rate(stream.get('r_frame_rate', '30/1')) or 30.0
    duration = float(fmt.get('duration') or 0)
    nb = int(stream.get('nb_frames') or 0) or int(duration * fps)
    return {
        'fps': fps,
        'width': int(stream.get('width') or 0),
        'height': int(stream.get('height') or 0),
        'duration': duration,
        'frames_estimate': nb,
    }


# ═══════════════════════════════════════════════════════════
# CORE EXTRACTION — one decode, two encode tiers
# ═══════════════════════════════════════════════════════════
def extract_video_data(video_path, msg_id):
    """Extract all frames (full + preview tier) via one ffmpeg pass.
    Returns the number of frames written. Raises on failure (worker retries)."""
    log(f"🔍 Probing video — ID: {msg_id}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise ValueError(f"Video file is empty (0 bytes): {video_path}")
    file_size_mb = file_size / (1024 * 1024)

    # ── Disk Space Guard ──
    free_gb = shutil.disk_usage(VIDEO_DIR).free / (1024 ** 3)
    while free_gb < DISK_PAUSE_THRESHOLD_GB:
        log(f"⏸️ Disk space low ({free_gb:.1f}GB) — pausing extraction...")
        time.sleep(30)
        free_gb = shutil.disk_usage(VIDEO_DIR).free / (1024 ** 3)
    if free_gb < DISK_WARN_THRESHOLD_GB:
        log(f"⚠️ Disk: {free_gb:.1f} GB available — LOW SPACE WARNING")

    meta = probe_video(video_path)
    fps = meta['fps']
    log(f"📊 {meta['width']}×{meta['height']} | {fps:.2f} fps | ~{meta['frames_estimate']} frames | {meta['duration']:.2f}s | {file_size_mb:.1f} MB")

    folder_id = f"frames_{msg_id}"
    frames_dir = os.path.join(VIDEO_DIR, folder_id)
    preview_dir = os.path.join(frames_dir, PREVIEW_DIR_NAME)
    os.makedirs(preview_dir, exist_ok=True)

    # ── ONE ffmpeg pass → both tiers ──
    # split=2: decode once, encode full-res and preview in parallel.
    # -start_number 0 keeps frame numbering aligned with the old convention.
    t_start = time.time()
    cmd = [
        'ffmpeg', '-y', '-v', 'error', '-i', video_path,
        '-filter_complex', f'[0:v]split=2[full][small];[small]scale={PREVIEW_WIDTH}:-2[preview]',
        '-map', '[full]', '-q:v', str(FULL_QUALITY), '-start_number', '0',
        os.path.join(frames_dir, 'raw_%05d.jpg'),
        '-map', '[preview]', '-q:v', str(PREVIEW_QUALITY), '-start_number', '0',
        os.path.join(preview_dir, 'frame_%05d.jpg'),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed: {proc.stderr[-300:]}")

    # ── Rename full-res frames to the timestamped convention the UI expects ──
    raw_frames = sorted(glob.glob(os.path.join(frames_dir, 'raw_*.jpg')))
    for raw in raw_frames:
        idx = int(re.search(r'raw_(\d+)\.jpg$', raw).group(1))
        ts = idx / fps if fps > 0 else 0.0
        os.rename(raw, os.path.join(frames_dir, f'frame_{idx:05d}_ts_{ts:.3f}s.jpg'))

    total_frames = len(raw_frames)
    extraction_time = time.time() - t_start
    actual_duration = total_frames / fps if fps > 0 else 0
    speed = total_frames / extraction_time if extraction_time > 0 else 0
    log(f"✅ Extracted {total_frames} frames ×2 tiers in {extraction_time:.1f}s ({speed:.0f} f/s)")

    if total_frames == 0:
        raise RuntimeError("ffmpeg produced zero frames — possible codec corruption")

    # ── Thumbnail from middle preview frame (already small — cheap copy) ──
    thumb_path = os.path.join(THUMB_DIR, f"{folder_id}.jpg")
    mid_preview = os.path.join(preview_dir, f'frame_{total_frames // 2:05d}.jpg')
    if os.path.exists(mid_preview) and not os.path.exists(thumb_path):
        shutil.copyfile(mid_preview, thumb_path)

    # ── Write to Database ──
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    try:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS videos
                     (msg_id INTEGER PRIMARY KEY, folder_id TEXT, title TEXT,
                      frames INTEGER, duration_sec REAL, duration_str TEXT,
                      thumb TEXT, first_frame TEXT, file_size_mb REAL, abs_path TEXT,
                      created_at REAL, fps REAL, width INTEGER, height INTEGER)''')
        for col, sqltype in (('created_at', 'REAL'), ('fps', 'REAL'),
                             ('width', 'INTEGER'), ('height', 'INTEGER')):
            try:
                c.execute(f"SELECT {col} FROM videos LIMIT 1")
            except sqlite3.OperationalError:
                c.execute(f"ALTER TABLE videos ADD COLUMN {col} {sqltype}")

        c.execute('''INSERT INTO videos
                     (msg_id, folder_id, title, frames, duration_sec, duration_str,
                      thumb, first_frame, file_size_mb, abs_path, created_at, fps, width, height)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT(msg_id) DO UPDATE SET
                      folder_id=excluded.folder_id, title=excluded.title,
                      frames=excluded.frames, duration_sec=excluded.duration_sec,
                      duration_str=excluded.duration_str, thumb=excluded.thumb,
                      first_frame=excluded.first_frame, file_size_mb=excluded.file_size_mb,
                      abs_path=excluded.abs_path, fps=excluded.fps,
                      width=excluded.width, height=excluded.height''',
                  (msg_id, folder_id, f"Video #{msg_id}", total_frames, actual_duration,
                   f"{actual_duration:.1f}s", f"/thumbs/{folder_id}.jpg",
                   f"/data/{folder_id}/frame_00000_ts_0.000s.jpg", file_size_mb,
                   frames_dir, time.time(), fps, meta['width'], meta['height']))
        conn.commit()
    finally:
        conn.close()
    log(f"💾 DB committed: ID={msg_id} | {total_frames} frames | {actual_duration:.1f}s")

    return total_frames


# ═══════════════════════════════════════════════════════════
# WORKER LOOP — claim → process → ack/fail (+ push analyze job)
# ═══════════════════════════════════════════════════════════
def run_worker():
    log("═══════════════════════════════════════════════")
    log("🚀 CV FRAME ENGINE v2 (ffmpeg dual-tier) — ONLINE")
    log(f"📂 Frames: {VIDEO_DIR} | 🖼️ Thumbs: {THUMB_DIR}")
    log(f"📡 Queue: {QUEUE_NAME} → pushes {QUEUE_ANALYZE} on success")
    log("═══════════════════════════════════════════════")

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)

    jobs_completed = 0
    jobs_failed = 0
    total_frames = 0

    while True:
        try:
            job, job_raw = claim_job(QUEUE_NAME, timeout=5)
        except (redis_lib.exceptions.TimeoutError, redis_lib.exceptions.ConnectionError, OSError) as e:
            log(f"⚠️ Redis connection issue: {type(e).__name__} — retrying in 3s...")
            time.sleep(3)
            continue
        except Exception as e:
            log(f"⚠️ Unexpected queue error: {type(e).__name__}: {e} — retrying in 3s...")
            time.sleep(3)
            continue

        if not job:
            # Check if paused by Admin panel
            try:
                r = redis_lib.Redis(host='localhost', port=6379, decode_responses=True, socket_timeout=1)
                if r.get('CV_PAUSED') == '1':
                    log('⏸️ Paused by Admin — waiting...')
                    while r.get('CV_PAUSED') == '1':
                        time.sleep(5)
                    log('▶️ Resumed by Admin')
            except Exception:
                pass
            continue

        payload = job.get("payload", job)
        msg_id = payload.get("msg_id")
        video_path = payload.get("path")
        retry_num = job.get("retries", 0)

        if not msg_id or not video_path:
            log(f"⚠️ Malformed job payload — acking to discard: {payload}")
            ack_job(QUEUE_NAME, job, job_raw)
            continue

        retry_tag = f" [RETRY #{retry_num}]" if retry_num > 0 else ""
        log(f"📥 JOB CLAIMED: Video #{msg_id}{retry_tag}")

        try:
            frames_extracted = extract_video_data(video_path, msg_id)
            ack_job(QUEUE_NAME, job, job_raw)
            jobs_completed += 1
            total_frames += frames_extracted

            # Hand off to the GPU analysis stage (idempotent by msg_id)
            push_job(QUEUE_ANALYZE, {"msg_id": msg_id, "path": video_path,
                                     "folder_id": f"frames_{msg_id}"})

            elapsed = time.time() - job.get("claimed_at", time.time())
            log(f"🏁 JOB #{msg_id} COMPLETED — {frames_extracted} frames in {elapsed:.2f}s → queued for GPU analysis")
            log(f"📊 Lifetime: {jobs_completed} ok | {jobs_failed} failed | {total_frames} frames")

        except Exception as e:
            result = fail_job(QUEUE_NAME, job, job_raw, str(e))
            jobs_failed += 1
            log(f"❌ JOB #{msg_id} FAILED → {result} | {str(e)[:200]}")
            time.sleep(2)


if __name__ == "__main__":
    run_worker()
