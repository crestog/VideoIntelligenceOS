"""
VIOS CV Frame Engine — GPU-Free Video Frame Extraction Worker

Consumes jobs from QUEUE_VISION via the reliable claim/ack/fail lifecycle.
For each video:
  1. Pre-calculates FPS, total frame count, duration, resolution
  2. Extracts EVERY frame as timestamped JPEGs
  3. Generates a 320x180 thumbnail from the middle frame
  4. Writes structured metadata to the SQLite 'videos' table
  5. Logs comprehensive progress to Kaggle console output

Crash Safety:
  - Uses claim_job (job moves to PROCESSING set on claim)
  - Uses ack_job on success (removes from PROCESSING)
  - Uses fail_job on failure (auto-retries up to 3x, then dead-letters)
  - Orphaned PROCESSING jobs are recovered by boot.py on restart
"""

import os
import cv2
import sqlite3
import time
import shutil
import redis as redis_lib
from queue_manager import claim_job, ack_job, fail_job
from config import VIDEO_DIR, THUMB_DIR, DB_PATH, QUEUE_VISION, DISK_PAUSE_THRESHOLD_GB, DISK_WARN_THRESHOLD_GB
from logger import vios_log

QUEUE_NAME = QUEUE_VISION


# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════
def log(msg):
    vios_log(msg, "CV", "INFO")


# ═══════════════════════════════════════════════════════════
# CORE FRAME EXTRACTION ENGINE
# ═══════════════════════════════════════════════════════════
def extract_video_data(video_path, msg_id):
    """
    Full video frame extraction pipeline.
    
    Reads every frame from the video file, calculates actual variable FPS
    and exact frame count. Saves each frame as a timestamped JPEG.
    Generates a centered thumbnail and writes all metadata to SQLite.
    
    Returns the total number of frames extracted.
    Raises exceptions on failure (caught by the worker loop for retry).
    """
    log(f"🔍 Analyzing video stream — ID: {msg_id}")

    # ── File Validation ──
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise ValueError(f"Video file is empty (0 bytes): {video_path}")

    file_size_mb = file_size / (1024 * 1024)
    log(f"📦 File: {os.path.basename(video_path)} | {file_size_mb:.2f} MB")

    # ── Disk Space Guard ──
    free_gb = shutil.disk_usage(VIDEO_DIR).free / (1024**3)
    while free_gb < DISK_PAUSE_THRESHOLD_GB:
        log(f"⏸️ Disk space low ({free_gb:.1f}GB) — pausing extraction, waiting for space...")
        time.sleep(30)
        free_gb = shutil.disk_usage(VIDEO_DIR).free / (1024**3)
    if free_gb < DISK_WARN_THRESHOLD_GB:
        log(f"⚠️ Disk: {free_gb:.1f} GB available — LOW SPACE WARNING")
    else:
        log(f"💾 Disk: {free_gb:.1f} GB available")

    # ── Open Video Stream ──
    folder_id = f"frames_{msg_id}"
    frames_dir = os.path.join(VIDEO_DIR, folder_id)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video — possible codec corruption")

    # ── Pre-Calculate Video Properties ──
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec_raw >> 8 * i) & 0xFF) for i in range(4)]) if codec_raw else "N/A"
    duration_estimate = total_frames_estimate / fps if fps > 0 else 0

    log(f"📊 Video Properties:")
    log(f"   Resolution: {width}×{height}")
    log(f"   FPS: {fps:.2f}")
    log(f"   Codec: {codec_str}")
    log(f"   Estimated Frames: {total_frames_estimate}")
    log(f"   Estimated Duration: {duration_estimate:.2f}s")

    # ── Extract EVERY Frame ──
    current_frame = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_sec = current_frame / fps
        filename = f"frame_{current_frame:05d}_ts_{timestamp_sec:.3f}s.jpg"
        cv2.imwrite(os.path.join(frames_dir, filename), frame)
        current_frame += 1

        # Progress logging every 200 frames
        if current_frame % 200 == 0:
            elapsed = time.time() - t_start
            speed = current_frame / elapsed if elapsed > 0 else 0
            if total_frames_estimate > 0:
                pct = current_frame / total_frames_estimate * 100
                eta = (total_frames_estimate - current_frame) / speed if speed > 0 else 0
                log(f"   ⏳ {current_frame}/{total_frames_estimate} ({pct:.1f}%) — {speed:.0f} f/s — ETA {eta:.1f}s")
            else:
                log(f"   ⏳ {current_frame} frames extracted — {speed:.0f} f/s")

    cap.release()

    extraction_time = time.time() - t_start
    actual_duration = current_frame / fps if fps > 0 else 0
    decode_speed = current_frame / extraction_time if extraction_time > 0 else 0

    log(f"✅ Extraction Complete:")
    log(f"   Actual Frames: {current_frame} (estimated {total_frames_estimate})")
    log(f"   Actual Duration: {actual_duration:.2f}s")
    log(f"   Decode Speed: {decode_speed:.0f} f/s")
    log(f"   Wall Time: {extraction_time:.2f}s")

    # ── Generate Thumbnail from Middle Frame ──
    thumb_path = os.path.join(THUMB_DIR, f"{folder_id}.jpg")
    if current_frame > 0:
        mid_idx = current_frame // 2
        mid_ts = mid_idx / fps
        target_frame_path = os.path.join(frames_dir, f"frame_{mid_idx:05d}_ts_{mid_ts:.3f}s.jpg")

        if os.path.exists(target_frame_path) and not os.path.exists(thumb_path):
            img = cv2.imread(target_frame_path)
            if img is not None:
                thumb_img = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
                cv2.imwrite(thumb_path, thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                log(f"🖼️ Thumbnail generated from frame #{mid_idx}")

    # ── Write to Database ──
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS videos
                     (msg_id INTEGER PRIMARY KEY, folder_id TEXT, title TEXT,
                      frames INTEGER, duration_sec REAL, duration_str TEXT,
                      thumb TEXT, first_frame TEXT, file_size_mb REAL, abs_path TEXT)''')

        c.execute('''INSERT OR REPLACE INTO videos
                     (msg_id, folder_id, title, frames, duration_sec, duration_str,
                      thumb, first_frame, file_size_mb, abs_path)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (msg_id, folder_id, f"Video #{msg_id}", current_frame, actual_duration,
                   f"{actual_duration:.1f}s", f"/thumbs/{folder_id}.jpg",
                   f"/data/{folder_id}/frame_00000_ts_0.000s.jpg", file_size_mb, frames_dir))
        conn.commit()
        conn.close()
        log(f"💾 DB committed: ID={msg_id} | {current_frame} frames | {actual_duration:.1f}s | {file_size_mb:.1f}MB")
    except Exception as e:
        raise RuntimeError(f"Database write failed: {e}")

    return current_frame


# ═══════════════════════════════════════════════════════════
# WORKER LOOP — Reliable claim → process → ack/fail
# ═══════════════════════════════════════════════════════════
def run_worker():
    log("═══════════════════════════════════════════════")
    log("🚀 CV FRAME EXTRACTION ENGINE — ONLINE")
    log(f"📂 Frame Output: {VIDEO_DIR}")
    log(f"🖼️ Thumbnails:   {THUMB_DIR}")
    log(f"🗄️ Database:     {DB_PATH}")
    log(f"📡 Queue:        {QUEUE_NAME} (reliable claim/ack/fail)")
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
            except:
                pass
            continue

        # Unwrap payload from job envelope
        payload = job.get("payload", job)
        msg_id = payload.get("msg_id")
        video_path = payload.get("path")
        retry_num = job.get("retries", 0)

        if not msg_id or not video_path:
            log(f"⚠️ Malformed job payload — acking to discard: {payload}")
            ack_job(QUEUE_NAME, job, job_raw)
            continue

        retry_tag = f" [RETRY #{retry_num}]" if retry_num > 0 else ""
        log(f"")
        log(f"{'='*50}")
        log(f"📥 JOB CLAIMED: Video #{msg_id}{retry_tag}")
        log(f"📁 Path: {video_path}")
        log(f"{'='*50}")

        try:
            frames_extracted = extract_video_data(video_path, msg_id)
            ack_job(QUEUE_NAME, job, job_raw)
            jobs_completed += 1
            total_frames += frames_extracted

            elapsed = time.time() - job.get("claimed_at", time.time())
            log(f"")
            log(f"🏁 JOB #{msg_id} COMPLETED — {frames_extracted} frames in {elapsed:.2f}s")
            log(f"📊 Worker Lifetime: {jobs_completed} completed | {jobs_failed} failed | {total_frames} total frames")
            log(f"")

        except Exception as e:
            result = fail_job(QUEUE_NAME, job, job_raw, str(e))
            jobs_failed += 1
            log(f"")
            log(f"❌ JOB #{msg_id} FAILED → {result}")
            log(f"   Error: {str(e)[:200]}")
            log(f"📊 Worker Lifetime: {jobs_completed} completed | {jobs_failed} failed | {total_frames} total frames")
            log(f"")
            time.sleep(2)  # Brief cooldown after failure


if __name__ == "__main__":
    run_worker()
