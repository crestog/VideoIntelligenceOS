"""
admin_backend.py — FastAPI router for the VIOS Admin Panel.

Provides endpoints for disk monitoring, category management,
storage cleanup (preview & execute), system health, logs,
and queue control (pause / resume).

All endpoints are synchronous and use sqlite3 directly.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
import os
import sqlite3
import shutil
import json
import time
import glob
import pathlib

from config import BASE_DIR, LAKE_DIR, DB_PATH, VIDEO_DIR, THUMB_DIR, FLAG_DIR
from logger import vios_log, get_recent_logs
from queue_manager import get_redis, get_queue_metrics

admin_router = APIRouter()

# ---------------------------------------------------------------------------
# Disk-usage cache (avoids re-walking the filesystem on every request)
# ---------------------------------------------------------------------------
_disk_cache: dict = {}
_disk_cache_ts: float = 0.0
_DISK_CACHE_TTL: float = 5.0  # seconds


def _bytes_to_gb(b: int | float) -> float:
    """Convert bytes to gigabytes, rounded to 3 decimals."""
    return round(b / (1024 ** 3), 3)


def _dir_size(path: str) -> int:
    """Return total size in bytes of all files under *path* (recursive)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _get_db() -> sqlite3.Connection:
    """Return a new SQLite connection with row-factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ───────────────────────────────────────────────────────────────────────────
# 1. GET /admin — Serve the admin UI HTML
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/admin", response_class=HTMLResponse)
def serve_admin_ui():
    """Serve the admin_ui.html file located next to this module."""
    try:
        html_path = os.path.join(os.path.dirname(__file__), "admin_ui.html")
        with open(html_path, "r", encoding="utf-8") as fh:
            return HTMLResponse(content=fh.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>admin_ui.html not found</h1>", status_code=404
        )
    except Exception as exc:
        vios_log(f"Error serving admin UI: {exc}", "admin", "error")
        return HTMLResponse(
            content=f"<h1>Internal error</h1><pre>{exc}</pre>", status_code=500
        )


# ───────────────────────────────────────────────────────────────────────────
# 2. GET /api/admin/disk — Disk usage breakdown (cached 5 s)
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/disk")
def get_disk_usage():
    """Return disk usage breakdown with a 5-second cache."""
    global _disk_cache, _disk_cache_ts

    try:
        now = time.time()
        if _disk_cache and (now - _disk_cache_ts) < _DISK_CACHE_TTL:
            return _disk_cache

        # Video files (*.mp4) directly in VIDEO_DIR
        videos_bytes = 0
        frames_bytes = 0
        for entry in os.scandir(VIDEO_DIR):
            if entry.is_file() and entry.name.endswith(".mp4"):
                try:
                    videos_bytes += entry.stat().st_size
                except OSError:
                    pass
            elif entry.is_dir() and entry.name.startswith("frames_"):
                frames_bytes += _dir_size(entry.path)

        thumbnails_bytes = _dir_size(THUMB_DIR) if os.path.isdir(THUMB_DIR) else 0

        # Total / free from the filesystem containing VIDEO_DIR
        usage = shutil.disk_usage(VIDEO_DIR if os.path.exists(VIDEO_DIR) else "/")
        total_bytes = usage.total
        used_bytes = usage.used
        free_bytes = usage.free

        known = videos_bytes + frames_bytes + thumbnails_bytes
        other_bytes = max(used_bytes - known, 0)

        _disk_cache = {
            "total_gb": _bytes_to_gb(total_bytes),
            "used_gb": _bytes_to_gb(used_bytes),
            "free_gb": _bytes_to_gb(free_bytes),
            "breakdown": {
                "videos_gb": _bytes_to_gb(videos_bytes),
                "frames_gb": _bytes_to_gb(frames_bytes),
                "thumbnails_gb": _bytes_to_gb(thumbnails_bytes),
                "other_gb": _bytes_to_gb(other_bytes),
            },
        }
        _disk_cache_ts = now
        return _disk_cache

    except Exception as exc:
        vios_log(f"Error computing disk usage: {exc}", "admin", "error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ───────────────────────────────────────────────────────────────────────────
# 3. GET /api/admin/categories — All categories with stats
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/categories")
def get_categories():
    """Return every category with video count, frame count, disk usage, and active status."""
    try:
        conn = _get_db()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                c.id                                    AS cat_id,
                c.name                                  AS cat_name,
                COUNT(DISTINCT CASE WHEN p.status = 'Harvested' THEN p.video_id END) AS video_count,
                COALESCE(SUM(v.frames), 0)              AS frame_count,
                COALESCE(SUM(v.file_size_mb), 0)        AS disk_usage_mb
            FROM categories c
            LEFT JOIN posts p   ON p.category_id = c.id
            LEFT JOIN videos v  ON v.msg_id = p.video_id AND p.status = 'Harvested'
            GROUP BY c.id, c.name
            """
        )
        rows = cur.fetchall()
        conn.close()

        # Fetch active list from Redis
        r = get_redis()
        active_raw = r.get("ADMIN_ACTIVE_CATEGORIES") if r else None
        active_list: list[str] = json.loads(active_raw) if active_raw else []
        active_set = set(active_list)

        categories = []
        for row in rows:
            name = row["cat_name"]
            is_active = name in active_set
            queue_position = active_list.index(name) if is_active else None
            categories.append(
                {
                    "id": row["cat_id"],
                    "name": name,
                    "video_count": row["video_count"],
                    "frame_count": row["frame_count"],
                    "disk_usage_mb": round(row["disk_usage_mb"], 2),
                    "is_active": is_active,
                    "queue_position": queue_position,
                }
            )

        # Sort: active first (by queue_position), then inactive alphabetically
        categories.sort(
            key=lambda c: (
                0 if c["is_active"] else 1,
                c["queue_position"] if c["is_active"] else 0,
                c["name"].lower() if not c["is_active"] else "",
            )
        )
        return {"categories": categories}

    except Exception as exc:
        vios_log(f"Error fetching categories: {exc}", "admin", "error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ───────────────────────────────────────────────────────────────────────────
# 4. POST /api/admin/categories/activate — Set active categories
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/categories/activate")
async def activate_categories(request: Request):
    """Store the ordered list of active categories in Redis."""
    try:
        body = await request.json()

        cat_list = body.get("categories", [])
        if not isinstance(cat_list, list):
            return JSONResponse(
                {"error": "'categories' must be a list"}, status_code=400
            )

        r = get_redis()
        if r is None:
            return JSONResponse({"error": "Redis unavailable"}, status_code=503)

        r.set("ADMIN_ACTIVE_CATEGORIES", json.dumps(cat_list))
        vios_log(
            f"Active categories updated: {cat_list}", "admin", "info"
        )
        return {"ok": True, "active_categories": cat_list}

    except Exception as exc:
        vios_log(f"Error activating categories: {exc}", "admin", "error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ───────────────────────────────────────────────────────────────────────────
# Helper: resolve videos for given category names
# ───────────────────────────────────────────────────────────────────────────
def _resolve_videos_for_categories(category_names: list[str]) -> list[dict]:
    """
    Return a list of dicts with keys (msg_id, file_size_mb) for all
    videos whose posts belong to *category_names*.
    """
    conn = _get_db()
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in category_names)
    cur.execute(
        f"""
        SELECT v.msg_id, v.file_size_mb
        FROM videos v
        JOIN posts p ON p.video_id = v.msg_id
        JOIN categories c ON c.id = p.category_id
        WHERE c.name IN ({placeholders})
        """,
        category_names,
    )
    rows = [{"msg_id": r["msg_id"], "file_size_mb": r["file_size_mb"]} for r in cur.fetchall()]
    conn.close()
    return rows


# ───────────────────────────────────────────────────────────────────────────
# 5. POST /api/admin/storage/preview — Preview cleanup
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/storage/preview")
async def storage_preview(request: Request):
    """Calculate how much space *would* be freed without deleting anything."""
    try:
        body = await request.json()

        category_names: list[str] = body.get("categories", [])
        delete_frames: bool = body.get("delete_frames", False)
        delete_videos: bool = body.get("delete_videos", False)

        if not category_names:
            return JSONResponse(
                {"error": "'categories' must be a non-empty list"}, status_code=400
            )

        videos = _resolve_videos_for_categories(category_names)

        would_free = 0
        affected_videos = 0
        affected_frame_folders = 0

        for v in videos:
            msg_id = v["msg_id"]

            if delete_frames:
                frames_dir = os.path.join(VIDEO_DIR, f"frames_{msg_id}")
                if os.path.isdir(frames_dir):
                    would_free += _dir_size(frames_dir)
                    affected_frame_folders += 1

            if delete_videos:
                mp4_path = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
                if os.path.isfile(mp4_path):
                    would_free += os.path.getsize(mp4_path)
                    affected_videos += 1

        return {
            "would_free_gb": _bytes_to_gb(would_free),
            "affected_videos": affected_videos,
            "affected_frame_folders": affected_frame_folders,
        }

    except Exception as exc:
        vios_log(f"Error in storage preview: {exc}", "admin", "error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ───────────────────────────────────────────────────────────────────────────
# 6. POST /api/admin/storage/cleanup — Execute cleanup
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/storage/cleanup")
async def storage_cleanup(request: Request):
    """Delete frames and/or video files for the given categories."""
    try:
        body = await request.json()

        category_names: list[str] = body.get("categories", [])
        delete_frames: bool = body.get("delete_frames", False)
        delete_videos: bool = body.get("delete_videos", False)

        if not category_names:
            return JSONResponse(
                {"error": "'categories' must be a non-empty list"}, status_code=400
            )

        videos = _resolve_videos_for_categories(category_names)

        freed = 0
        deleted_videos = 0
        deleted_frame_folders = 0
        deleted_msg_ids: list[int] = []

        for v in videos:
            msg_id = v["msg_id"]

            # --- frames ---
            if delete_frames:
                frames_dir = os.path.join(VIDEO_DIR, f"frames_{msg_id}")
                if os.path.isdir(frames_dir):
                    size = _dir_size(frames_dir)
                    shutil.rmtree(frames_dir, ignore_errors=True)
                    freed += size
                    deleted_frame_folders += 1
                    vios_log(
                        f"Deleted frames dir: frames_{msg_id} ({_bytes_to_gb(size)} GB)",
                        "admin",
                        "info",
                    )

            # --- video mp4 ---
            if delete_videos:
                mp4_path = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
                if os.path.isfile(mp4_path):
                    size = os.path.getsize(mp4_path)
                    os.remove(mp4_path)
                    freed += size
                    deleted_videos += 1
                    deleted_msg_ids.append(msg_id)
                    vios_log(
                        f"Deleted video: video_{msg_id}.mp4 ({_bytes_to_gb(size)} GB)",
                        "admin",
                        "info",
                    )

        # --- Database cleanup ---
        if deleted_msg_ids:
            conn = _get_db()
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in deleted_msg_ids)

            # Remove from videos table
            cur.execute(
                f"DELETE FROM videos WHERE msg_id IN ({placeholders})",
                deleted_msg_ids,
            )

            # Reset post status
            cur.execute(
                f"UPDATE posts SET status = 'Metadata_Only' WHERE video_id IN ({placeholders})",
                deleted_msg_ids,
            )
            conn.commit()
            conn.close()
            vios_log(
                f"DB cleanup: removed {len(deleted_msg_ids)} video rows, reset post statuses",
                "admin",
                "info",
            )

        # Invalidate disk cache
        global _disk_cache_ts
        _disk_cache_ts = 0.0

        return {
            "freed_gb": _bytes_to_gb(freed),
            "deleted_videos": deleted_videos,
            "deleted_frame_folders": deleted_frame_folders,
        }

    except Exception as exc:
        vios_log(f"Error in storage cleanup: {exc}", "admin", "error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ───────────────────────────────────────────────────────────────────────────
# 7. GET /api/admin/system — System health
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/system")
def system_health():
    """Check Redis connectivity, queue metrics, and CV pause state."""
    try:
        redis_online = False
        cv_paused = False
        try:
            r = get_redis()
            if r is not None:
                r.ping()
                redis_online = True
                cv_paused = r.get("CV_PAUSED") == "1"
        except Exception:
            redis_online = False

        queue_metrics = {}
        try:
            queue_metrics = get_queue_metrics()
        except Exception as qe:
            queue_metrics = {"error": str(qe)}

        return {
            "redis_online": redis_online,
            "cv_paused": cv_paused,
            "queue_metrics": queue_metrics,
        }

    except Exception as exc:
        vios_log(f"Error in system health check: {exc}", "admin", "error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ───────────────────────────────────────────────────────────────────────────
# 8. GET /api/admin/logs — Recent logs
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/logs")
def get_logs():
    """Return the most recent 200 log entries."""
    try:
        logs = get_recent_logs(200)
        return {"logs": logs}
    except Exception as exc:
        vios_log(f"Error fetching logs: {exc}", "admin", "error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ───────────────────────────────────────────────────────────────────────────
# 9. POST /api/admin/queue/pause — Pause the CV queue
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/queue/pause")
def queue_pause():
    """Set Redis key CV_PAUSED = '1' to pause processing."""
    try:
        r = get_redis()
        if r is None:
            return JSONResponse({"error": "Redis unavailable"}, status_code=503)
        r.set("CV_PAUSED", "1")
        vios_log("CV queue PAUSED via admin panel", "admin", "info")
        return {"ok": True, "paused": True}
    except Exception as exc:
        vios_log(f"Error pausing queue: {exc}", "admin", "error")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ───────────────────────────────────────────────────────────────────────────
# 10. POST /api/admin/queue/resume — Resume the CV queue
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/queue/resume")
def queue_resume():
    """Delete Redis key CV_PAUSED to resume processing."""
    try:
        r = get_redis()
        if r is None:
            return JSONResponse({"error": "Redis unavailable"}, status_code=503)
        r.delete("CV_PAUSED")
        vios_log("CV queue RESUMED via admin panel", "admin", "info")
        return {"ok": True, "paused": False}
    except Exception as exc:
        vios_log(f"Error resuming queue: {exc}", "admin", "error")
        return JSONResponse({"error": str(exc)}, status_code=500)
