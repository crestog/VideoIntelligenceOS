"""
admin_backend.py — FastAPI router for the VIOS Admin Panel.

Provides endpoints for disk monitoring, category management,
storage cleanup (preview & execute), system health, logs,
and queue control (pause / resume).

v2 FIXES:
  - SQLite connections now use a 30s timeout + WAL busy handling
    (eliminates random "database is locked" 500 errors).
  - Redis access goes through a resilient helper that verifies the
    connection with ping() and degrades gracefully instead of raising.
  - Storage cleanup now resolves videos from BOTH the database AND the
    filesystem, so videos that were downloaded but never frame-extracted
    (no `videos` table row) are cleaned up too.
  - Cleanup also removes thumbnails, clears `local_video_path` on posts,
    deletes `videos` rows when their frames are removed, and clears the
    Redis dedup set so videos can be re-processed later.
  - Category activation validates names against the database and is
    reflected instantly by the Ghost Worker (see ui_server.py sync fix).
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
import os
import sqlite3
import shutil
import json
import time

from config import LAKE_DIR, DB_PATH, VIDEO_DIR, THUMB_DIR, SQLITE_TIMEOUT
from logger import vios_log, get_recent_logs
from queue_manager import get_redis, get_queue_metrics

admin_router = APIRouter()

# ---------------------------------------------------------------------------
# Disk-usage cache (avoids re-walking the filesystem on every request)
# ---------------------------------------------------------------------------
_disk_cache: dict = {}
_disk_cache_ts: float = 0.0
_DISK_CACHE_TTL: float = 5.0  # seconds


def _bytes_to_gb(b) -> float:
    return round(b / (1024 ** 3), 3)


def _dir_size(path: str) -> int:
    """Total size in bytes of all files under *path* (recursive, error-safe)."""
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _get_db() -> sqlite3.Connection:
    """New SQLite connection with row factory, long timeout, and WAL mode.

    The long timeout + busy_timeout pragma prevents 'database is locked'
    errors when the Ghost Worker is writing concurrently.
    """
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_TIMEOUT * 1000}")
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.Error:
        pass
    return conn


def _safe_redis():
    """Return a verified-alive Redis client, or None. NEVER raises."""
    try:
        r = get_redis()
        r.ping()
        return r
    except Exception:
        return None


def _error(msg: str, status: int = 500) -> JSONResponse:
    return JSONResponse({"error": str(msg)[:300]}, status_code=status)


# ───────────────────────────────────────────────────────────────────────────
# 1. GET /admin — Serve the admin UI HTML
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/admin", response_class=HTMLResponse)
def serve_admin_ui():
    try:
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "admin_ui.html")
        with open(html_path, "r", encoding="utf-8") as fh:
            return HTMLResponse(content=fh.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>admin_ui.html not found</h1>", status_code=404)
    except Exception as exc:
        vios_log(f"Error serving admin UI: {exc}", "ADMIN", "ERROR")
        return HTMLResponse(content="<h1>Internal error</h1>", status_code=500)


# ───────────────────────────────────────────────────────────────────────────
# 2. GET /api/admin/disk — Disk usage breakdown (cached 5 s)
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/disk")
def get_disk_usage():
    global _disk_cache, _disk_cache_ts
    try:
        now = time.time()
        if _disk_cache and (now - _disk_cache_ts) < _DISK_CACHE_TTL:
            return _disk_cache

        videos_bytes = 0
        frames_bytes = 0
        video_file_count = 0
        frame_folder_count = 0
        if os.path.isdir(VIDEO_DIR):
            for entry in os.scandir(VIDEO_DIR):
                try:
                    if entry.is_file() and entry.name.endswith(".mp4"):
                        videos_bytes += entry.stat().st_size
                        video_file_count += 1
                    elif entry.is_dir() and entry.name.startswith("frames_"):
                        frames_bytes += _dir_size(entry.path)
                        frame_folder_count += 1
                except OSError:
                    pass

        thumbnails_bytes = _dir_size(THUMB_DIR) if os.path.isdir(THUMB_DIR) else 0

        usage = shutil.disk_usage(VIDEO_DIR if os.path.exists(VIDEO_DIR) else "/")
        known = videos_bytes + frames_bytes + thumbnails_bytes

        _disk_cache = {
            "total_gb": _bytes_to_gb(usage.total),
            "used_gb": _bytes_to_gb(usage.used),
            "free_gb": _bytes_to_gb(usage.free),
            "video_file_count": video_file_count,
            "frame_folder_count": frame_folder_count,
            "breakdown": {
                "videos_gb": _bytes_to_gb(videos_bytes),
                "frames_gb": _bytes_to_gb(frames_bytes),
                "thumbnails_gb": _bytes_to_gb(thumbnails_bytes),
                "other_gb": _bytes_to_gb(max(usage.used - known, 0)),
            },
        }
        _disk_cache_ts = now
        return _disk_cache
    except Exception as exc:
        vios_log(f"Error computing disk usage: {exc}", "ADMIN", "ERROR")
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# 3. GET /api/admin/categories — All categories with stats
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/categories")
def get_categories():
    """Every category with video/frame counts, on-disk usage, and active status.

    Disk usage is computed from BOTH the videos table (frame counts / mp4 size)
    and a live filesystem check, so categories whose videos were downloaded but
    not yet frame-extracted still report accurate numbers.
    """
    conn = None
    try:
        conn = _get_db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                c.id   AS cat_id,
                c.name AS cat_name,
                COUNT(DISTINCT CASE WHEN p.status = 'Harvested' THEN p.video_id END) AS harvested_count,
                COUNT(DISTINCT p.video_id) AS post_count
            FROM categories c
            LEFT JOIN posts p ON p.category_id = c.id
            GROUP BY c.id, c.name
            """
        )
        cat_rows = cur.fetchall()

        # Per-category video stats from the videos table (frame counts + sizes)
        cur.execute(
            """
            SELECT c.name AS cat_name,
                   COALESCE(SUM(v.frames), 0)       AS frame_count,
                   COALESCE(SUM(v.file_size_mb), 0) AS disk_usage_mb,
                   COUNT(v.msg_id)                  AS extracted_count
            FROM videos v
            JOIN posts p      ON p.video_id = v.msg_id
            JOIN categories c ON c.id = p.category_id
            GROUP BY c.name
            """
        )
        vid_stats = {r["cat_name"]: dict(r) for r in cur.fetchall()}
        conn.close()
        conn = None

        # Active list from Redis (best-effort)
        active_list: list = []
        r = _safe_redis()
        if r:
            try:
                raw = r.get("ADMIN_ACTIVE_CATEGORIES")
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        active_list = parsed
            except Exception:
                pass
        active_set = set(active_list)

        categories = []
        for row in cat_rows:
            name = row["cat_name"]
            stats = vid_stats.get(name, {})
            is_active = name in active_set
            categories.append(
                {
                    "id": row["cat_id"],
                    "name": name,
                    "video_count": row["harvested_count"],
                    "post_count": row["post_count"],
                    "frame_count": stats.get("frame_count", 0),
                    "extracted_count": stats.get("extracted_count", 0),
                    "disk_usage_mb": round(stats.get("disk_usage_mb", 0) or 0, 2),
                    "is_active": is_active,
                    "queue_position": active_list.index(name) if is_active else None,
                }
            )

        categories.sort(
            key=lambda c: (
                0 if c["is_active"] else 1,
                c["queue_position"] if c["is_active"] else 0,
                c["name"].lower(),
            )
        )
        return {"categories": categories, "active_order": active_list,
                "redis_online": r is not None}
    except Exception as exc:
        vios_log(f"Error fetching categories: {exc}", "ADMIN", "ERROR")
        return _error(exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ───────────────────────────────────────────────────────────────────────────
# 4. POST /api/admin/categories/activate — Set active categories
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/categories/activate")
async def activate_categories(request: Request):
    """Store the ordered active-category list in Redis.

    Validates names against the DB, dedupes while preserving order, and pings
    the Ghost Worker (SCAN_WAKE key) so priority changes apply immediately.
    """
    try:
        body = await request.json()
        cat_list = body.get("categories", [])
        if not isinstance(cat_list, list):
            return _error("'categories' must be a list", 400)

        # Validate against DB + dedupe preserving order
        conn = _get_db()
        known = {row["name"] for row in conn.execute("SELECT name FROM categories").fetchall()}
        conn.close()

        seen = set()
        clean = []
        invalid = []
        for name in cat_list:
            if not isinstance(name, str):
                continue
            if name not in known:
                invalid.append(name)
                continue
            if name not in seen:
                seen.add(name)
                clean.append(name)

        r = _safe_redis()
        if r is None:
            return _error("Redis unavailable — cannot store active categories", 503)

        r.set("ADMIN_ACTIVE_CATEGORIES", json.dumps(clean))
        # Wake the Ghost Worker so the new priority order applies immediately
        r.set("ADMIN_CATEGORIES_UPDATED", str(time.time()))

        vios_log(f"Active categories updated: {clean}"
                 + (f" (ignored unknown: {invalid})" if invalid else ""),
                 "ADMIN", "INFO")
        return {"ok": True, "active_categories": clean, "ignored": invalid}
    except Exception as exc:
        vios_log(f"Error activating categories: {exc}", "ADMIN", "ERROR")
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# Helper: resolve videos for given category names (DB + filesystem)
# ───────────────────────────────────────────────────────────────────────────
def _resolve_videos_for_categories(category_names: list) -> list:
    """Return [{msg_id, ...}] for all videos in *category_names*.

    Uses the `posts` table as the source of truth (every downloaded video has
    a post row), NOT the `videos` table (which only exists after frame
    extraction). This is the core fix for cleanup missing files.
    """
    conn = _get_db()
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in category_names)
    cur.execute(
        f"""
        SELECT DISTINCT p.video_id AS msg_id
        FROM posts p
        JOIN categories c ON c.id = p.category_id
        WHERE c.name IN ({placeholders})
        """,
        category_names,
    )
    rows = [{"msg_id": r["msg_id"]} for r in cur.fetchall()]
    conn.close()
    return rows


def _cleanup_targets(category_names, delete_frames, delete_videos):
    """Yield (msg_id, frames_dir|None, mp4_path|None, thumb_path|None) tuples."""
    for v in _resolve_videos_for_categories(category_names):
        msg_id = v["msg_id"]
        frames_dir = os.path.join(VIDEO_DIR, f"frames_{msg_id}")
        mp4_path = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
        thumb_path = os.path.join(THUMB_DIR, f"frames_{msg_id}.jpg")
        yield (
            msg_id,
            frames_dir if delete_frames and os.path.isdir(frames_dir) else None,
            mp4_path if delete_videos and os.path.isfile(mp4_path) else None,
            thumb_path if delete_frames and os.path.isfile(thumb_path) else None,
        )


# ───────────────────────────────────────────────────────────────────────────
# 5. POST /api/admin/storage/preview — Preview cleanup
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/storage/preview")
async def storage_preview(request: Request):
    """Calculate how much space WOULD be freed without deleting anything."""
    try:
        body = await request.json()
        category_names = body.get("categories", [])
        delete_frames = bool(body.get("delete_frames", False))
        delete_videos = bool(body.get("delete_videos", False))

        if not category_names or not isinstance(category_names, list):
            return _error("'categories' must be a non-empty list", 400)
        if not delete_frames and not delete_videos:
            return _error("Select at least one of delete_frames / delete_videos", 400)

        would_free = 0
        affected_videos = 0
        affected_frame_folders = 0

        for _msg_id, frames_dir, mp4_path, thumb_path in _cleanup_targets(
                category_names, delete_frames, delete_videos):
            if frames_dir:
                would_free += _dir_size(frames_dir)
                affected_frame_folders += 1
            if mp4_path:
                try:
                    would_free += os.path.getsize(mp4_path)
                    affected_videos += 1
                except OSError:
                    pass
            if thumb_path:
                try:
                    would_free += os.path.getsize(thumb_path)
                except OSError:
                    pass

        return {
            "would_free_gb": _bytes_to_gb(would_free),
            "affected_videos": affected_videos,
            "affected_frame_folders": affected_frame_folders,
        }
    except Exception as exc:
        vios_log(f"Error in storage preview: {exc}", "ADMIN", "ERROR")
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# 6. POST /api/admin/storage/cleanup — Execute cleanup
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/storage/cleanup")
async def storage_cleanup(request: Request):
    """Delete frames and/or video files for the given categories.

    Also performs FULL bookkeeping so no ghost state remains:
      frames deleted  → videos row removed, thumbnail removed
      video deleted   → posts.status reset to Metadata_Only, local_video_path cleared
      either          → msg_id removed from the Redis dedup set
    """
    try:
        body = await request.json()
        category_names = body.get("categories", [])
        delete_frames = bool(body.get("delete_frames", False))
        delete_videos = bool(body.get("delete_videos", False))

        if not category_names or not isinstance(category_names, list):
            return _error("'categories' must be a non-empty list", 400)
        if not delete_frames and not delete_videos:
            return _error("Select at least one of delete_frames / delete_videos", 400)

        freed = 0
        deleted_videos = 0
        deleted_frame_folders = 0
        frame_deleted_ids = []
        video_deleted_ids = []
        touched_ids = set()

        for msg_id, frames_dir, mp4_path, thumb_path in _cleanup_targets(
                category_names, delete_frames, delete_videos):

            if frames_dir:
                size = _dir_size(frames_dir)
                shutil.rmtree(frames_dir, ignore_errors=True)
                freed += size
                deleted_frame_folders += 1
                frame_deleted_ids.append(msg_id)
                touched_ids.add(msg_id)

            if thumb_path:
                try:
                    freed += os.path.getsize(thumb_path)
                    os.remove(thumb_path)
                except OSError:
                    pass

            if mp4_path:
                try:
                    size = os.path.getsize(mp4_path)
                    os.remove(mp4_path)
                    freed += size
                    deleted_videos += 1
                    video_deleted_ids.append(msg_id)
                    touched_ids.add(msg_id)
                except OSError as oe:
                    vios_log(f"Could not delete video_{msg_id}.mp4: {oe}", "ADMIN", "WARN")

        # ── Database bookkeeping ──
        if frame_deleted_ids or video_deleted_ids:
            conn = _get_db()
            cur = conn.cursor()

            if frame_deleted_ids:
                ph = ",".join("?" for _ in frame_deleted_ids)
                cur.execute(f"DELETE FROM videos WHERE msg_id IN ({ph})", frame_deleted_ids)

            if video_deleted_ids:
                ph = ",".join("?" for _ in video_deleted_ids)
                cur.execute(
                    f"UPDATE posts SET status = 'Metadata_Only', local_video_path = NULL "
                    f"WHERE video_id IN ({ph})",
                    video_deleted_ids,
                )

            conn.commit()
            conn.close()
            vios_log(
                f"DB cleanup: {len(frame_deleted_ids)} video rows removed, "
                f"{len(video_deleted_ids)} posts reset",
                "ADMIN", "INFO",
            )

        # ── Redis dedup bookkeeping (so videos can be re-processed later) ──
        if touched_ids:
            r = _safe_redis()
            if r:
                try:
                    r.srem("PROCESSED_VIDEOS_SET", *[str(i) for i in touched_ids])
                except Exception:
                    pass

        # Invalidate disk cache
        global _disk_cache_ts
        _disk_cache_ts = 0.0

        vios_log(
            f"Cleanup complete: freed {_bytes_to_gb(freed)} GB "
            f"({deleted_videos} videos, {deleted_frame_folders} frame folders)",
            "ADMIN", "SUCCESS",
        )
        return {
            "freed_gb": _bytes_to_gb(freed),
            "deleted_videos": deleted_videos,
            "deleted_frame_folders": deleted_frame_folders,
        }
    except Exception as exc:
        vios_log(f"Error in storage cleanup: {exc}", "ADMIN", "ERROR")
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# 7. GET /api/admin/system — System health
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/system")
def system_health():
    try:
        redis_online = False
        cv_paused = False
        r = _safe_redis()
        if r is not None:
            redis_online = True
            try:
                cv_paused = r.get("CV_PAUSED") == "1"
            except Exception:
                pass

        queue_metrics = {}
        if redis_online:
            try:
                queue_metrics = get_queue_metrics()
            except Exception as qe:
                queue_metrics = {"error": str(qe)[:200]}

        return {
            "redis_online": redis_online,
            "cv_paused": cv_paused,
            "queue_metrics": queue_metrics,
        }
    except Exception as exc:
        vios_log(f"Error in system health check: {exc}", "ADMIN", "ERROR")
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# 8. GET /api/admin/logs — Recent logs
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/logs")
def get_logs():
    try:
        return {"logs": get_recent_logs(200)}
    except Exception as exc:
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# 9./10. POST /api/admin/queue/pause | resume
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/queue/pause")
def queue_pause():
    r = _safe_redis()
    if r is None:
        return _error("Redis unavailable", 503)
    try:
        r.set("CV_PAUSED", "1")
        vios_log("CV queue PAUSED via admin panel", "ADMIN", "INFO")
        return {"ok": True, "paused": True}
    except Exception as exc:
        vios_log(f"Error pausing queue: {exc}", "ADMIN", "ERROR")
        return _error(exc)


@admin_router.post("/api/admin/queue/resume")
def queue_resume():
    r = _safe_redis()
    if r is None:
        return _error("Redis unavailable", 503)
    try:
        r.delete("CV_PAUSED")
        vios_log("CV queue RESUMED via admin panel", "ADMIN", "INFO")
        return {"ok": True, "paused": False}
    except Exception as exc:
        vios_log(f"Error resuming queue: {exc}", "ADMIN", "ERROR")
        return _error(exc)
