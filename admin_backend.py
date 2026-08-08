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

from config import (LAKE_DIR, DB_PATH, VIDEO_DIR, THUMB_DIR, SQLITE_TIMEOUT,
                    BASE_DIR, SCRATCH_DIR, MODEL_CACHE_DIR, QDRANT_PATH,
                    NEO4J_DATA_DIR, ARCHIVE_DIR)
from logger import vios_log, get_recent_logs
from queue_manager import get_redis, get_queue_metrics

admin_router = APIRouter()

# ---------------------------------------------------------------------------
# Disk-usage cache (avoids re-walking the filesystem on every request)
# The walk now covers the model cache and vector store too — tens of thousands
# of files — so the TTL is longer than the original 5 s.
# ---------------------------------------------------------------------------
_disk_cache: dict = {}
_disk_cache_ts: float = 0.0
_DISK_CACHE_TTL: float = 20.0  # seconds


def _bytes_to_gb(b) -> float:
    return round(b / (1024 ** 3), 3)


def _dir_size(path: str, _seen: set = None) -> int:
    """Bytes of real storage consumed under *path*.

    Two things this must not do, both of which the previous version did:

      * Follow symlinks. `os.path.getsize` stats the TARGET, so a HuggingFace
        cache — which stores every weight once in `blobs/` and then points at
        it from `snapshots/<rev>/<name>` — had each blob counted once as the
        real file and again for every snapshot referencing it. That is why the
        panel claimed 54 GB of model cache over roughly 5.5 GB of weights.
      * Count hardlinked files more than once, for the same reason.

    Both are handled by stat'ing without following links and skipping any
    (device, inode) pair already counted. Symlinks themselves contribute
    nothing: their target is counted wherever it actually lives.

    `os.scandir` is used over `os.walk` because it returns stat data from the
    directory read itself — on a cache with tens of thousands of entries that
    is roughly an order of magnitude fewer syscalls.
    """
    if _seen is None:
        _seen = set()
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue          # pointer, not storage
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                            continue
                        st = entry.stat(follow_symlinks=False)
                        # Windows serves DirEntry.stat() from the directory
                        # listing and leaves st_ino/st_dev at 0, so hardlinks
                        # would go undetected. Pay for one real stat in that
                        # case; on Linux (the deployment target) st_ino is
                        # already populated and this never fires.
                        if not st.st_ino:
                            try:
                                st = os.stat(entry.path, follow_symlinks=False)
                            except OSError:
                                pass
                        key = (st.st_dev, st.st_ino)
                        if st.st_ino:
                            if key in _seen:
                                continue
                            _seen.add(key)
                        total += st.st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def _free_bytes(path: str) -> int:
    """Free bytes on the mount holding *path*, walking up to the first real
    directory (the path may not exist yet on a cold boot)."""
    try:
        probe = path
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def _host_usage() -> dict:
    """The physical mount's own numbers, kept strictly separate from VIOS's
    footprint. On shared infrastructure most of `used` belongs to other
    tenants, so this is context for headroom — never a VIOS measurement."""
    try:
        probe = SCRATCH_DIR if os.path.isdir(SCRATCH_DIR) else "/"
        usage = shutil.disk_usage(probe)
        return {"total_gb": _bytes_to_gb(usage.total),
                "used_gb": _bytes_to_gb(usage.used),
                "free_gb": _bytes_to_gb(usage.free),
                "path": probe}
    except OSError:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "path": "?"}


# Kaggle caps /kaggle/working at 19.5 GiB (20 GB) regardless of what the
# underlying mount reports. shutil.disk_usage() sees the HOST filesystem —
# roughly 8 TB shared across tenants — so "free space" on the output tier was
# off by three orders of magnitude and could never warn before the quota hit.
# That quota is what actually kills a session ("No space left on device"), so
# it is tracked explicitly. Override with VIOS_OUTPUT_QUOTA_GB when running
# somewhere with different limits; 0 disables quota tracking.
_OUTPUT_QUOTA_BYTES = int(
    float(os.environ.get("VIOS_OUTPUT_QUOTA_GB", "19.5")) * (1024 ** 3))


def _output_quota(used_bytes: int) -> dict:
    """Remaining OUTPUT budget: the quota, not the host mount's free space."""
    if _OUTPUT_QUOTA_BYTES <= 0:
        return {"tracked": False}
    free = max(0, _OUTPUT_QUOTA_BYTES - used_bytes)
    # Never promise more room than the physical mount actually has.
    physical_free = _free_bytes(BASE_DIR)
    effective = min(free, physical_free) if physical_free else free
    pct = (used_bytes / _OUTPUT_QUOTA_BYTES * 100) if _OUTPUT_QUOTA_BYTES else 0
    return {"tracked": True,
            "quota_gb": _bytes_to_gb(_OUTPUT_QUOTA_BYTES),
            "used_gb": _bytes_to_gb(used_bytes),
            "free_gb": _bytes_to_gb(effective),
            "pct_used": round(pct, 1),
            "state": "critical" if pct >= 90 else "warn" if pct >= 75 else "ok"}


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
# 2. GET /api/admin/disk — VIOS's own storage footprint (cached)
#
# This reports what VIOS is using, NOT what the host disk holds. The previous
# version took shutil.disk_usage(VIDEO_DIR).used as the headline, which on
# Kaggle is the whole /kaggle mount — every other tenant's data included. It
# read "7075 GB used of 8062 GB" with "Other 7065 GB", i.e. 99.9% of the panel
# was somebody else's bytes and the four real numbers were invisible next to
# them. `used` is now the sum of the directories VIOS actually writes, and
# host free space is reported separately as headroom, clearly labelled.
# ───────────────────────────────────────────────────────────────────────────
_DISK_CATEGORIES = (
    # (key, label, path, tier)  — tier is where the bytes live
    ("videos",     "Videos",      None,             "scratch"),  # computed below
    ("frames",     "Frames",      None,             "scratch"),  # computed below
    ("thumbnails", "Thumbnails",  THUMB_DIR,        "scratch"),
    ("models",     "Model cache", MODEL_CACHE_DIR,  "scratch"),
    ("qdrant",     "Vectors",     QDRANT_PATH,      "scratch"),
    ("neo4j",      "Graph",       NEO4J_DATA_DIR,   "scratch"),
    ("archive",    "Omni archive", ARCHIVE_DIR,     "scratch"),
    ("datalake",   "Data lake",   LAKE_DIR,         "output"),
    # Export bundles kept with "keep a local copy". They sit in the output tier
    # and so consume the same 19.5 GB quota the data lake does — invisible bytes
    # here would show up as an unexplained quota drop.
    ("exports",    "DB exports",  os.path.join(BASE_DIR, "exports"), "output"),
)


@admin_router.get("/api/admin/disk")
def get_disk_usage(refresh: int = 0):
    """VIOS's own storage footprint, per tier.

    `refresh=1` bypasses the TTL cache — the panel's Refresh button was
    otherwise a no-op for up to 20 seconds, which read as a frozen UI.
    """
    global _disk_cache, _disk_cache_ts
    try:
        now = time.time()
        if not refresh and _disk_cache and (now - _disk_cache_ts) < _DISK_CACHE_TTL:
            return dict(_disk_cache, cached=True,
                        age_s=round(now - _disk_cache_ts, 1))

        videos_bytes = 0
        frames_bytes = 0
        video_file_count = 0
        frame_folder_count = 0
        # One inode ledger for the whole sweep. Categories can overlap on disk
        # (LAKE_DIR nested under BASE_DIR, a blob hardlinked into two caches),
        # and counting a file under two categories would inflate the total the
        # same way following symlinks did. First category to reach a byte owns
        # it, so the breakdown always sums to the headline.
        seen_inodes: set = set()
        if os.path.isdir(VIDEO_DIR):
            for entry in os.scandir(VIDEO_DIR):
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_file() and entry.name.endswith(".mp4"):
                        st = entry.stat(follow_symlinks=False)
                        key = (st.st_dev, st.st_ino)
                        if st.st_ino and key in seen_inodes:
                            continue
                        if st.st_ino:
                            seen_inodes.add(key)
                        videos_bytes += st.st_size
                        video_file_count += 1
                    elif entry.is_dir() and entry.name.startswith("frames_"):
                        frames_bytes += _dir_size(entry.path, seen_inodes)
                        frame_folder_count += 1
                except OSError:
                    pass

        sizes = {"videos": videos_bytes, "frames": frames_bytes}
        for key, _label, path, _tier in _DISK_CATEGORIES:
            if key in sizes:
                continue
            sizes[key] = (_dir_size(path, seen_inodes)
                          if path and os.path.isdir(path) else 0)

        vios_bytes = sum(sizes.values())

        # Free space per tier: the OUTPUT quota and the SCRATCH pool are
        # different mounts with different lifetimes, so a single "free" number
        # can't answer "will the next download fit?". Both are reported.
        tier_bytes = {"scratch": 0, "output": 0}
        for key, _label, _path, tier in _DISK_CATEGORIES:
            tier_bytes[tier] = tier_bytes.get(tier, 0) + sizes.get(key, 0)

        _disk_cache = {
            # Headline: VIOS's footprint, not the host's.
            "vios_used_gb": _bytes_to_gb(vios_bytes),
            "video_file_count": video_file_count,
            "frame_folder_count": frame_folder_count,
            "breakdown": {f"{k}_gb": _bytes_to_gb(v) for k, v in sizes.items()},
            "labels": {k: label for k, label, _p, _t in _DISK_CATEGORIES},
            "tiers": {
                "scratch": {"path": SCRATCH_DIR,
                            "free_gb": _bytes_to_gb(_free_bytes(SCRATCH_DIR)),
                            "used_gb": _bytes_to_gb(tier_bytes["scratch"]),
                            "note": "ephemeral — cleared when the session ends"},
                "output": {"path": BASE_DIR,
                           # Quota-aware: the host mount has terabytes free but
                           # Kaggle stops writes at the quota, so that is the
                           # number the panel must show.
                           "free_gb": _output_quota(tier_bytes["output"]).get(
                               "free_gb", _bytes_to_gb(_free_bytes(BASE_DIR))),
                           "used_gb": _bytes_to_gb(tier_bytes["output"]),
                           "quota": _output_quota(tier_bytes["output"]),
                           "note": "kept across sessions — counts against quota"},
            },
            # Kept for callers that still read the old field names. These are
            # the HOST's numbers and are labelled as such in the UI.
            "host": _host_usage(),
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
@admin_router.get("/api/admin/services")
def omni_services():
    """Postgres / Qdrant / Neo4j: up or down, and when down, why.

    The engine that starts these lives in another process, so its own module
    flags are not readable from here. It publishes the report to Redis after
    every `ensure_services()`; this reads that. No report means the omni layer
    has not booted yet, which is itself the answer.
    """
    try:
        r = _safe_redis()
        raw = r.get("VIOS_OMNI_SERVICES") if r is not None else None
        if raw:
            report = json.loads(raw)
            report["source"] = "omni engine"
            return report
        return {
            "available": {"postgres": False, "qdrant": False, "neo4j": False},
            "diagnostics": {"postgres": [
                "the omni engine has not reported yet — it is still booting, "
                "or it is disabled (VIOS_OMNI=0), or it exited before it "
                "reached the database step. Check the log for 🔮 [OMNI]."]},
            "source": "not reported",
        }
    except Exception as exc:
        return _error(exc)


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
    """Legacy single-queue pause, kept so old clients keep working.

    It now goes through system_control so the CV worker's heartbeat and the
    admin log say the same thing as the global button. Same Redis key as
    before (CV_PAUSED), so nothing about the wire format changed.
    """
    try:
        import system_control
        res = system_control.set_component("cv", True)
        return res if res.get("ok") else _error(res.get("error"), 503)
    except Exception as exc:
        vios_log(f"Error pausing queue: {exc}", "ADMIN", "ERROR")
        return _error(exc)


@admin_router.post("/api/admin/queue/resume")
def queue_resume():
    try:
        import system_control
        res = system_control.set_component("cv", False)
        return res if res.get("ok") else _error(res.get("error"), 503)
    except Exception as exc:
        vios_log(f"Error resuming queue: {exc}", "ADMIN", "ERROR")
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# 11.-13. Database export — seal the DB into a bundle, upload it to the same
#         Telegram channel the reels are harvested from.
#
# Kaggle sessions are disposable: scratch is wiped between them and even the
# 19.5 GB OUTPUT tier goes when the notebook is deleted. The channel is the only
# storage in this system that outlives the machine, so it doubles as the backup
# target. db_export owns the work; these three routes only start it and report.
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/export/start")
async def export_start(request: Request):
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass          # no body is the normal case — keep_local defaults off
        import db_export
        res = db_export.start_export(keep_local=bool(body.get("keep_local")))
        if not res.get("ok"):
            return _error(res.get("error", "Could not start export"), 409)
        vios_log("Database export started from admin panel", "ADMIN", "INFO")
        return res
    except Exception as exc:
        vios_log(f"Error starting export: {exc}", "ADMIN", "ERROR")
        return _error(exc)


@admin_router.get("/api/admin/export/status")
def export_status():
    try:
        import db_export
        st = db_export.export_status()
        # Telegram readiness is reported alongside the job so the panel can warn
        # before a run rather than after it has already built a bundle.
        from config import missing_telegram_secrets
        st["telegram_missing"] = missing_telegram_secrets()
        if st["state"] != "running":
            st["resumable"] = db_export.resumable_bundle()
        return st
    except Exception as exc:
        return _error(exc)


@admin_router.post("/api/admin/export/cancel")
def export_cancel():
    """Stop a running export.

    Cooperative rather than a kill: the flag is read between stages and inside
    the upload progress callback, so the thread unwinds cleanly and the parts it
    already committed stay on disk for the next run to resume from.
    """
    try:
        import db_export
        res = db_export.cancel_export()
        if not res.get("ok"):
            return _error(res.get("error", "Nothing to cancel"), 409)
        vios_log("Export cancellation requested from admin panel",
                 "ADMIN", "WARN")
        return res
    except Exception as exc:
        return _error(exc)


@admin_router.get("/api/admin/telegram/probe")
def telegram_probe():
    """Can we actually reach the channel with the bot token we have?

    Worth its own route: every previous export failure looked identical from the
    panel (a bar that stopped moving), and about half of them were a missing or
    wrong credential that this answers in one second without writing anything.
    """
    try:
        from config import missing_telegram_secrets
        missing = missing_telegram_secrets()
        import tg_transport as tg
        if not tg.available():
            return {"ok": False, "missing": missing,
                    "error": "VIOS_BOT_TOKEN is not set — export cannot upload."}
        res = tg.probe()
        res["missing"] = missing
        return res
    except Exception as exc:
        return _error(exc)


@admin_router.get("/api/admin/export/bundles")
def export_bundles():
    try:
        import db_export
        return {"bundles": db_export.list_local_bundles()}
    except Exception as exc:
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# 14.-15. Database restore — pull the newest bundle back out of the channel.
#
# The other half of export, and the half that makes it worth running. Postgres
# lives on the container's ephemeral disk, so without this every session starts
# with an empty Omniscient store and re-narrates reels the GPU already did.
#
# Two steps on purpose. `inspect` reads only the manifest and reports what a
# restore would do — which bundle, how much to download, and how its row counts
# compare to the local ones. `apply` overwrites. The panel will not offer apply
# until inspect has run, so nobody destroys a database from a single click.
# ───────────────────────────────────────────────────────────────────────────
@admin_router.post("/api/admin/restore/start")
async def restore_start(request: Request):
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass          # no body means inspect, the non-destructive default
        mode = (body.get("mode") or "inspect").strip()
        seq = body.get("seq") or None
        import db_restore
        res = db_restore.start_restore(mode=mode, seq=seq)
        if not res.get("ok"):
            return _error(res.get("error", "Could not start restore"), 409)
        vios_log(f"Database restore ({mode}) started from admin panel"
                 + (f" for bundle {seq}" if seq else ""), "ADMIN",
                 "WARN" if mode == "apply" else "INFO")
        return res
    except Exception as exc:
        vios_log(f"Error starting restore: {exc}", "ADMIN", "ERROR")
        return _error(exc)


@admin_router.get("/api/admin/restore/status")
def restore_status():
    try:
        import db_restore
        st = db_restore.restore_status()
        from config import missing_telegram_secrets
        st["telegram_missing"] = missing_telegram_secrets()
        return st
    except Exception as exc:
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# 16.-19. Global pause — one switch across harvest, CV, analysis and Omniscient.
#
# The old panel had a single "queue pause" that wrote CV_PAUSED, which the frame
# worker only read when its queue came back empty. On a machine with a backlog
# the button did nothing visible, and it never touched the harvester at all, so
# downloads kept filling the disk that the pause was usually pressed to protect.
# system_control owns the flags; these routes are a thin shell over it.
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/pause/state")
def pause_state():
    """Requested state and observed state, side by side.

    They are reported separately because a disagreement is information: for a
    few seconds after a click it means a worker is finishing its chunk, and
    permanently it means that worker is dead.
    """
    try:
        import system_control
        return system_control.pause_state()
    except Exception as exc:
        return _error(exc)


@admin_router.post("/api/admin/pause/all")
async def pause_all(request: Request):
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        import system_control
        res = system_control.pause_all(reason=str(body.get("reason") or ""))
        return res if res.get("ok") else _error(res.get("error"), 503)
    except Exception as exc:
        vios_log(f"Error pausing system: {exc}", "ADMIN", "ERROR")
        return _error(exc)


@admin_router.post("/api/admin/resume/all")
def resume_all():
    try:
        import system_control
        res = system_control.resume_all()
        return res if res.get("ok") else _error(res.get("error"), 503)
    except Exception as exc:
        vios_log(f"Error resuming system: {exc}", "ADMIN", "ERROR")
        return _error(exc)


@admin_router.post("/api/admin/pause/component")
async def pause_component(request: Request):
    """Pause or resume one stage. Useful on its own: pausing analysis while the
    harvester keeps running is how you free the GPU without losing the scan."""
    try:
        body = await request.json()
    except Exception:
        return _error("Expected JSON body {component, paused}", 400)
    component = str(body.get("component") or "").strip()
    try:
        import system_control
        res = system_control.set_component(component, bool(body.get("paused")))
        return res if res.get("ok") else _error(res.get("error"), 400)
    except Exception as exc:
        return _error(exc)


# ───────────────────────────────────────────────────────────────────────────
# 20.-22. Factory reset — actually delete it, the way Kaggle's own reset does.
#
# Scoped, priced and confirmed. `preview` measures every target so the decision
# is made against real numbers (the question on a Kaggle box is always "will
# this free enough to keep going"), `start` requires the phrase typed exactly,
# and the Telegram channel is never touched because it is the only way back.
# ───────────────────────────────────────────────────────────────────────────
@admin_router.get("/api/admin/reset/preview")
def reset_preview():
    try:
        import system_control
        return system_control.reset_preview()
    except Exception as exc:
        return _error(exc)


@admin_router.post("/api/admin/reset/start")
async def reset_start(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _error("Expected JSON body {scope, confirm}", 400)
    try:
        import system_control
        # The typed phrase is re-validated inside start_reset, not here — this
        # route only forwards it, so there is no way to reach the wipe by
        # calling the module with the check skipped.
        res = system_control.start_reset(
            scope=body.get("scope") or [],
            confirm=str(body.get("confirm") or ""),
            restart=bool(body.get("restart", True)))
        if not res.get("ok"):
            return _error(res.get("error", "Could not start reset"), 409)
        return res
    except Exception as exc:
        vios_log(f"Error starting factory reset: {exc}", "ADMIN", "ERROR")
        return _error(exc)


@admin_router.get("/api/admin/reset/status")
def reset_status():
    try:
        import system_control
        return system_control.reset_status()
    except Exception as exc:
        return _error(exc)
