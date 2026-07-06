"""
VIOS V17 Workspace Backend v2 — Frame Analysis & Inspection Engine

Endpoints:
  /v17                              → Serve workspace UI
  /api/database                     → Video library with category data
  /api/workspace/{folder}           → Frame index + fps + completeness
  /api/workspace/{folder}/status    → Polling endpoint for extraction progress
  /api/v17/categories               → Category list for filter dropdown
  /api/frame/{folder}/{idx}         → ONE frame, instant first paint  [NEW]
  /api/batch/{folder}               → Packed frames; ?step= sparse ladder,
                                      ?tier=preview|full               [NEW]
  /api/notes/{folder}               → Per-frame timestamps+descriptions [NEW]
  /api/db/tables                    → All tables + row counts          [NEW]
  /api/db/table/{name}              → Paginated read-only table browser [NEW]
  /api/db/overview                  → Headline counts + DB size        [NEW]
  /api/db/export | /api/db/import   → Telegram snapshot cycle          [NEW]
  /api/logs                         → System log stream

Performance notes:
  • Frame lists are cached in-process (mtime-invalidated) — the old code
    re-globbed the whole folder on EVERY batch request.
  • The preview tier (~8 KB/frame) plus ?step= sampling means a workspace
    becomes scrubbable in <2s regardless of video length.
"""

import asyncio
import glob
import json
import os
import re
import sqlite3
import struct
import time

from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from config import (DB_PATH, VIDEO_DIR, THUMB_DIR, BATCH_FRAME_COUNT,
                    PREVIEW_DIR_NAME, SQLITE_TIMEOUT)
from logger import vios_log, get_recent_logs

v17_router = APIRouter()

FOLDER_RE = re.compile(r'^frames_\d+$')


def _check_folder(folder_name: str) -> str:
    """Validate folder name (no path traversal) and return its absolute path."""
    if not FOLDER_RE.match(folder_name):
        raise HTTPException(status_code=400, detail="Invalid folder name")
    return os.path.join(VIDEO_DIR, folder_name)


def _db(row_factory=True) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    if row_factory:
        conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_TIMEOUT * 1000}")
    return conn


# ═══════════════════════════════════════════════════════════
# FRAME LIST CACHE — one glob per folder per change, not per request
# ═══════════════════════════════════════════════════════════
_frame_cache: dict = {}   # folder -> {names, mtime, ts}
_FRAME_CACHE_TTL = 2.0    # re-stat at most every 2s while extraction runs


def get_frame_names(folder_name: str):
    """Sorted frame filenames for a folder, cached and mtime-invalidated."""
    folder_path = _check_folder(folder_name)
    now = time.time()
    entry = _frame_cache.get(folder_name)
    if entry and (now - entry['ts']) < _FRAME_CACHE_TTL:
        return entry['names']
    try:
        mtime = os.path.getmtime(folder_path)
    except OSError:
        raise HTTPException(status_code=404, detail="Folder not found")
    if entry and entry['mtime'] == mtime:
        entry['ts'] = now
        return entry['names']
    names = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(folder_path, 'frame_*.jpg')))
    _frame_cache[folder_name] = {'names': names, 'mtime': mtime, 'ts': now}
    return names


def _frame_path(folder_name: str, filename: str, tier: str) -> str:
    """Resolve a frame filename to full-res or preview tier on disk."""
    folder_path = _check_folder(folder_name)
    if tier == 'preview':
        # preview files are frame_%05d.jpg (no ts suffix)
        idx_part = filename.split('_ts_')[0]          # frame_00042
        p = os.path.join(folder_path, PREVIEW_DIR_NAME, idx_part + '.jpg')
        if os.path.exists(p):
            return p
        # legacy folders have no preview tier — fall through to full
    return os.path.join(folder_path, filename)


# ═══════════════════════════════════════════════════════════
# DATABASE SYNC — Discovers frame folders not yet in DB
# ═══════════════════════════════════════════════════════════
def sync_v17_database():
    """Auto-discovers frames_* folders and ensures they're registered in the videos table."""
    try:
        conn = _db(row_factory=False)
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
        conn.commit()

        # Backfill NULL created_at from folder mtime (real download time)
        c.execute("SELECT msg_id, abs_path FROM videos WHERE created_at IS NULL OR created_at < 1000000000")
        for row_msg_id, row_path in c.fetchall():
            try:
                ts = os.path.getmtime(row_path) if row_path and os.path.exists(row_path) else 0.0
            except OSError:
                ts = 0.0
            if ts > 0:
                c.execute("UPDATE videos SET created_at = ? WHERE msg_id = ?", (ts, row_msg_id))

        legacy_folders = glob.glob(os.path.join(VIDEO_DIR, 'frames_*'))
        migrated = 0
        for f in legacy_folders:
            folder_id = os.path.basename(f)
            msg_id_str = folder_id.replace('frames_', '')
            if not msg_id_str.isdigit():
                continue
            msg_id = int(msg_id_str)

            c.execute("SELECT 1 FROM videos WHERE msg_id=?", (msg_id,))
            if c.fetchone():
                continue

            frames = sorted(glob.glob(os.path.join(f, "frame_*.jpg")))
            count = len(frames)
            if count == 0:
                continue

            try:
                ts_end = float(os.path.basename(frames[-1]).split('_ts_')[1].replace('s.jpg', ''))
            except (IndexError, ValueError):
                ts_end = 0.0

            mp4_path = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
            file_size_mb = os.path.getsize(mp4_path) / (1024 * 1024) if os.path.exists(mp4_path) else 0.0
            fps_est = (count / ts_end) if ts_end > 0 else 30.0
            try:
                folder_ts = os.path.getmtime(f)
            except OSError:
                folder_ts = time.time()

            c.execute('''INSERT INTO videos
                         (msg_id, folder_id, title, frames, duration_sec, duration_str,
                          thumb, first_frame, file_size_mb, abs_path, created_at, fps)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (msg_id, folder_id, f"Video #{msg_id}", count, ts_end, f"{ts_end:.1f}s",
                       f"/thumbs/{folder_id}.jpg", f"/data/{folder_id}/{os.path.basename(frames[0])}",
                       file_size_mb, f, folder_ts, round(fps_est, 3)))
            migrated += 1

        conn.commit()
        conn.close()
        if migrated > 0:
            vios_log(f"Auto-recovered {migrated} datasets into V17 DB", "SYS", "SUCCESS")
    except Exception as e:
        vios_log(f"DB Sync Failed: {e}", "SYS", "ERROR")


sync_v17_database()


# ═══════════════════════════════════════════════════════════
# UI SERVING
# ═══════════════════════════════════════════════════════════
@v17_router.get("/v17", response_class=HTMLResponse)
async def serve_v17_ui():
    with open("v17_ui.html", "r", encoding="utf-8") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════
# VIDEO LIBRARY
# ═══════════════════════════════════════════════════════════
@v17_router.get("/api/database")
def get_database():
    sync_v17_database()
    try:
        conn = _db()
        c = conn.cursor()
        c.execute("""
            SELECT v.created_at as download_ts, v.*,
                   COALESCE(cat.name, 'Uncategorized') as category
            FROM videos v
            LEFT JOIN posts p ON v.msg_id = p.video_id
            LEFT JOIN categories cat ON p.category_id = cat.id
            ORDER BY v.created_at DESC, v.msg_id DESC
        """)
        rows = c.fetchall()
        conn.close()

        payload = []
        for idx, r in enumerate(rows):
            d = dict(r)
            d["id"] = d["folder_id"]
            d["numeric_id"] = d["msg_id"]
            d["download_order"] = len(rows) - idx
            d["download_ts"] = d.get("download_ts") or d.get("created_at") or 0
            d["duration"] = d["duration_str"]
            d["size"] = f"{d['file_size_mb']:.1f} MB"
            payload.append(d)
        return {"folders": payload}
    except Exception as e:
        vios_log(f"Database fetch error: {e}", "SYS", "ERROR")
        return {"folders": []}


# ═══════════════════════════════════════════════════════════
# WORKSPACE — Frame index + fps + completeness
# ═══════════════════════════════════════════════════════════
@v17_router.get("/api/workspace/{folder_name}")
def get_workspace_data(folder_name: str):
    folder_path = _check_folder(folder_name)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    filenames = get_frame_names(folder_name)

    # fps: DB first (exact, from ffprobe), filename-derived fallback
    fps = 0.0
    expected_frames = len(filenames)
    is_complete = True
    has_notes = False
    msg_id_str = folder_name.replace('frames_', '')
    try:
        conn = _db(row_factory=False)
        row = conn.execute('SELECT frames, fps FROM videos WHERE msg_id = ?',
                           (int(msg_id_str),)).fetchone()
        if row:
            if row[0]:
                expected_frames = row[0]
                is_complete = len(filenames) >= expected_frames
            if len(row) > 1 and row[1]:
                fps = float(row[1])
        try:
            n = conn.execute('SELECT COUNT(*) FROM frame_notes WHERE msg_id = ?',
                             (int(msg_id_str),)).fetchone()
            has_notes = bool(n and n[0])
        except sqlite3.OperationalError:
            pass
        conn.close()
    except Exception:
        pass

    if fps <= 0 and len(filenames) > 1:
        try:
            ts_end = float(filenames[-1].split('_ts_')[1].replace('s.jpg', ''))
            if ts_end > 0:
                fps = (len(filenames) - 1) / ts_end
        except (IndexError, ValueError):
            pass
    if fps <= 0:
        fps = 30.0

    has_preview = os.path.isdir(os.path.join(folder_path, PREVIEW_DIR_NAME))

    return {
        "frames": filenames,
        "native_fps": round(fps, 3),
        "total_expected": expected_frames,
        "is_complete": is_complete,
        "has_preview": has_preview,
        "has_notes": has_notes,
    }


@v17_router.get("/api/workspace/{folder_name}/status")
def get_workspace_status(folder_name: str):
    folder_path = _check_folder(folder_name)
    frames_ready = 0
    if os.path.exists(folder_path):
        _frame_cache.pop(folder_name, None)   # force fresh count during extraction
        frames_ready = len(get_frame_names(folder_name))

    total_expected = frames_ready
    is_complete = True
    try:
        msg_id_str = folder_name.replace('frames_', '')
        conn = _db(row_factory=False)
        row = conn.execute('SELECT frames FROM videos WHERE msg_id = ?',
                           (int(msg_id_str),)).fetchone()
        conn.close()
        if row and row[0]:
            total_expected = row[0]
            is_complete = frames_ready >= total_expected
    except Exception:
        pass

    return {"frames_ready": frames_ready, "total_expected": total_expected,
            "is_complete": is_complete}


# ═══════════════════════════════════════════════════════════
# CATEGORIES
# ═══════════════════════════════════════════════════════════
@v17_router.get("/api/v17/categories")
def get_v17_categories():
    try:
        conn = _db()
        c = conn.cursor()
        c.execute("""
            SELECT cat.name, COUNT(DISTINCT v.msg_id) as video_count
            FROM categories cat
            LEFT JOIN posts p ON cat.id = p.category_id AND p.status = 'Harvested'
            LEFT JOIN videos v ON p.video_id = v.msg_id
            GROUP BY cat.name
            HAVING video_count > 0
            ORDER BY video_count DESC
        """)
        rows = c.fetchall()
        conn.close()
        return {"categories": [{"name": r["name"], "video_count": r["video_count"]} for r in rows]}
    except Exception as e:
        vios_log(f"V17 categories error: {e}", "SYS", "ERROR")
        return {"categories": []}


# ═══════════════════════════════════════════════════════════
# SINGLE FRAME — instant first paint (<300 ms)
# ═══════════════════════════════════════════════════════════
@v17_router.get("/api/frame/{folder_name}/{idx}")
def get_single_frame(folder_name: str, idx: int, tier: str = "preview"):
    names = get_frame_names(folder_name)
    if not names:
        raise HTTPException(status_code=404, detail="No frames extracted yet")
    idx = max(0, min(idx, len(names) - 1))
    path = _frame_path(folder_name, names[idx], tier)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


# ═══════════════════════════════════════════════════════════
# BINARY FRAME BATCH — packed JPEGs; sparse (?step) + tiered (?tier)
# Format: [4-byte LE frame-index][4-byte LE size][JPEG bytes] ...
# (index prefix added so sparse batches self-describe their positions)
# ═══════════════════════════════════════════════════════════
@v17_router.get("/api/batch/{folder_name}")
def get_frame_batch(folder_name: str, start: int = 0, count: int = BATCH_FRAME_COUNT,
                    step: int = 1, tier: str = "full"):
    names = get_frame_names(folder_name)
    if not names:
        raise HTTPException(status_code=404, detail="No frames found")

    step = max(1, step)
    total = len(names)
    packet = bytearray()
    picked = 0
    i = max(0, start)
    while i < total and picked < count:
        try:
            with open(_frame_path(folder_name, names[i], tier), "rb") as f:
                img = f.read()
            packet.extend(struct.pack('<II', i, len(img)))
            packet.extend(img)
            picked += 1
        except OSError:
            pass
        i += step

    return Response(content=bytes(packet), media_type="application/octet-stream",
                    headers={"Cache-Control": "public, max-age=31536000, immutable",
                             "X-Frame-Total": str(total)})


# ═══════════════════════════════════════════════════════════
# FRAME NOTES — timestamps + "what is happening" per frame
# ═══════════════════════════════════════════════════════════
@v17_router.get("/api/notes/{folder_name}")
def get_frame_notes(folder_name: str):
    msg_id_str = folder_name.replace('frames_', '')
    if not msg_id_str.isdigit():
        raise HTTPException(status_code=400, detail="Invalid folder name")
    try:
        conn = _db()
        notes = [dict(r) for r in conn.execute(
            '''SELECT frame_idx, ts_sec, description, ocr_text, objects
               FROM frame_notes WHERE msg_id = ? ORDER BY frame_idx''',
            (int(msg_id_str),)).fetchall()]
        transcript = [dict(r) for r in conn.execute(
            '''SELECT start_sec, end_sec, text
               FROM transcripts WHERE msg_id = ? ORDER BY start_sec''',
            (int(msg_id_str),)).fetchall()]
        conn.close()
        return {"notes": notes, "transcript": transcript}
    except sqlite3.OperationalError:
        return {"notes": [], "transcript": []}   # tables not created yet
    except Exception as e:
        vios_log(f"Notes fetch error: {e}", "SYS", "ERROR")
        return {"notes": [], "transcript": []}


# ═══════════════════════════════════════════════════════════
# DATABASE VIEWER — read-only browser over EVERY table
# ═══════════════════════════════════════════════════════════
@v17_router.get("/api/db/tables")
def db_tables():
    conn = _db(row_factory=False)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()]
        out = []
        for t in tables:
            try:
                cnt = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except sqlite3.Error:
                cnt = -1
            out.append({"name": t, "rows": cnt})
        return {"tables": out, "db_path": DB_PATH,
                "db_size_mb": round(os.path.getsize(DB_PATH) / (1024 * 1024), 2)
                              if os.path.exists(DB_PATH) else 0}
    finally:
        conn.close()


@v17_router.get("/api/db/table/{name}")
def db_table(name: str, offset: int = 0, limit: int = 50, q: str = ""):
    conn = _db()
    try:
        known = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if name not in known:
            raise HTTPException(status_code=404, detail="Unknown table")
        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")').fetchall()]

        where = ""
        params: list = []
        if q.strip():
            # LIKE across all columns (cast to text) — read-only + parameterized
            like = " OR ".join(f'CAST("{c}" AS TEXT) LIKE ?' for c in cols)
            where = f" WHERE {like}"
            params = [f"%{q.strip()}%"] * len(cols)

        total = conn.execute(f'SELECT COUNT(*) FROM "{name}"{where}', params).fetchone()[0]
        rows = conn.execute(
            f'SELECT * FROM "{name}"{where} LIMIT ? OFFSET ?',
            params + [limit, offset]).fetchall()

        def _cell(v):
            if isinstance(v, bytes):
                return f"<blob {len(v)} bytes>"
            if isinstance(v, str) and len(v) > 400:
                return v[:400] + "…"
            return v

        return {"columns": cols,
                "rows": [[_cell(v) for v in tuple(r)] for r in rows],
                "total": total, "offset": offset, "limit": limit}
    finally:
        conn.close()


@v17_router.get("/api/db/overview")
def db_overview():
    conn = _db(row_factory=False)
    try:
        def _count(sql):
            try:
                return conn.execute(sql).fetchone()[0]
            except sqlite3.Error:
                return 0
        return {
            "posts": _count("SELECT COUNT(*) FROM posts"),
            "harvested": _count("SELECT COUNT(*) FROM posts WHERE status='Harvested'"),
            "videos_extracted": _count("SELECT COUNT(*) FROM videos"),
            "total_frames": _count("SELECT COALESCE(SUM(frames),0) FROM videos"),
            "frame_notes": _count("SELECT COUNT(*) FROM frame_notes"),
            "transcript_segments": _count("SELECT COUNT(*) FROM transcripts"),
            "categories": _count("SELECT COUNT(*) FROM categories"),
            "creators": _count("SELECT COUNT(*) FROM creators"),
            "db_size_mb": round(os.path.getsize(DB_PATH) / (1024 * 1024), 2)
                          if os.path.exists(DB_PATH) else 0,
        }
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════
# SNAPSHOT — Telegram export/import, background with status
# ═══════════════════════════════════════════════════════════
_snap_status = {"state": "idle", "detail": "", "started": 0.0}


def _snap_running():
    return _snap_status["state"] in ("exporting", "importing")


async def _run_snapshot(kind: str):
    import snapshot_manager
    _snap_status.update(state=f"{kind}ing", detail="starting…", started=time.time())
    try:
        if kind == "export":
            msg_id = await snapshot_manager.export_snapshot()
            _snap_status.update(state="done",
                                detail=f"Exported OK — manifest msg #{msg_id}")
        else:
            ok = await snapshot_manager.import_snapshot()
            _snap_status.update(state="done" if ok else "error",
                                detail="Imported OK — DB restored" if ok
                                       else "No snapshot found in channel")
    except Exception as e:
        _snap_status.update(state="error", detail=str(e)[:300])
        vios_log(f"Snapshot {kind} failed: {e}", "SNAP", "ERROR")


@v17_router.post("/api/db/export")
async def db_export():
    if _snap_running():
        return {"ok": False, "message": "A snapshot operation is already running"}
    asyncio.create_task(_run_snapshot("export"))
    return {"ok": True, "message": "Export started — uploading DB snapshot to Telegram"}


@v17_router.post("/api/db/import")
async def db_import():
    if _snap_running():
        return {"ok": False, "message": "A snapshot operation is already running"}
    asyncio.create_task(_run_snapshot("import"))
    return {"ok": True, "message": "Import started — restoring latest snapshot from Telegram"}


@v17_router.get("/api/db/snapshot/status")
def db_snapshot_status():
    return dict(_snap_status)


# ═══════════════════════════════════════════════════════════
# LOGS
# ═══════════════════════════════════════════════════════════
@v17_router.get("/api/logs")
def get_live_logs():
    logs = get_recent_logs(100)
    formatted = [f"[{l.get('ts', '')}] {l.get('message', '')}" for l in logs]
    return {"logs": formatted}
