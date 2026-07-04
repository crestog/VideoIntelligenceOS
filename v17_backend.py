"""
VIOS V17 Workspace Backend — Frame Analysis & Inspection Engine

Endpoints:
  /v17                            → Serve workspace UI
  /api/database                   → Video library with category data
  /api/workspace/{folder}         → Frame list + completeness status
  /api/workspace/{folder}/status  → Polling endpoint for extraction progress
  /api/v17/categories             → Category list for filter dropdown
  /api/batch/{folder}             → Binary-packed frame batch (high-perf)
  /api/logs                       → System log stream
"""

import os, glob, struct, sqlite3, time
from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import HTMLResponse
from config import DB_PATH, VIDEO_DIR, THUMB_DIR, BATCH_FRAME_COUNT
from logger import vios_log, get_recent_logs

v17_router = APIRouter()

# ═══════════════════════════════════════════════════════════
# DATABASE SYNC — Discovers frame folders not yet in DB
# ═══════════════════════════════════════════════════════════

def sync_v17_database():
    """Auto-discovers frames_* folders and ensures they're registered in the videos table."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS videos 
                     (msg_id INTEGER PRIMARY KEY, folder_id TEXT, title TEXT, 
                      frames INTEGER, duration_sec REAL, duration_str TEXT, 
                      thumb TEXT, first_frame TEXT, file_size_mb REAL, abs_path TEXT,
                      created_at REAL)''')
        
        # Migration: add created_at column if missing (existing DBs)
        try:
            c.execute("SELECT created_at FROM videos LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE videos ADD COLUMN created_at REAL")
            conn.commit()

        # Backfill NULL created_at from the frame folder's mtime — a REAL
        # epoch timestamp that reflects actual download/extraction time.
        # (The old code mixed msg_id values with epoch times in the same
        # column, which broke 'Latest Downloaded' sorting entirely.)
        c.execute("SELECT msg_id, abs_path FROM videos WHERE created_at IS NULL")
        for row_msg_id, row_path in c.fetchall():
            try:
                ts = os.path.getmtime(row_path) if row_path and os.path.exists(row_path) else 0.0
            except OSError:
                ts = 0.0
            c.execute("UPDATE videos SET created_at = ? WHERE msg_id = ?", (ts, row_msg_id))

        # Repair legacy rows where msg_id was written into created_at
        # (msg_ids are tiny compared to any modern epoch timestamp).
        c.execute("SELECT msg_id, abs_path FROM videos WHERE created_at < 1000000000")
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
            except:
                ts_end = 0.0
            
            mp4_path = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
            file_size_mb = os.path.getsize(mp4_path) / (1024 * 1024) if os.path.exists(mp4_path) else 0.0

            # Use the folder's real mtime — NOT a shared "now" value, which
            # previously gave every recovered folder the same timestamp and
            # made 'Latest Downloaded' ordering random.
            try:
                folder_ts = os.path.getmtime(f)
            except OSError:
                folder_ts = time.time()

            c.execute('''INSERT INTO videos 
                         (msg_id, folder_id, title, frames, duration_sec, duration_str, thumb, first_frame, file_size_mb, abs_path, created_at) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (msg_id, folder_id, f"Video #{msg_id}", count, ts_end, f"{ts_end:.1f}s",
                       f"/thumbs/{folder_id}.jpg", f"/data/{folder_id}/{os.path.basename(frames[0])}", file_size_mb, f, folder_ts))
            migrated += 1
            
        conn.commit()
        conn.close()
        if migrated > 0:
            vios_log(f"Auto-recovered {migrated} datasets into V17 DB", "SYS", "SUCCESS")
    except Exception as e:
        vios_log(f"DB Sync Failed: {e}", "SYS", "ERROR")

# Run on import
sync_v17_database()


# ═══════════════════════════════════════════════════════════
# UI SERVING
# ═══════════════════════════════════════════════════════════

@v17_router.get("/v17", response_class=HTMLResponse)
async def serve_v17_ui():
    with open("v17_ui.html", "r", encoding="utf-8") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════
# VIDEO LIBRARY — Now includes category data for filtering
# ═══════════════════════════════════════════════════════════

@v17_router.get("/api/database")
def get_database():
    sync_v17_database()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Join videos with posts+categories to get category name
        # Use created_at DESC for reliable download order (rowid is unstable with INSERT OR REPLACE)
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
            d["download_order"] = len(rows) - idx  # Higher = more recent
            d["download_ts"] = d.get("download_ts") or d.get("created_at") or 0
            d["duration"] = d["duration_str"]
            d["size"] = f"{d['file_size_mb']:.1f} MB"
            # category is already included from the JOIN
            payload.append(d)
        return {"folders": payload}
    except Exception as e:
        vios_log(f"Database fetch error: {e}", "SYS", "ERROR")
        return {"folders": []}


# ═══════════════════════════════════════════════════════════
# WORKSPACE — Frame list + completeness status
# ═══════════════════════════════════════════════════════════

@v17_router.get("/api/workspace/{folder_name}")
def get_workspace_data(folder_name: str):
    """Returns frame list, FPS, and extraction completeness status."""
    folder_path = os.path.join(VIDEO_DIR, folder_name)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    
    frames = sorted(glob.glob(os.path.join(folder_path, "frame_*.jpg")))
    filenames = [os.path.basename(f) for f in frames]
    
    # Calculate native FPS from timestamps
    fps = 30.0
    if len(frames) > 1:
        try:
            ts_start = float(filenames[0].split('_ts_')[1].replace('s.jpg', ''))
            ts_end = float(filenames[-1].split('_ts_')[1].replace('s.jpg', ''))
            dur = ts_end - ts_start
            if dur > 0:
                fps = len(frames) / dur
        except:
            pass
    
    # Check expected frame count from DB for completeness
    expected_frames = len(filenames)
    is_complete = True
    try:
        msg_id_str = folder_name.replace('frames_', '')
        if msg_id_str.isdigit():
            conn = sqlite3.connect(DB_PATH, timeout=10)
            cursor = conn.execute('SELECT frames FROM videos WHERE msg_id = ?', (int(msg_id_str),))
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                expected_frames = row[0]
                is_complete = len(filenames) >= expected_frames
    except:
        pass
    
    return {
        "frames": filenames,
        "native_fps": round(fps, 2),
        "total_expected": expected_frames,
        "is_complete": is_complete
    }


@v17_router.get("/api/workspace/{folder_name}/status")
def get_workspace_status(folder_name: str):
    """Polling endpoint for extraction progress. Used by UI auto-refresh."""
    folder_path = os.path.join(VIDEO_DIR, folder_name)
    frames_ready = 0
    if os.path.exists(folder_path):
        frames_ready = len(glob.glob(os.path.join(folder_path, "frame_*.jpg")))
    
    total_expected = frames_ready
    is_complete = True
    try:
        msg_id_str = folder_name.replace('frames_', '')
        if msg_id_str.isdigit():
            conn = sqlite3.connect(DB_PATH, timeout=10)
            cursor = conn.execute('SELECT frames FROM videos WHERE msg_id = ?', (int(msg_id_str),))
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                total_expected = row[0]
                is_complete = frames_ready >= total_expected
    except:
        pass
    
    return {
        "frames_ready": frames_ready,
        "total_expected": total_expected,
        "is_complete": is_complete
    }


# ═══════════════════════════════════════════════════════════
# CATEGORIES — For V17 filter dropdown
# ═══════════════════════════════════════════════════════════

@v17_router.get("/api/v17/categories")
def get_v17_categories():
    """Returns all categories with video count for the library filter."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20)
        conn.row_factory = sqlite3.Row
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
# BINARY FRAME BATCH — High-performance packed JPEG delivery
# ═══════════════════════════════════════════════════════════

@v17_router.get("/api/batch/{folder_name}")
def get_frame_batch(folder_name: str, start: int = 0, count: int = BATCH_FRAME_COUNT):
    """Returns packed binary: [4-byte LE size][JPEG bytes][4-byte LE size][JPEG bytes]..."""
    folder_path = os.path.join(VIDEO_DIR, folder_name)
    frames = sorted(glob.glob(os.path.join(folder_path, "frame_*.jpg")))
    if not frames:
        raise HTTPException(status_code=404, detail="No frames found")
    
    packet = bytearray()
    end = min(start + count, len(frames))
    for i in range(start, end):
        try:
            with open(frames[i], "rb") as f:
                img_data = f.read()
                packet.extend(struct.pack('<I', len(img_data)))
                packet.extend(img_data)
        except:
            pass
    
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    return Response(content=bytes(packet), media_type="application/octet-stream", headers=headers)


# ═══════════════════════════════════════════════════════════
# LOGS — Structured system logs for console panel
# ═══════════════════════════════════════════════════════════

@v17_router.get("/api/logs")
def get_live_logs():
    """Returns recent structured logs from the unified logger."""
    logs = get_recent_logs(100)
    formatted = [f"[{l.get('ts', '')}] {l.get('message', '')}" for l in logs]
    return {"logs": formatted}
