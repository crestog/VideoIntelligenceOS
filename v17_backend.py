import os, glob, time, shutil, struct, sqlite3
from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import HTMLResponse

v17_router = APIRouter()

VIDEO_DIR = '/kaggle/working/Insta-Vault/DataLake/videos'
THUMB_DIR = '/kaggle/working/Insta-Vault/DataLake/.thumbnails'
FLAG_DIR = '/kaggle/working/Insta-Vault/DataLake/_Flagged_Dataset'
DB_PATH = '/kaggle/working/Insta-Vault/DataLake/lake.db'
KAGGLE_LOGS = ["[SYSTEM] V17 Workspace Booted & Synced with DataLake."]

# Bulletproof directory creation
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(FLAG_DIR, exist_ok=True)

def sync_v17_database():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS videos 
                     (msg_id INTEGER PRIMARY KEY, folder_id TEXT, title TEXT, 
                      frames INTEGER, duration_sec REAL, duration_str TEXT, 
                      thumb TEXT, first_frame TEXT, file_size_mb REAL, abs_path TEXT)''')
        
        legacy_folders = glob.glob(os.path.join(VIDEO_DIR, 'frames_*'))
        migrated = 0
        for f in legacy_folders:
            folder_id = os.path.basename(f)
            msg_id_str = folder_id.replace('frames_', '')
            if not msg_id_str.isdigit(): continue
            msg_id = int(msg_id_str)
            
            c.execute("SELECT 1 FROM videos WHERE msg_id=?", (msg_id,))
            if c.fetchone(): continue
            
            frames = sorted(glob.glob(os.path.join(f, "frame_*.jpg")))
            count = len(frames)
            if count == 0: continue
            
            try: ts_end = float(os.path.basename(frames[-1]).split('_ts_')[1].replace('s.jpg', ''))
            except: ts_end = 0.0
            
            mp4_path = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
            file_size_mb = os.path.getsize(mp4_path) / (1024*1024) if os.path.exists(mp4_path) else 0.0
            
            c.execute('''INSERT INTO videos 
                         (msg_id, folder_id, title, frames, duration_sec, duration_str, thumb, first_frame, file_size_mb, abs_path) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (msg_id, folder_id, f"Video #{msg_id}", count, ts_end, f"{ts_end:.1f}s", f"/thumbs/{folder_id}.jpg", f"/data/{folder_id}/{os.path.basename(frames[0])}", file_size_mb, f))
            migrated += 1
            
        conn.commit()
        conn.close()
        if migrated > 0: KAGGLE_LOGS.append(f"✅ Auto-recovered {migrated} datasets.")
    except Exception as e:
        KAGGLE_LOGS.append(f"❌ DB Sync Failed: {e}")

# Run DB sync on import
sync_v17_database()

@v17_router.get("/v17", response_class=HTMLResponse)
async def serve_v17_ui():
    with open("v17_ui.html", "r", encoding="utf-8") as f:
        return f.read()

@v17_router.get("/api/database")
def get_database():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM videos ORDER BY msg_id DESC")
        rows = c.fetchall()
        conn.close()
        payload = []
        for r in rows:
            d = dict(r)
            d["id"] = d["folder_id"]
            d["numeric_id"] = d["msg_id"]
            d["duration"] = d["duration_str"]
            d["size"] = f"{d['file_size_mb']:.1f} MB"
            payload.append(d)
        return {"folders": payload}
    except Exception: return {"folders": []}

@v17_router.get("/api/workspace/{folder_name}")
def get_workspace_data(folder_name: str):
    folder_path = os.path.join(VIDEO_DIR, folder_name)
    if not os.path.exists(folder_path): raise HTTPException(status_code=404, detail="Not Found")
    frames = sorted(glob.glob(os.path.join(folder_path, "frame_*.jpg")))
    filenames = [os.path.basename(f) for f in frames]
    fps = 30.0
    if len(frames) > 1:
        try:
            ts_start = float(filenames[0].split('_ts_')[1].replace('s.jpg', ''))
            ts_end = float(filenames[-1].split('_ts_')[1].replace('s.jpg', ''))
            dur = ts_end - ts_start
            if dur > 0: fps = len(frames) / dur
        except: pass
    return {"frames": filenames, "native_fps": fps}

@v17_router.get("/api/batch/{folder_name}")
def get_frame_batch(folder_name: str, start: int = 0, count: int = 100):
    folder_path = os.path.join(VIDEO_DIR, folder_name)
    frames = sorted(glob.glob(os.path.join(folder_path, "frame_*.jpg")))
    if not frames: raise HTTPException(status_code=404)
    packet = bytearray()
    end = min(start + count, len(frames))
    for i in range(start, end):
        try:
            with open(frames[i], "rb") as f:
                img_data = f.read()
                packet.extend(struct.pack('<I', len(img_data)))
                packet.extend(img_data)
        except: pass
    return Response(content=bytes(packet), media_type="application/octet-stream")

@v17_router.get("/api/logs")
def get_live_logs():
    return {"logs": KAGGLE_LOGS}

@v17_router.post("/api/flag/{folder_name}/{frame_filename}")
def flag_frame(folder_name: str, frame_filename: str):
    source_path = os.path.join(VIDEO_DIR, folder_name, frame_filename)
    target_path = os.path.join(FLAG_DIR, f"{folder_name}_{frame_filename}")
    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, target_path)
            return {"status": "success"}
        except: raise HTTPException(status_code=500)
    raise HTTPException(status_code=404)
