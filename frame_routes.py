import os, glob, struct, shutil, sqlite3
from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import HTMLResponse

# Create an independent router
frame_router = APIRouter()

# Map Paths to your actual DataLake system
BASE_DIR = '/kaggle/working/Insta-Vault'
LAKE_DIR = os.path.join(BASE_DIR, 'DataLake')
VIDEO_DIR = os.path.join(LAKE_DIR, 'videos')
THUMB_DIR = os.path.join(LAKE_DIR, '.thumbnails')
FLAG_DIR = os.path.join(LAKE_DIR, '_Flagged_Dataset')
DB_PATH = os.path.join(LAKE_DIR, 'lake.db')
os.makedirs(FLAG_DIR, exist_ok=True)

# 1. Mount the HTML directly to a single endpoint
@frame_router.get("/frames", response_class=HTMLResponse)
def serve_frames_ui():
    with open("frame_ui.html", "r", encoding="utf-8") as f:
        return f.read()

# 2. Database API (Matches exact logic, targets 'frame_videos' table from your worker)
@frame_router.get("/api/database")
def get_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM frame_videos ORDER BY msg_id DESC")
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

@frame_router.get("/api/workspace/{folder_name}")
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

@frame_router.get("/api/batch/{folder_name}")
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

@frame_router.post("/api/flag/{folder_name}/{frame_filename}")
def flag_frame(folder_name: str, frame_filename: str):
    source_path = os.path.join(VIDEO_DIR, folder_name, frame_filename)
    target_path = os.path.join(FLAG_DIR, f"{folder_name}_{frame_filename}")
    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, target_path)
            return {"status": "success"}
        except: raise HTTPException(status_code=500)
    raise HTTPException(status_code=404)

@frame_router.get("/api/logs")
def get_live_logs():
    return {"logs": ["[SYSTEM] Frame UI Connected & Synchronized with DataLake."]}
