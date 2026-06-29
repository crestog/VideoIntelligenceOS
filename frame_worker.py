import os, cv2, sqlite3, time, json, redis

# Environment Paths
BASE_DIR = '/kaggle/working/Insta-Vault'
LAKE_DIR = os.path.join(BASE_DIR, 'DataLake')
VIDEO_DIR = os.path.join(LAKE_DIR, 'videos')
THUMB_DIR = os.path.join(LAKE_DIR, '.thumbnails')
DB_PATH = os.path.join(LAKE_DIR, 'lake.db')

def log_event(msg):
    print(f"[{time.strftime('%H:%M:%S')}] 🎞️ [CV-ENGINE] {msg}", flush=True)

# ==========================================
# 4. OPENCV CORE ENGINE (USER EXACT LOGIC)
# ==========================================
def extract_video_data(video_path, msg_id):
    log_event(f"🔍 Parsing stream ID: {msg_id}")
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0: return

    folder_id = f"frames_{msg_id}"
    frames_dir = os.path.join(VIDEO_DIR, folder_id)
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        log_event(f"❌ Corrupt Codec on ID: {msg_id}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break 
        timestamp_sec = current_frame / fps
        filename = f"frame_{current_frame:05d}_ts_{timestamp_sec:.3f}s.jpg"
        cv2.imwrite(os.path.join(frames_dir, filename), frame)
        current_frame += 1

    cap.release()
    
    thumb_path = os.path.join(THUMB_DIR, f"{folder_id}.jpg")
    mid_idx = current_frame // 2
    target_frame = os.path.join(frames_dir, f"frame_{mid_idx:05d}_ts_{mid_idx/fps:.3f}s.jpg")
    
    if os.path.exists(target_frame) and not os.path.exists(thumb_path):
        img = cv2.imread(target_frame)
        if img is not None:
            thumb_img = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.imwrite(thumb_path, thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    try:
        duration_sec = current_frame / fps
        file_size_mb = os.path.getsize(video_path) / (1024*1024)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        
        # Make sure the table exists just in case
        c.execute('''CREATE TABLE IF NOT EXISTS videos 
                 (msg_id INTEGER PRIMARY KEY, folder_id TEXT, title TEXT, 
                  frames INTEGER, duration_sec REAL, duration_str TEXT, 
                  thumb TEXT, first_frame TEXT, file_size_mb REAL, abs_path TEXT)''')
                  
        c.execute('''INSERT OR REPLACE INTO videos 
                     (msg_id, folder_id, title, frames, duration_sec, duration_str, thumb, first_frame, file_size_mb, abs_path) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (msg_id, folder_id, f"Video #{msg_id}", current_frame, duration_sec, f"{duration_sec:.1f}s", f"/thumbs/{folder_id}.jpg", f"/data/{folder_id}/frame_00000_ts_0.000s.jpg", file_size_mb, frames_dir))
        conn.commit()
        conn.close()
        log_event(f"💾 Committed {current_frame} frames for ID: {msg_id}")
    except Exception as e: 
        log_event(f"❌ DB write failed: {str(e)}")

# ==========================================
# QUEUE CONNECTION LOGIC
# ==========================================
def run_worker():
    log_event("Connecting to Message Queue...")
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
    except Exception:
        log_event("❌ Redis not reachable.")
        return

    while True:
        try:
            # Listens to QUEUE_VISION. Blpop waits until a job arrives.
            result = r.blpop("QUEUE_VISION", timeout=5)
            if result:
                _, data_str = result
                data = json.loads(data_str)
                msg_id = data.get("msg_id")
                video_path = data.get("path")
                if msg_id and video_path:
                    extract_video_data(video_path, msg_id)
        except Exception as e:
            time.sleep(2)

if __name__ == "__main__":
    run_worker()
