

import os
import sqlite3
import pandas as pd
import re
import builtins
import time
import traceback
import asyncio
import urllib.request
import json
import uvicorn
import threading
import subprocess
import atexit
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pyrogram import Client
from pyrogram.errors import FloodWait
import nest_asyncio
from queue_manager import push_job


# Patch Kaggle's IPython event loop 
nest_asyncio.apply()

def custom_print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(f"[{time.strftime('%H:%M:%S')}]", *args, **kwargs)

# --- SYSTEM CLEANUP ON SHUTDOWN ---
def terminate_zombies():
    custom_print("🧹 Sweeping orphaned processes...")
    os.system("pkill -9 ffmpeg > /dev/null 2>&1")
    os.system("pkill -9 ffprobe > /dev/null 2>&1")
    os.system("pkill -9 cloudflared > /dev/null 2>&1")
atexit.register(terminate_zombies)
terminate_zombies()

# --- CREDENTIALS ---
API_ID = 37392880
API_HASH = "4037344084ae998be2cdaee3192bd8f8"
BOT_TOKEN = "8269867642:AAH76B2_aFbqc6OqNiCAm-NenTTmG_SWavU"
CHANNEL_ID = -1003762735924

# --- DIRECTORY ROUTING ---
BASE_DIR = '/kaggle/working/Insta-Vault'
LAKE_DIR = os.path.join(BASE_DIR, 'DataLake')
DB_PATH = os.path.join(LAKE_DIR, 'lake.db')
VIDEO_DIR = os.path.join(LAKE_DIR, 'videos')
SESSION_PATH = os.path.join(LAKE_DIR, 'bot_session')
STATE_FILE = os.path.join(LAKE_DIR, 'state.txt')

for j_file in [f"{SESSION_PATH}.session-journal", f"{DB_PATH}-journal", f"{DB_PATH}-wal"]:
    if os.path.exists(j_file):
        try: os.remove(j_file)
        except: pass

# --- SHARED GLOBALS ---
GLOBAL_STATUS = "⏳ Omega Server Booting..."
CATEGORY_QUEUE = []
SCAN_TRIGGER = threading.Event()
SCAN_TRIGGER.set()

# ==========================================
# 1. HIGH-SPEED DATABASE LOGIC
# ==========================================
def init_db():
    os.makedirs(LAKE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    cursor = conn.cursor()
    
    cursor.execute("CREATE TABLE IF NOT EXISTS categories (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
    cursor.execute("CREATE TABLE IF NOT EXISTS creators (id INTEGER PRIMARY KEY, username TEXT UNIQUE)")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            video_id INTEGER PRIMARY KEY, category_id INTEGER, creator_id INTEGER,
            likes INTEGER, caption TEXT, local_video_path TEXT, status TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS posts_search USING fts5(
            video_id UNINDEXED, caption, creator, category
        )
    ''')
    
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS sync_posts_search AFTER INSERT ON posts BEGIN
            INSERT INTO posts_search(video_id, caption, creator, category)
            VALUES (
                new.video_id, 
                new.caption, 
                (SELECT username FROM creators WHERE id = new.creator_id), 
                (SELECT name FROM categories WHERE id = new.category_id)
            );
        END;
    ''')
    
    conn.commit()
    conn.close()

def get_playlist_data(category):
    conn = sqlite3.connect(DB_PATH, timeout=20)
    cursor = conn.cursor()
    base_query = '''SELECT cr.username, c.name, p.likes, p.caption, p.local_video_path, p.video_id
                    FROM posts p JOIN creators cr ON p.creator_id = cr.id JOIN categories c ON p.category_id = c.id
                    WHERE p.local_video_path IS NOT NULL'''
    if category and category != "All Categories":
        base_query += " AND c.name = ? ORDER BY p.video_id DESC"
        cursor.execute(base_query, (category,))
    else:
        base_query += " ORDER BY p.video_id DESC"
        cursor.execute(base_query)
    
    rows = cursor.fetchall()
    conn.close()
    
    playlist = []
    for row in rows:
        filename = os.path.basename(row[4]) if row[4] else ""
        playlist.append({
            'username': row[0], 'name': row[1], 'likes': row[2],
            'caption': row[3], 'filename': filename, 'id': row[5]
        })
    return playlist

# ==========================================
# 2. GHOST WORKER (BACKGROUND TELEGRAM LOGIC)
# ==========================================
def extract_text(pattern, text):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else "Unknown"

def extract_num(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1).replace(",", "").strip()) if match else 0

async def ensure_web_safe(video_path):
    try:
        proc = await asyncio.create_subprocess_exec('ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', video_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        codec = stdout.decode().strip().split('\n')[0].lower()
        temp_path = video_path + ".tmp.mp4"

        if codec and codec not in ['h264', 'av1', 'hevc', 'h265']:
            custom_print(f"⚙️ FFmpeg Codec Upgrade ({codec}) -> Lossless CRF 17")
            conv_proc = await asyncio.create_subprocess_exec('ffmpeg', '-y', '-i', video_path, '-c:v', 'libx264', '-preset', 'superfast', '-crf', '17', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', temp_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        else:
            custom_print(f"⚡ Instant Remux: 100% Quality Retention & FastStart...")
            conv_proc = await asyncio.create_subprocess_exec('ffmpeg', '-y', '-i', video_path, '-c', 'copy', '-movflags', '+faststart', temp_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        
        await conv_proc.communicate()
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            os.replace(temp_path, video_path)
    except Exception as e: 
        custom_print(f"❌ FFmpeg Error: {e}")

async def background_downloader():
    global GLOBAL_STATUS, CATEGORY_QUEUE
    custom_print("\n=======================================")
    custom_print("🚀 GHOST WORKER: SYSTEMS ONLINE")
    custom_print("=======================================\n")

    app_client = Client(SESSION_PATH, api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)
    await app_client.start()

    try:
        ping = await app_client.send_message(CHANNEL_ID, ".", disable_notification=True)
        latest_id = ping.id
    except Exception:
        GLOBAL_STATUS = "⚙️ Auto-Resolving Telegram Cache..."
        amnesia_event = asyncio.Event()
        
        @app_client.on_message()
        async def handler(client, message):
            if message.chat and message.chat.id == CHANNEL_ID:
                amnesia_event.set()

        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            payload = json.dumps({"chat_id": CHANNEL_ID, "text": ".", "disable_notification": True}).encode('utf-8')
            req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req)
            await asyncio.wait_for(amnesia_event.wait(), timeout=10.0)
        except Exception as e:
            GLOBAL_STATUS = "❌ Bot access denied."
            await asyncio.sleep(10)
            return

    while True:
        if not SCAN_TRIGGER.is_set():
            GLOBAL_STATUS = "💤 Idling. Background queue is empty."
            while not SCAN_TRIGGER.is_set():
                await asyncio.sleep(1)

        SCAN_TRIGGER.clear()
        GLOBAL_STATUS = "⚡ Running Flash Metadata Scan across Telegram..."
        custom_print("⚡ Scanning Telegram for new Posts...")

        try:
            ping = await app_client.send_message(CHANNEL_ID, ".", disable_notification=True)
            latest_id = ping.id
            await ping.delete()

            conn = sqlite3.connect(DB_PATH, timeout=20)
            cursor = conn.cursor()
            missing_ids = [i for i in range(latest_id, 0, -1) if not cursor.execute("SELECT video_id FROM posts WHERE video_id = ?", (i,)).fetchone()]
            conn.close()

            BATCH_SIZE = 200
            for i in range(0, len(missing_ids), BATCH_SIZE):
                chunk = missing_ids[i:i + BATCH_SIZE]
                try:
                    msgs = await app_client.get_messages(CHANNEL_ID, chunk)
                    if not isinstance(msgs, list): msgs = [msgs]
                    conn = sqlite3.connect(DB_PATH, timeout=20)
                    cursor = conn.cursor()
                    for msg in msgs:
                        if getattr(msg, 'empty', False) or not (msg.video or (msg.document and 'video' in str(msg.document.mime_type))): continue
                        text = msg.caption if msg.caption else ""
                        cat_name = extract_text(r"📁 Category:\s*(.+)", text)
                        creator_name = extract_text(r"👤 Creator:\s*(.+)", text)
                        likes = extract_num(r"❤️ Likes:\s*([\d,]+)", text)
                        cap_match = re.search(r"📝 Caption:\n(.*?)(?=\n🔗 Link:|$)", text, re.DOTALL)
                        clean_caption = cap_match.group(1).strip() if cap_match else ""

                        cursor.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (cat_name,))
                        cat_id = cursor.execute("SELECT id FROM categories WHERE name = ?", (cat_name,)).fetchone()[0]
                        cursor.execute("INSERT OR IGNORE INTO creators (username) VALUES (?)", (creator_name,))
                        creator_id = cursor.execute("SELECT id FROM creators WHERE username = ?", (creator_name,)).fetchone()[0]
                        
                        cursor.execute("INSERT OR IGNORE INTO posts (video_id, category_id, creator_id, likes, caption, status) VALUES (?, ?, ?, ?, ?, ?)", (msg.id, cat_id, creator_id, likes, clean_caption, "Metadata_Only"))
                    conn.commit()
                    conn.close()
                    await asyncio.sleep(2)
                except FloodWait as e: await asyncio.sleep(e.value)
                except Exception as e: pass

            while True:
                conn = sqlite3.connect(DB_PATH, timeout=20)
                cursor = conn.cursor()
                pending = cursor.execute("SELECT video_id, category_id FROM posts WHERE status = 'Metadata_Only' ORDER BY video_id DESC").fetchall()

                if not pending:
                    conn.close()
                    GLOBAL_STATUS = "✅ Sync Complete! Queue empty."
                    custom_print("✅ Sync Complete. Downloader going to sleep.")
                    CATEGORY_QUEUE.clear()
                    break

                target_vid = None
                target_cat_name = "Global"
                
                for queued_cat in CATEGORY_QUEUE:
                    cat_row = cursor.execute("SELECT id FROM categories WHERE name = ?", (queued_cat,)).fetchone()
                    if cat_row:
                        target_vid = next((r[0] for r in pending if r[1] == cat_row[0]), None)
                        if target_vid: 
                            target_cat_name = queued_cat
                            break

                if not target_vid and CATEGORY_QUEUE:
                    if CATEGORY_QUEUE[0] in CATEGORY_QUEUE:
                        CATEGORY_QUEUE.pop(0)
                        continue

                if not target_vid: 
                    target_vid = pending[0][0]
                    target_cat_name = "Global"
                
                conn.close()

                GLOBAL_STATUS = f"🚀 Fetching Video #{target_vid} [{target_cat_name}]"
                custom_print(f"⬇️ Downloading Video #{target_vid} ({target_cat_name})")
                
                await asyncio.sleep(1.5)
                try:
                    msg = await app_client.get_messages(CHANNEL_ID, target_vid)
                    target_video = os.path.join(VIDEO_DIR, f"video_{target_vid}.mp4")
                    await app_client.download_media(msg, file_name=target_video)
                    await ensure_web_safe(target_video)

                    conn = sqlite3.connect(DB_PATH, timeout=20)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE posts SET local_video_path = ?, status = 'Harvested' WHERE video_id = ?", (target_video, target_vid))
                    conn.commit()
                    conn.close()
                    push_job("QUEUE_VISION", {"msg_id": target_vid, "path": target_video})

                except FloodWait as e: await asyncio.sleep(e.value)
                except Exception as e:
                    conn = sqlite3.connect(DB_PATH, timeout=20)
                    conn.execute("UPDATE posts SET status = 'Error' WHERE video_id = ?", (target_vid,))
                    conn.commit()
                    conn.close()

        except Exception as e:
            GLOBAL_STATUS = f"❌ Worker Error: {str(e)[:50]}"
            await asyncio.sleep(5)

# ==========================================
# 3. FASTAPI SETUP & HIGH-FIDELITY ENDPOINTS
# ==========================================
app = FastAPI()
os.makedirs(VIDEO_DIR, exist_ok=True)

@app.get("/videos/{video_name}")
async def video_endpoint(request: Request, video_name: str):
    file_path = os.path.join(VIDEO_DIR, video_name)
    if not os.path.exists(file_path):
        return Response(status_code=404)
    
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range")
    
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=31536000, immutable",
        "Access-Control-Allow-Origin": "*",
    }
    
    if range_header:
        byte1, byte2 = 0, None
        match = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if match:
            byte1 = int(match.group(1))
            if match.group(2): byte2 = int(match.group(2))
        
        if byte2 is None: byte2 = file_size - 1
        length = byte2 - byte1 + 1
        
        def file_iterator():
            with open(file_path, "rb") as f:
                f.seek(byte1)
                remaining = length
                chunk_size = 2 * 1024 * 1024 
                while remaining > 0:
                    data = f.read(min(chunk_size, remaining))
                    if not data: break
                    remaining -= len(data)
                    yield data
                    
        headers["Content-Range"] = f"bytes {byte1}-{byte2}/{file_size}"
        headers["Content-Length"] = str(length)
        headers["Content-Type"] = "video/mp4"
        return StreamingResponse(file_iterator(), status_code=206, headers=headers)
    else:
        headers["Content-Length"] = str(file_size)
        headers["Content-Type"] = "video/mp4"
        return FileResponse(file_path, headers=headers)

@app.get("/api/status")
def get_status(): return {"status": GLOBAL_STATUS}

@app.post("/api/scan")
def trigger_scan(category: str = None):
    global CATEGORY_QUEUE
    if category and category != "All Categories":
        if category in CATEGORY_QUEUE:
            CATEGORY_QUEUE.remove(category)
        CATEGORY_QUEUE.insert(0, category) 
    SCAN_TRIGGER.set() 
    return {"message": "Scan triggered"}

@app.get("/api/categories")
def api_categories():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT c.name FROM categories c JOIN posts p ON c.id = p.category_id")
    cats = sorted([row[0] for row in cursor.fetchall()])
    conn.close()
    return {"categories": ["All Categories"] + cats, "queue": CATEGORY_QUEUE}

@app.get("/api/playlist")
def api_playlist(category: str = "All Categories"): return {"playlist": get_playlist_data(category)}

@app.get("/api/explore")
def api_explore(q: str = ""):
    conn = sqlite3.connect(DB_PATH, timeout=20)
    cursor = conn.cursor()
    
    # Expanded LIMIT to 1000 to feed the new IntersectionObserver Infinite Scroll
    if q.strip():
        cursor.execute('''
            SELECT p.video_id AS "ID", ps.creator AS "Creator", ps.category AS "Category", p.likes AS "Likes", ps.caption AS "Caption", p.local_video_path 
            FROM posts_search ps
            JOIN posts p ON ps.video_id = p.video_id
            WHERE ps.posts_search MATCH ? AND p.local_video_path IS NOT NULL
            ORDER BY ps.rank LIMIT 1000
        ''', (f"{q}*",))
    else:
        cursor.execute('''
            SELECT p.video_id AS "ID", cr.username AS "Creator", c.name AS "Category", p.likes AS "Likes", p.caption AS "Caption", p.local_video_path 
            FROM posts p JOIN creators cr ON p.creator_id = cr.id JOIN categories c ON p.category_id = c.id 
            WHERE p.local_video_path IS NOT NULL ORDER BY p.video_id DESC LIMIT 1000
        ''')
        
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        results.append({
            "id": r[0], "creator": r[1], "category": r[2],
            "likes": f"{r[3]:,}", "caption": r[4], 
            "filename": os.path.basename(r[5])
        })
    return {"data": results}

@app.post("/api/state")
def set_state(index: int):
    with open(STATE_FILE, 'w') as f: f.write(str(index))
    return {"success": True}

@app.get("/api/state")
def get_state():
    try:
        with open(STATE_FILE, 'r') as f: return {"index": int(f.read().strip())}
    except: return {"index": 0}

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_ui(id: str = "", t: str = ""):
    return f"""
    <html>
    <body style='background:#050505; color:white; font-family:sans-serif; padding:50px; text-align:center;'>
        <h2 style='color:#00e5ff; margin-bottom: 10px;'>🔍 Pattern Recognition Module</h2>
        <div style='background:#111; padding:30px; border-radius:16px; border:1px solid #333; display:inline-block; text-align:left;'>
            <p style='font-size:18px;'><b>Video ID:</b> {id}</p>
            <p style='font-size:18px;'><b>Exact Timestamp:</b> {t}s</p>
            <hr style='border-color:#333; margin: 20px 0;'/>
            <p style='color:#888; font-style:italic;'>Multi-modal analysis engines standing by...</p>
        </div>
    </body>
    </html>
    """

# ==========================================
# 4. FRONTEND UI & UX - COMMAND DECK WITH HEURISTICS
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Insta-Vault Enterprise</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #030303; --surface: #111111; --surface-hover: #1c1c1c;
                --accent: #00e5ff; --accent-dim: rgba(0, 229, 255, 0.15);
                --text-main: #ffffff; --text-dim: #999999; --radius: 18px;
            }

            * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
            body { background-color: var(--bg); color: var(--text-main); font-family: 'Inter', sans-serif; margin: 0; padding: 15px; overflow-x: hidden; touch-action: manipulation; }
            
            #toast-container { position: fixed; top: 20px; left: 50%; transform: translateX(-50%); z-index: 1000; display: flex; flex-direction: column; gap: 10px; align-items: center; pointer-events: none;}
            .toast { background: rgba(18, 18, 18, 0.95); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); color: #fff; padding: 14px 28px; border-radius: 50px; font-weight: 600; font-size: 14px; box-shadow: 0 10px 40px rgba(0,0,0,0.8); border: 1px solid #333; opacity: 0; transform: translateY(-20px); transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55); display: flex; align-items: center; gap: 8px;}
            .toast.show { opacity: 1; transform: translateY(0); }

            .tabs { display: flex; gap: 24px; margin-bottom: 20px; border-bottom: 1px solid #222; padding-bottom: 12px; justify-content: center; }
            .tab-btn { background: none; color: var(--text-dim); border: none; padding: 8px 12px; cursor: pointer; font-size: 16px; font-weight: 700; position: relative; transition: color 0.3s ease; outline: none; }
            .tab-btn::after { content: ''; position: absolute; bottom: -13px; left: 0; width: 0; height: 3px; background: var(--accent); transition: width 0.3s ease; border-radius: 3px 3px 0 0; box-shadow: 0 0 12px var(--accent);}
            .tab-btn.active { color: var(--text-main); text-shadow: 0 0 15px rgba(255,255,255,0.2);}
            .tab-btn.active::after { width: 100%; }
            .tab-content { display: none; }
            .tab-content.active { display: block; animation: fadeIn 0.3s ease; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

            .reel-card { max-width: 460px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
            
            .custom-select-wrapper { position: relative; width: 100%; user-select: none; z-index: 50; }
            .custom-select { display: flex; justify-content: space-between; align-items: center; background: rgba(17, 17, 17, 0.8); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); padding: 18px 22px; border-radius: var(--radius); border: 1px solid #2a2a2a; cursor: pointer; font-weight: 600; font-size: 15px; transition: all 0.2s ease; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
            .custom-select:hover { border-color: #444; background: rgba(25, 25, 25, 0.9); }
            .custom-select.open { border-color: var(--accent); border-bottom-left-radius: 0; border-bottom-right-radius: 0; box-shadow: 0 0 15px var(--accent-dim);}
            .custom-select-icon { transition: transform 0.3s ease; }
            .custom-select.open .custom-select-icon { transform: rotate(180deg); }
            
            .custom-options { position: absolute; top: 100%; left: 0; right: 0; background: rgba(17, 17, 17, 0.95); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px); border: 1px solid #2a2a2a; border-top: none; border-bottom-left-radius: var(--radius); border-bottom-right-radius: var(--radius); z-index: 100; max-height: 300px; overflow-y: auto; opacity: 0; visibility: hidden; transform: translateY(-5px); transition: all 0.2s ease; box-shadow: 0 20px 40px rgba(0,0,0,0.8); }
            .custom-select-wrapper.open .custom-options { opacity: 1; visibility: visible; transform: translateY(0); }
            .custom-option { padding: 15px 22px; font-weight: 500; font-size: 14px; color: #ccc; cursor: pointer; transition: all 0.2s; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 8px;}
            .custom-option:last-child { border-bottom: none; }
            .custom-option:hover { background: var(--surface-hover); color: var(--accent); padding-left:26px;}
            .custom-options::-webkit-scrollbar { width: 5px; }
            .custom-options::-webkit-scrollbar-track { background: transparent; }
            .custom-options::-webkit-scrollbar-thumb { background: #555; border-radius: 10px; }

            .player-wrapper { 
                position: relative; width: 100%; border-radius: var(--radius); overflow: hidden; 
                background: #000; border: 1px solid #222; box-shadow: 0 15px 40px rgba(0,0,0,0.8); 
                user-select: none; touch-action: pan-y; 
            }
            .reel-video { 
                width: 100%; max-height: 70vh; display: block; object-fit: contain; 
                cursor: pointer; will-change: transform, opacity; transition: transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275), opacity 0.2s; 
            }
            .reel-video.swipe-anim-left { transform: translateX(-20px) scale(0.97); opacity: 0.7; }
            .reel-video.swipe-anim-right { transform: translateX(20px) scale(0.97); opacity: 0.7; }
            .reel-video.tap-anim { filter: brightness(1.3); }

            .play-state-icon { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) scale(1); width: 64px; height: 64px; background: rgba(0,0,0,0.6); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); border-radius: 50%; display: flex; justify-content: center; align-items: center; pointer-events: none; opacity: 0; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); z-index: 10; border: 1px solid rgba(255,255,255,0.1); }
            .player-wrapper.paused .play-state-icon { opacity: 1; transform: translate(-50%, -50%) scale(1.1); }
            .player-wrapper.playing-anim .play-state-icon { opacity: 0; transform: translate(-50%, -50%) scale(1.5); }

            .player-controls { position: absolute; bottom: 0; left: 0; right: 0; background: linear-gradient(transparent, rgba(0,0,0,0.95)); padding: 25px 18px 18px; display: flex; flex-direction: column; gap: 12px; opacity: 0; transition: opacity 0.3s ease; z-index: 10; }
            .player-wrapper:hover .player-controls, .player-wrapper.paused .player-controls { opacity: 1; }
            
            .progress-area { width: 100%; height: 6px; background: rgba(255,255,255,0.25); border-radius: 4px; cursor: pointer; position: relative; overflow: hidden; transition: height 0.2s; }
            .progress-area:hover { height: 10px; }
            .progress-bar { height: 100%; background: var(--accent); width: 0%; pointer-events: none; transition: width 0.1s linear; box-shadow: 0 0 12px var(--accent);}
            
            .control-actions { display: flex; justify-content: space-between; align-items: center; }
            .control-group { display: flex; gap: 18px; align-items: center;}
            .icon-btn { background: none; border: none; color: white; cursor: pointer; padding: 0; display: flex; align-items: center; justify-content: center; transition: transform 0.2s, color 0.2s; outline: none; }
            .icon-btn:hover { transform: scale(1.15); color: var(--accent); }
            .time-display { font-size: 12px; font-weight: 700; color: #eee; font-variant-numeric: tabular-nums; text-shadow: 0 1px 4px rgba(0,0,0,0.9);}
            
            .info-box { 
                background: rgba(18, 18, 18, 0.8); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); padding: 20px; 
                border-radius: var(--radius); border-left: 3px solid var(--accent); border: 1px solid #222;
                max-height: 160px; overflow-y: auto; touch-action: pan-y; overscroll-behavior: contain;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .info-box::-webkit-scrollbar { width: 4px; }
            .info-box::-webkit-scrollbar-track { background: transparent; }
            .info-box::-webkit-scrollbar-thumb { background: #555; border-radius: 4px; }
            
            .info-box h3 { margin: 0 0 8px 0; font-size: 15px; color: #fff; display: flex; justify-content: space-between; align-items: center;}
            .info-box p { margin: 0; color: #aaa; white-space: pre-wrap; font-size: 13.5px; line-height: 1.6; font-weight: 400;}
            
            .command-deck { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
            .action-btn { 
                background: var(--surface); color: var(--text-main); border: 1px solid #2a2a2a; 
                border-radius: 14px; padding: 16px; cursor: pointer; font-weight: 600; font-size: 14px; 
                transition: all 0.2s; display: flex; justify-content: center; align-items: center; gap: 10px; outline:none; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
            }
            .action-btn:hover { background: var(--surface-hover); border-color: #444; transform: translateY(-2px);}
            .action-btn:active { transform: translateY(1px) scale(0.98); }
            
            .primary-btn { background: #fff; color: #000; border: none; font-weight: 800; font-size: 15px;}
            .primary-btn:hover { background: #e0e0e0; box-shadow: 0 8px 25px rgba(255,255,255,0.15); }
            
            .live-status-bar { background: var(--surface); color: var(--accent); border: 1px solid #2a2a2a; padding: 15px; border-radius: 14px; font-size: 12px; font-weight: 600; text-align: center; font-family: monospace; letter-spacing: -0.2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; box-shadow: 0 4px 15px rgba(0,0,0,0.2);}

            .explore-container { display: flex; gap: 24px; }
            .explore-left { flex: 2; }
            .explore-right { flex: 1; position: sticky; top: 20px; align-self: flex-start;}
            .search-row { display: flex; gap: 12px; margin-bottom: 20px; }
            .search-row input { flex: 3; padding: 18px 22px; background: var(--surface); color: white; border: 1px solid #2a2a2a; border-radius: 14px; outline: none; font-size: 15px; font-family: 'Inter'; transition: all 0.2s; box-shadow: 0 4px 15px rgba(0,0,0,0.2);}
            .search-row input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-dim); }
            
            table { width: 100%; border-collapse: separate; border-spacing: 0; background: var(--surface); border-radius: 14px; overflow: hidden; border: 1px solid #2a2a2a; box-shadow: 0 10px 30px rgba(0,0,0,0.4);}
            th { background-color: #0a0a0a; color: var(--text-dim); padding: 16px; text-align: left; position: sticky; top: 0; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;}
            td { padding: 16px; border-bottom: 1px solid #1a1a1a; color: #ccc; cursor: pointer; font-size: 14px; transition: background 0.2s;}
            tr:last-child td { border-bottom: none; }
            tr:hover td { background-color: #1a1a1a; color: #fff;}
            
            .sentinel-row { text-align: center; padding: 20px; color: var(--accent); font-weight: 600; }

            @media (max-width: 768px) { .explore-container { flex-direction: column; } .explore-right { position: relative; top: 0; } }
        </style>
    </head>
    <body>
        <div id="toast-container"></div>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('reels')">📱 View Reels</button>
            <button class="tab-btn" onclick="switchTab('explore')">🔍 Explore DB</button>
        </div>

        <div id="reels" class="tab-content active">
            <div class="reel-card">
                
                <div class="custom-select-wrapper" id="customSelectWrapper">
                    <div class="custom-select" id="customSelectTrigger">
                        <span id="customSelectLabel">🎵 Playing: <span style="color:var(--accent);">All Categories</span></span>
                        <svg class="custom-select-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    </div>
                    <div class="custom-options" id="customOptions"></div>
                </div>

                <div class="player-wrapper" id="playerWrapper">
                    <video id="reel-vid" class="reel-video" autoplay loop playsinline preload="auto" decoding="async"></video>
                    
                    <div class="play-state-icon" id="playStateIcon">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="white"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                    </div>
                    
                    <div class="player-controls">
                        <div class="progress-area" id="progressArea">
                            <div class="progress-bar" id="progressBar"></div>
                        </div>
                        <div class="control-actions">
                            <div class="control-group">
                                <button class="icon-btn" id="muteBtn" title="Mute (M)">
                                    <svg id="vol-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line></svg>
                                </button>
                                <span class="time-display" id="timeDisplay">0:00 / 0:00</span>
                            </div>
                            <div class="control-group">
                                <button class="icon-btn" id="copyBtn" onclick="copyCaption(event)" title="Copy Caption (C)">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                                </button>
                                <button class="icon-btn" id="fullscreenBtn" onclick="toggleFullscreen(event)" title="Fullscreen (F)">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="info-box">
                    <h3 id="user-display">Loading...</h3>
                    <p id="cap-display">...</p>
                </div>
                
                <div class="command-deck">
                    <button class="action-btn" onclick="goPrev()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"></polyline></svg>
                        Prev Video
                    </button>
                    <button class="action-btn primary-btn" onclick="goNext()">
                        Next Video
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="9 18 15 12 9 6"></polyline></svg>
                    </button>
                    <button class="action-btn" onclick="triggerScan()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="2"></circle><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"></path></svg>
                        Scan Web
                    </button>
                    <button class="action-btn" onclick="forceSyncFeed()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
                        Sync Data
                    </button>
                </div>

                <div class="live-status-bar" id="live-status">⏳ Cloud Server Booting...</div>
            </div>
        </div>

        <div id="explore" class="tab-content">
            <p style="color: var(--text-dim); font-size: 14px; margin-bottom: 20px;">Live FTS5 Search Index.</p>
            <div class="explore-container">
                <div class="explore-left">
                    <div class="search-row">
                        <input type="text" id="search-input" placeholder="Type to instantly filter captions, creators...">
                    </div>
                    <div style="overflow-x: auto; max-height: 70vh; overflow-y: auto; border-radius: 14px; border: 1px solid #222;" id="explore-table-container">
                        <table id="explore-table">
                            <thead><tr><th>ID</th><th>Creator</th><th>Category</th><th>Likes</th><th>Caption</th></tr></thead>
                            <tbody id="explore-tbody"></tbody>
                        </table>
                    </div>
                </div>
                <div class="explore-right">
                    <div class="player-wrapper" style="margin-bottom:0;">
                        <video id="explore-vid" class="reel-video" autoplay controls playsinline preload="auto" decoding="async"></video>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentPlaylist = [];
            let currentIndex = 0;
            let currentPlayingCat = localStorage.getItem('instaVaultCategory') || "All Categories"; 
            let pendingCategorySwitch = null;

            // Explore Tab Infinite Scroll Data
            let exploreData = [];
            let exploreRendered = 0;
            const RENDER_CHUNK = 30;

            // Ghost Player Memory Management
            const ghostPlayer = document.createElement('video');
            ghostPlayer.muted = true;
            ghostPlayer.preload = 'auto';

            function clearGhostMemory() {
                ghostPlayer.removeAttribute('src');
                ghostPlayer.load();
            }

            function showToast(message, duration = 3000) {
                const container = document.getElementById('toast-container');
                const toast = document.createElement('div');
                toast.className = 'toast';
                toast.innerHTML = message;
                container.appendChild(toast);
                void toast.offsetWidth;
                toast.classList.add('show');
                setTimeout(() => {
                    toast.classList.remove('show');
                    setTimeout(() => toast.remove(), 300);
                }, duration);
            }

            const vidPlayer = document.getElementById('reel-vid');
            const playerWrapper = document.getElementById('playerWrapper');
            const progressArea = document.getElementById('progressArea');
            const progressBar = document.getElementById('progressBar');
            const muteBtn = document.getElementById('muteBtn');
            const volIcon = document.getElementById('vol-icon');
            const timeDisplay = document.getElementById('timeDisplay');
            
            let isMuted = localStorage.getItem('instaVaultMuted') !== 'false'; 
            vidPlayer.muted = isMuted;
            updateMuteIcon();

            function togglePlay() {
                if (vidPlayer.paused) {
                    vidPlayer.play().catch(e => console.log("Autoplay blocked:", e));
                    playerWrapper.classList.remove('paused');
                    playerWrapper.classList.add('playing-anim');
                    setTimeout(() => playerWrapper.classList.remove('playing-anim'), 300);
                } else {
                    vidPlayer.pause();
                    playerWrapper.classList.add('paused');
                    playerWrapper.classList.remove('playing-anim');
                }
            }

            function updateMuteIcon() {
                if(isMuted) {
                    volIcon.innerHTML = `<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line>`;
                } else {
                    volIcon.innerHTML = `<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>`;
                }
            }

            muteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                isMuted = !isMuted;
                vidPlayer.muted = isMuted;
                localStorage.setItem('instaVaultMuted', isMuted);
                updateMuteIcon();
            });

            function formatTime(seconds) {
                if(isNaN(seconds)) return "0:00";
                let min = Math.floor(seconds / 60);
                let sec = Math.floor(seconds % 60);
                return `${min}:${sec < 10 ? '0' : ''}${sec}`;
            }

            vidPlayer.addEventListener('timeupdate', () => {
                if (!vidPlayer.duration) return;
                const percent = (vidPlayer.currentTime / vidPlayer.duration) * 100;
                progressBar.style.width = `${percent}%`;
                timeDisplay.innerText = `${formatTime(vidPlayer.currentTime)} / ${formatTime(vidPlayer.duration)}`;
            });

            progressArea.addEventListener('click', (e) => {
                e.stopPropagation();
                const rect = progressArea.getBoundingClientRect();
                const pos = (e.clientX - rect.left) / rect.width;
                vidPlayer.currentTime = pos * vidPlayer.duration;
            });

            function toggleFullscreen(e) {
                e.stopPropagation();
                if (!document.fullscreenElement) {
                    playerWrapper.requestFullscreen().catch(err => {
                        showToast("Error attempting to enable fullscreen.");
                    });
                } else {
                    document.exitFullscreen();
                }
            }

            function copyCaption(e) {
                e.stopPropagation();
                const text = document.getElementById('cap-display').innerText;
                navigator.clipboard.writeText(text).then(() => {
                    showToast("📋 Caption Copied to Clipboard!");
                });
            }

            function extractCurrentVideo() {
                const vid = currentPlaylist[currentIndex];
                const t = vidPlayer.currentTime.toFixed(2);
                showToast(`🔍 Opening Analysis for Video #${vid.id}`);
                window.open(`/analyze?id=${vid.id}&t=${t}`, '_blank');
            }

            // --- POINTER ENGINE ---
            let startX = 0, startY = 0, startTime = 0;
            let isSwiping = false;
            let pressTimer = null;
            let clickCount = 0;
            let clickTimer = null;

            playerWrapper.addEventListener('pointerdown', e => {
                if(e.target.closest('.player-controls')) return;
                startX = e.clientX;
                startY = e.clientY;
                startTime = Date.now();
                isSwiping = false;
                
                pressTimer = setTimeout(() => {
                    if(!isSwiping) {
                        vidPlayer.playbackRate = 2.0;
                        showToast("⚡ 2x Fast Forward", 1000);
                    }
                }, 400);
            });

            playerWrapper.addEventListener('pointerup', e => {
                if(e.target.closest('.player-controls')) return;
                clearTimeout(pressTimer);
                
                if (vidPlayer.playbackRate > 1.0) {
                    vidPlayer.playbackRate = 1.0;
                    return; 
                }

                let deltaX = startX - e.clientX;
                let deltaY = startY - e.clientY;
                let duration = Date.now() - startTime;

                if (Math.abs(deltaX) > 40 && Math.abs(deltaX) > Math.abs(deltaY)) {
                    isSwiping = true;
                    if (deltaX > 0) {
                        vidPlayer.classList.add('swipe-anim-left');
                        setTimeout(() => vidPlayer.classList.remove('swipe-anim-left'), 150);
                        setTimeout(goNext, 50);
                    } else {
                        vidPlayer.classList.add('swipe-anim-right');
                        setTimeout(() => vidPlayer.classList.remove('swipe-anim-right'), 150);
                        setTimeout(goPrev, 50);
                    }
                    return;
                }

                if (!isSwiping && duration < 400 && Math.abs(deltaX) < 10 && Math.abs(deltaY) < 10) {
                    clickCount++;
                    if (clickCount === 1) {
                        clickTimer = setTimeout(() => {
                            togglePlay();
                            clickCount = 0;
                        }, 250);
                    } else if (clickCount === 2) {
                        clearTimeout(clickTimer);
                        extractCurrentVideo();
                        clickCount = 0;
                    }
                }
            });

            playerWrapper.addEventListener('pointercancel', e => {
                clearTimeout(pressTimer);
                vidPlayer.playbackRate = 1.0;
            });

            // Keyboard Macros
            window.addEventListener('keydown', e => {
                if (document.activeElement.tagName === 'INPUT') return;
                if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
                if (e.code === 'KeyM') { e.preventDefault(); muteBtn.click(); }
                if (e.code === 'KeyF') { e.preventDefault(); toggleFullscreen(e); }
                if (e.code === 'KeyC') { e.preventDefault(); copyCaption(e); }
                if (e.code === 'KeyA') { e.preventDefault(); extractCurrentVideo(); }
                if (e.code === 'ArrowDown' || e.code === 'ArrowRight') { 
                    e.preventDefault(); 
                    vidPlayer.classList.add('swipe-anim-left');
                    setTimeout(() => vidPlayer.classList.remove('swipe-anim-left'), 150);
                    setTimeout(goNext, 50); 
                }
                if (e.code === 'ArrowUp' || e.code === 'ArrowLeft') { 
                    e.preventDefault(); 
                    vidPlayer.classList.add('swipe-anim-right');
                    setTimeout(() => vidPlayer.classList.remove('swipe-anim-right'), 150);
                    setTimeout(goPrev, 50); 
                }
            });

            const selectWrapper = document.getElementById('customSelectWrapper');
            const selectTrigger = document.getElementById('customSelectTrigger');
            const selectLabel = document.getElementById('customSelectLabel');
            const optionsContainer = document.getElementById('customOptions');

            selectTrigger.addEventListener('click', (e) => {
                e.stopPropagation(); 
                selectWrapper.classList.toggle('open');
                selectTrigger.classList.toggle('open');
            });

            document.addEventListener('click', (e) => {
                if (!selectWrapper.contains(e.target)) {
                    selectWrapper.classList.remove('open');
                    selectTrigger.classList.remove('open');
                }
            });

            function populateCustomDropdown(categories, queueArray) {
                optionsContainer.innerHTML = '';
                let queuedCats = [];
                let unqueuedCats = [];
                
                categories.forEach(cat => {
                    let qIndex = queueArray.indexOf(cat);
                    if (qIndex !== -1) { queuedCats.push({name: cat, rank: qIndex}); } 
                    else { unqueuedCats.push(cat); }
                });
                
                queuedCats.sort((a, b) => a.rank - b.rank);
                let finalCats = queuedCats.map(c => c.name).concat(unqueuedCats);

                finalCats.forEach(cat => {
                    const opt = document.createElement('div');
                    opt.className = 'custom-option';
                    let qIndex = queueArray.indexOf(cat);
                    
                    if (qIndex !== -1) {
                        opt.innerHTML = `<span style="color:var(--accent); font-weight:800; min-width:24px; font-size:13px;">${qIndex + 1}.</span> <span>${cat}</span>`;
                        opt.style.backgroundColor = 'rgba(0,229,255,0.05)';
                    } else {
                        opt.innerHTML = `<span style="min-width:24px;"></span> <span>${cat}</span>`;
                    }

                    opt.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        selectWrapper.classList.remove('open');
                        selectTrigger.classList.remove('open');
                        
                        await fetch(`/api/scan?category=${encodeURIComponent(cat)}`, { method: 'POST' });
                        showToast(`🚀 Queued at #1 Priority: ${cat}`);
                        
                        pendingCategorySwitch = cat;
                        loadCategories(); 
                    });
                    optionsContainer.appendChild(opt);
                });
                selectLabel.innerHTML = `🎵 Playing: <span style="color:var(--accent);">${currentPlayingCat}</span>`;
            }

            function switchTab(tabId) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
                event.currentTarget.classList.add('active');
                if(tabId === 'explore' && document.getElementById('explore-tbody').innerHTML === '') searchExplore();
            }

            async function init() {
                loadCategories();
                const stateRes = await fetch('/api/state');
                const stateData = await stateRes.json();
                currentIndex = stateData.index || 0;
                
                await fetchPlaylist(currentPlayingCat);
                
                setInterval(updateStatus, 1500); 
                setInterval(autoSyncFeed, 3000); 
            }

            async function updateStatus() {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('live-status').innerText = data.status;
            }

            async function loadCategories() {
                const res = await fetch('/api/categories');
                const data = await res.json();
                populateCustomDropdown(data.categories, data.queue);
            }

            async function fetchPlaylist(category) {
                const res = await fetch(`/api/playlist?category=${encodeURIComponent(category)}`);
                const data = await res.json();
                currentPlaylist = data.playlist;
                loadVideo(currentIndex);
            }

            function preloadNextVideo(index) {
                clearGhostMemory(); // Free up previous chunk
                if (index + 1 < currentPlaylist.length) {
                    const nextVidFile = currentPlaylist[index + 1].filename;
                    ghostPlayer.src = "/videos/" + nextVidFile;
                    ghostPlayer.load();
                }
            }

            function loadVideo(index) {
                if (!currentPlaylist || currentPlaylist.length === 0) return;
                if (index >= currentPlaylist.length) return; 
                if (index < 0) index = 0;
                
                const video = currentPlaylist[index];
                currentIndex = index;

                document.getElementById('user-display').innerHTML = `<span style="font-weight:700;">👤 @${video.username}</span><span style="font-size:12px; color:var(--accent); font-weight: 700; letter-spacing:0.5px; text-transform:uppercase;">📁 ${video.name} &nbsp;•&nbsp; ❤️ ${video.likes}</span>`;
                document.getElementById('cap-display').innerText = video.caption;
                
                const targetSrc = "/videos/" + video.filename;
                
                if (!vidPlayer.src.endsWith(encodeURI(video.filename))) {
                    vidPlayer.src = targetSrc;
                    playerWrapper.classList.remove('paused');
                    
                    const playPromise = vidPlayer.play();
                    if (playPromise !== undefined) {
                        playPromise.catch(error => {
                            playerWrapper.classList.add('paused');
                        });
                    }
                }
                
                preloadNextVideo(currentIndex);
                fetch(`/api/state?index=${currentIndex}`, { method: 'POST' });
            }

            function goNext() { 
                if (currentIndex + 1 >= currentPlaylist.length) {
                    showToast("⏳ End of list. Queue working in background...");
                    return; 
                }
                loadVideo(currentIndex + 1); 
            }
            function goPrev() { loadVideo(currentIndex - 1); }

            async function forceSyncFeed() {
                await autoSyncFeed();
                showToast("✅ Web Feed Synced.");
            }

            async function autoSyncFeed() {
                if (pendingCategorySwitch) {
                    const pRes = await fetch(`/api/playlist?category=${encodeURIComponent(pendingCategorySwitch)}`);
                    const pData = await pRes.json();
                    
                    if (pData.playlist && pData.playlist.length > 0) {
                        currentPlaylist = pData.playlist;
                        currentIndex = 0;
                        currentPlayingCat = pendingCategorySwitch;
                        localStorage.setItem('instaVaultCategory', currentPlayingCat); 
                        pendingCategorySwitch = null;
                        
                        loadVideo(0);
                        showToast(`✅ Now Playing: ${currentPlayingCat}`);
                        loadCategories(); 
                    }
                } else {
                    const res = await fetch(`/api/playlist?category=${encodeURIComponent(currentPlayingCat)}`);
                    const data = await res.json();
                    
                    if (data.playlist && data.playlist.length > currentPlaylist.length) {
                        let wasEmpty = currentPlaylist.length === 0;
                        currentPlaylist = data.playlist;
                        if (wasEmpty) {
                            currentIndex = 0;
                            loadVideo(0);
                        }
                    }
                }
                loadCategories(); 
            }

            async function triggerScan() {
                await fetch('/api/scan', { method: 'POST' });
                showToast("📡 Forced Full Telegram Scan.");
            }

            // --- INTERSECTION OBSERVER FOR INFINITE SCROLL ---
            let exploreObserver = new IntersectionObserver((entries) => {
                if(entries[0].isIntersecting) {
                    renderExploreChunk();
                }
            });

            let debounceTimer;
            document.getElementById('search-input').addEventListener('input', (e) => {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => searchExplore(e.target.value), 200);
            });

            async function searchExplore(query = "") {
                const res = await fetch(`/api/explore?q=${encodeURIComponent(query)}`);
                const data = await res.json();
                exploreData = data.data;
                document.getElementById('explore-tbody').innerHTML = '';
                exploreRendered = 0;
                renderExploreChunk();
            }

            function renderExploreChunk() {
                const tbody = document.getElementById('explore-tbody');
                const fragment = document.createDocumentFragment();
                const end = Math.min(exploreRendered + RENDER_CHUNK, exploreData.length);
                
                const oldSentinel = document.getElementById('sentinel-row');
                if (oldSentinel) oldSentinel.remove();

                for (let i = exploreRendered; i < end; i++) {
                    const row = exploreData[i];
                    const tr = document.createElement('tr');
                    tr.onclick = () => playExploreVideo(row.filename, row.id);
                    tr.innerHTML = `<td>${row.id}</td><td style="font-weight:600;">@${row.creator}</td><td><span style="background:#222; padding:4px 8px; border-radius:6px; font-size:12px; font-weight:600; color:var(--accent);">${row.category}</span></td><td style="font-weight:600;">${row.likes}</td><td>${row.caption.substring(0,60)}...</td>`;
                    fragment.appendChild(tr);
                }
                
                exploreRendered = end;
                
                if (exploreRendered < exploreData.length) {
                    const sentinel = document.createElement('tr');
                    sentinel.id = 'sentinel-row';
                    sentinel.innerHTML = `<td colspan="5" class="sentinel-row">Fetching more records...</td>`;
                    fragment.appendChild(sentinel);
                }
                
                tbody.appendChild(fragment);
                
                const newSentinel = document.getElementById('sentinel-row');
                if (newSentinel) exploreObserver.observe(newSentinel);
            }

            function playExploreVideo(filename, id) {
                document.getElementById('explore-vid').src = "/videos/" + filename;
            }

            init();
            searchExplore();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    init_db()
    
    def start_cloudflared():
        time.sleep(2) 
        cmd = ["./cloudflared", "tunnel", "--url", "http://localhost:8000"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if "trycloudflare.com" in line:
                url = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if url:
                    print("\n" + "="*60)
                    print(f"🌍 WEB APP IS LIVE AT: {url.group(0)}")
                    print("="*60 + "\n")
                    break

    threading.Thread(target=start_cloudflared, daemon=True).start()

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(background_downloader())
    except KeyboardInterrupt:
        custom_print("🛑 Process manually stopped by user.")
    except Exception as e:
        custom_print(f"❌ Main loop crashed: {e}")
