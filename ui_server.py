import os, sqlite3, aiosqlite, redis, re, builtins, time, asyncio, urllib.request, json, uvicorn, threading, subprocess
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
import logging
import aiofiles
from fastapi.staticfiles import StaticFiles
from pyrogram import Client
from pyrogram.errors import FloodWait
import nest_asyncio
from queue_manager import push_job, get_queue_metrics, get_queue_depth, replay_dlq
from v17_backend import v17_router
from admin_backend import admin_router
from explorer_backend import explorer_router, rpc_call_sync
from db_schema import init_sqlite_schema
from config import BASE_DIR, LAKE_DIR, DB_PATH, VIDEO_DIR, SESSION_DIR, STATE_FILE, THUMB_DIR, SQLITE_TIMEOUT

nest_asyncio.apply()

logger = logging.getLogger("VIOS")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S'))
    logger.addHandler(ch)

class WsManager:
    def __init__(self): self.clients = []
    async def connect(self, ws: WebSocket): await ws.accept(); self.clients.append(ws)
    def disconnect(self, ws: WebSocket): self.clients.remove(ws)
    async def broadcast(self, msg: dict):
        for c in self.clients.copy():
            try: await c.send_json(msg)
            except: pass
ws_mgr = WsManager()

def custom_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logger.info(msg)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(ws_mgr.broadcast({"type": "log", "message": f"[{time.strftime('%H:%M:%S')}] {msg}"}))
    except: pass

# --- CREDENTIALS (single source of truth: config.py, env-overridable) ---
from config import API_ID, API_HASH, BOT_TOKEN, CHANNEL_ID

# --- DIRECTORY ROUTING (from config.py) ---
SESSION_PATH = os.path.join(LAKE_DIR, 'bot_session')

for j_file in [f"{SESSION_PATH}.session-journal", f"{DB_PATH}-journal", f"{DB_PATH}-wal"]:
    if os.path.exists(j_file):
        try: os.remove(j_file)
        except: pass

GLOBAL_STATUS = "⏳ Omega Server Booting..."
CATEGORY_QUEUE = []
try: REDIS_CACHE = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
except: REDIS_CACHE = None
SCAN_TRIGGER = threading.Event()
SCAN_TRIGGER.set()

# ── Live category-queue sync (Admin Panel ⇄ Ghost Worker) ──
# The admin panel writes ADMIN_ACTIVE_CATEGORIES + bumps ADMIN_CATEGORIES_UPDATED.
# We track the last-seen update stamp so changes apply MID-CYCLE (not just at
# the start of a scan), fixing the "category control not working" bug.
_LAST_CAT_SYNC_STAMP = ""

def sync_category_queue(force=False):
    """Refresh CATEGORY_QUEUE from Redis. Cheap: only re-parses when the
    admin panel bumped ADMIN_CATEGORIES_UPDATED (or when force=True)."""
    global CATEGORY_QUEUE, _LAST_CAT_SYNC_STAMP
    if not REDIS_CACHE:
        return
    try:
        stamp = REDIS_CACHE.get("ADMIN_CATEGORIES_UPDATED") or ""
        if not force and stamp == _LAST_CAT_SYNC_STAMP:
            return
        _LAST_CAT_SYNC_STAMP = stamp
        raw = REDIS_CACHE.get("ADMIN_ACTIVE_CATEGORIES")
        new_queue = json.loads(raw) if raw else []
        if isinstance(new_queue, list) and new_queue != CATEGORY_QUEUE:
            CATEGORY_QUEUE = new_queue
            custom_print(f"📂 Admin category sync: {CATEGORY_QUEUE or '(cleared)'}")
            # Wake the downloader so a new priority applies instantly
            SCAN_TRIGGER.set()
    except Exception:
        pass

def persist_category_queue():
    """Write the worker's local queue state back to Redis so the admin panel
    always reflects reality (fixes the stale one-way sync)."""
    global _LAST_CAT_SYNC_STAMP
    if not REDIS_CACHE:
        return
    try:
        REDIS_CACHE.set("ADMIN_ACTIVE_CATEGORIES", json.dumps(CATEGORY_QUEUE))
        stamp = str(time.time())
        REDIS_CACHE.set("ADMIN_CATEGORIES_UPDATED", stamp)
        _LAST_CAT_SYNC_STAMP = stamp
    except Exception:
        pass

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS categories (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
    cursor.execute("CREATE TABLE IF NOT EXISTS creators (id INTEGER PRIMARY KEY, username TEXT UNIQUE)")
    cursor.execute('''CREATE TABLE IF NOT EXISTS posts (video_id INTEGER PRIMARY KEY, category_id INTEGER, creator_id INTEGER, likes INTEGER, caption TEXT, local_video_path TEXT, status TEXT)''')
    cursor.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS posts_search USING fts5(video_id UNINDEXED, caption, creator, category)''')
    cursor.execute('''CREATE TRIGGER IF NOT EXISTS sync_posts_search AFTER INSERT ON posts BEGIN INSERT INTO posts_search(video_id, caption, creator, category) VALUES (new.video_id, new.caption, (SELECT username FROM creators WHERE id = new.creator_id), (SELECT name FROM categories WHERE id = new.category_id)); END;''')
    conn.commit()
    conn.close()

async def get_playlist_data(category):
    async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
        base_query = '''SELECT cr.username, c.name, p.likes, p.caption, p.local_video_path, p.video_id FROM posts p JOIN creators cr ON p.creator_id = cr.id JOIN categories c ON p.category_id = c.id WHERE p.local_video_path IS NOT NULL'''
        if category and category != "All Categories":
            base_query += " AND c.name = ? ORDER BY p.video_id DESC"
            async with conn.execute(base_query, (category,)) as cursor:
                rows = await cursor.fetchall()
        else:
            base_query += " ORDER BY p.video_id DESC"
            async with conn.execute(base_query) as cursor:
                rows = await cursor.fetchall()
        
    playlist = []
    for row in rows:
        filename = os.path.basename(row[4]) if row[4] else ""
        playlist.append({'username': row[0], 'name': row[1], 'likes': row[2], 'caption': row[3], 'filename': filename, 'id': row[5]})
    return playlist

def extract_text(pattern, text):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else "Unknown"

def extract_num(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1).replace(",", "").strip()) if match else 0

ffmpeg_semaphore = asyncio.Semaphore(2)

async def ensure_web_safe(video_path):
    try:
        proc = await asyncio.create_subprocess_exec('ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', video_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        codec = stdout.decode().strip().split('\n')[0].lower()
        temp_path = video_path + ".tmp.mp4"

        async with ffmpeg_semaphore:
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
        await ping.delete()
    except Exception:
        GLOBAL_STATUS = "⚙️ Auto-Resolving Telegram Cache..."
        amnesia_event = asyncio.Event()
        @app_client.on_message()
        async def handler(client, message):
            if message.chat and message.chat.id == CHANNEL_ID: amnesia_event.set()
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            payload = json.dumps({"chat_id": CHANNEL_ID, "text": ".", "disable_notification": True}).encode('utf-8')
            req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req)
            await asyncio.wait_for(amnesia_event.wait(), timeout=10.0)
            ping = await app_client.send_message(CHANNEL_ID, ".", disable_notification=True)
            latest_id = ping.id
            await ping.delete()
        except Exception as e:
            GLOBAL_STATUS = "❌ Bot access denied."
            await asyncio.sleep(10)
            return

    import shutil
    while True:
        free_gb = shutil.disk_usage(VIDEO_DIR).free / (1024**3)
        if free_gb < 1.0:
            GLOBAL_STATUS = f"⚠️ Storage Critical ({free_gb:.1f}GB left) - Pausing"
            await asyncio.sleep(60)
            continue

        if not SCAN_TRIGGER.is_set():
            GLOBAL_STATUS = "💤 Idling. Background queue is empty."
            while not SCAN_TRIGGER.is_set(): await asyncio.sleep(1)

        SCAN_TRIGGER.clear()

        sync_category_queue(force=True)

        GLOBAL_STATUS = "⚡ Running Flash Metadata Scan across Telegram..."
        custom_print("⚡ Scanning Telegram for new Posts...")

        try:
            ping = await app_client.send_message(CHANNEL_ID, ".", disable_notification=True)
            latest_id = ping.id
            await ping.delete()

            # One query + set difference in Python (was one SELECT per message
            # ID — O(N) round-trips that froze the scan on large channels)
            async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
                async with conn.execute("SELECT video_id FROM posts") as cursor:
                    known_ids = {row[0] for row in await cursor.fetchall()}
            missing_ids = [i for i in range(latest_id, 0, -1) if i not in known_ids]

            BATCH_SIZE = 200
            for i in range(0, len(missing_ids), BATCH_SIZE):
                chunk = missing_ids[i:i + BATCH_SIZE]
                try:
                    msgs = await app_client.get_messages(CHANNEL_ID, chunk)
                    if not isinstance(msgs, list): msgs = [msgs]
                    async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
                        for msg in msgs:
                            if getattr(msg, 'empty', False) or not (msg.video or (msg.document and 'video' in str(msg.document.mime_type))): continue
                            text = msg.caption if msg.caption else ''
                            cat_name = extract_text(r'📁 Category:\s*(.+)', text)
                            creator_name = extract_text(r'👤 Creator:\s*(.+)', text)
                            likes = extract_num(r'❤️ Likes:\s*([\d,]+)', text)
                            cap_match = re.search(r'📝 Caption:\n(.*?)(?=\n🔗 Link:|$)', text, re.DOTALL)
                            clean_caption = cap_match.group(1).strip() if cap_match else ''

                            await conn.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (cat_name,))
                            async with conn.execute('SELECT id FROM categories WHERE name = ?', (cat_name,)) as cursor:
                                cat_row = await cursor.fetchone()
                                cat_id = cat_row[0] if cat_row else 1
                            await conn.execute('INSERT OR IGNORE INTO creators (username) VALUES (?)', (creator_name,))
                            async with conn.execute('SELECT id FROM creators WHERE username = ?', (creator_name,)) as cursor:
                                creator_row = await cursor.fetchone()
                                creator_id = creator_row[0] if creator_row else 1
                            await conn.execute('INSERT OR IGNORE INTO posts (video_id, category_id, creator_id, likes, caption, status) VALUES (?, ?, ?, ?, ?, ?)', (msg.id, cat_id, creator_id, likes, clean_caption, 'Metadata_Only'))
                        await conn.commit()
                    await asyncio.sleep(2)
                except FloodWait as e: await asyncio.sleep(e.value)
                except Exception as e: 
                    custom_print(f"⚠️ Scan Network Error: {str(e)[:50]}")
                    await asyncio.sleep(5)

            while True:
                # Pick up admin panel changes IMMEDIATELY (mid-cycle), not
                # only at the start of a scan. This is the category-control fix.
                sync_category_queue()

                async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
                    async with conn.execute("SELECT video_id, category_id FROM posts WHERE status = 'Metadata_Only' ORDER BY video_id DESC") as cursor:
                        pending = await cursor.fetchall()
                    if not pending:
                        GLOBAL_STATUS = "✅ Sync Complete! Queue empty."
                        custom_print("✅ Sync Complete. Downloader going to sleep.")
                        CATEGORY_QUEUE.clear()
                        persist_category_queue()  # keep admin panel in sync
                        break
                    target_vid = None
                    target_cat_name = "Global"
                    exhausted_cats = []
                    for queued_cat in CATEGORY_QUEUE:
                        async with conn.execute("SELECT id FROM categories WHERE name = ?", (queued_cat,)) as cursor:
                            cat_row = await cursor.fetchone()
                        if cat_row:
                            target_vid = next((r[0] for r in pending if r[1] == cat_row[0]), None)
                            if target_vid:
                                target_cat_name = queued_cat
                                break
                            exhausted_cats.append(queued_cat)
                        else:
                            exhausted_cats.append(queued_cat)
                    # Remove ONLY the categories that have nothing pending
                    # (old code blindly popped index 0, which could drop a
                    # still-active category) — and persist to Redis so the
                    # admin panel reflects the change.
                    if exhausted_cats and not target_vid:
                        for c in exhausted_cats:
                            if c in CATEGORY_QUEUE:
                                CATEGORY_QUEUE.remove(c)
                                custom_print(f"📂 Category '{c}' exhausted — removed from queue")
                        persist_category_queue()
                        continue
                    if not target_vid:
                        target_vid = pending[0][0]
                        target_cat_name = "Global" 

                GLOBAL_STATUS = f"🚀 Fetching Video #{target_vid} [{target_cat_name}]"
                custom_print(f"⬇️ Downloading Video #{target_vid} ({target_cat_name})")
                await asyncio.sleep(1.5)
                try:
                    msg = await app_client.get_messages(CHANNEL_ID, target_vid)
                    target_video = os.path.join(VIDEO_DIR, f"video_{target_vid}.mp4")
                    await app_client.download_media(msg, file_name=target_video)
                    await ensure_web_safe(target_video)

                    async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
                        await conn.execute("UPDATE posts SET local_video_path = ?, status = 'Harvested' WHERE video_id = ?", (target_video, target_vid))
                        await conn.commit()

                    # Any admin-activated category gets CV priority (old code
                    # only honored position 0, so 2nd/3rd active categories
                    # silently lost their priority flag)
                    is_active_cat = target_cat_name in CATEGORY_QUEUE
                    if REDIS_CACHE and REDIS_CACHE.sadd("PROCESSED_VIDEOS_SET", target_vid):
                        push_job("QUEUE_VISION", {"msg_id": target_vid, "path": target_video}, is_priority=is_active_cat)
                        custom_print(f"📤 Queued #{target_vid} for CV Engine ({'🔴 PRIORITY' if is_active_cat else '⚪ DEFAULT'})")
                    elif not REDIS_CACHE:
                        push_job("QUEUE_VISION", {"msg_id": target_vid, "path": target_video}, is_priority=is_active_cat)
                        custom_print(f"📤 Queued #{target_vid} for CV Engine (no Redis)")
                    
                    await asyncio.sleep(4)

                except FloodWait as e: 
                    custom_print(f"⚠️ FloodWait: Sleeping for {e.value}s")
                    await asyncio.sleep(e.value + 2)
                except Exception as e:
                    custom_print(f"⚠️ Network Error on #{target_vid}: {str(e)[:50]}")
                    async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
                        await conn.execute("UPDATE posts SET status = 'Error' WHERE video_id = ?", (target_vid,))
                        await conn.commit()
                    await asyncio.sleep(15)

        except Exception as e:
            GLOBAL_STATUS = f"❌ Worker Error: {str(e)[:50]}"
            await asyncio.sleep(5)

# --- FASTAPI APP ---
app = FastAPI(title="Insta-Vault Modular OS")

app.mount("/data", StaticFiles(directory=VIDEO_DIR), name="v17_data")
app.mount("/thumbs", StaticFiles(directory=os.path.join(LAKE_DIR, '.thumbnails')), name="v17_thumbs")
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="main_videos")

app.include_router(v17_router)
app.include_router(admin_router)
app.include_router(explorer_router)

@app.get("/api/status")
def get_status():
    try:
        metrics = get_queue_metrics("QUEUE_VISION")
    except:
        metrics = {}
    # Disk usage
    try:
        import shutil
        disk = shutil.disk_usage(LAKE_DIR)
        free_gb = f"{disk.free / (1024**3):.1f} GB"
    except:
        free_gb = "N/A"
    return {"status": GLOBAL_STATUS, "queue": metrics, "disk_free": free_gb}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_mgr.connect(ws)
    async def push_status():
        last_s = ""
        tick = 0
        seq = 0  # monotonic sequence — lets the client drop out-of-order frames
        while True:
            gs = globals().get("GLOBAL_STATUS", "")
            if gs != last_s:
                last_s = gs
                try: await ws.send_json({"type":"status", "message":gs})
                except: break
            # Broadcast ALL queue metrics every 5 seconds (25 × 0.2s)
            tick += 1
            if tick % 25 == 0:
                try:
                    m = await asyncio.to_thread(get_queue_metrics)
                    seq += 1
                    await ws.send_json({"type": "queues", "seq": seq,
                                        "ts": time.time(), "data": m})
                    # Legacy message shape kept for older UI code paths
                    await ws.send_json({"type": "queue", "data": m.get("QUEUE_VISION", {})})
                except: pass
            await asyncio.sleep(0.2)
    t = asyncio.create_task(push_status())
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        t.cancel()
        ws_mgr.disconnect(ws)

@app.post("/api/prioritize/{video_id}")
def prioritize_video(video_id: int):
    target_video = os.path.join(VIDEO_DIR, f"video_{video_id}.mp4")
    if not os.path.exists(target_video):
        return {"success": False, "message": "Video file not found on disk"}
    # Clear from dedup set so it can be re-processed
    if REDIS_CACHE:
        REDIS_CACHE.srem("PROCESSED_VIDEOS_SET", video_id)
    push_job("QUEUE_VISION", {"msg_id": video_id, "path": target_video}, is_priority=True)
    return {"success": True, "message": f"Video #{video_id} queued with PRIORITY"}

@app.get("/api/queue")
def api_queue_metrics():
    """Full queue system observability endpoint."""
    try:
        return get_queue_metrics()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/queue/replay-dlq")
def api_replay_dlq():
    """Move all dead-lettered jobs back to the default queue for reprocessing."""
    count = replay_dlq("QUEUE_VISION")
    return {"replayed": count, "message": f"Replayed {count} dead-lettered jobs"}
@app.post("/api/scan")
def trigger_scan(category: str = None):
    global CATEGORY_QUEUE
    if category and category != "All Categories":
        if category in CATEGORY_QUEUE: CATEGORY_QUEUE.remove(category)
        CATEGORY_QUEUE.insert(0, category)
        persist_category_queue()  # keep admin panel in sync
    SCAN_TRIGGER.set(); return {"message": "Scan triggered"}
@app.get("/api/categories")
async def api_categories():
    async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
        async with conn.execute("SELECT DISTINCT c.name FROM categories c JOIN posts p ON c.id = p.category_id") as cursor:
            cats = sorted([row[0] for row in await cursor.fetchall()])
    return {"categories": ["All Categories"] + cats, "queue": CATEGORY_QUEUE}
@app.get("/api/playlist")
async def api_playlist(category: str = "All Categories"): return {"playlist": await get_playlist_data(category)}
@app.get("/api/explore")
async def api_explore(q: str = "", offset: int = 0):
    async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
        if q.strip():
            async with conn.execute('''SELECT p.video_id AS "ID", ps.creator AS "Creator", ps.category AS "Category", p.likes AS "Likes", ps.caption AS "Caption", p.local_video_path FROM posts_search ps JOIN posts p ON ps.video_id = p.video_id WHERE ps.posts_search MATCH ? AND p.local_video_path IS NOT NULL ORDER BY ps.rank LIMIT 30 OFFSET ?''', (f"{q}*", offset)) as cursor:
                rows = await cursor.fetchall()
        else:
            async with conn.execute('''SELECT p.video_id AS "ID", cr.username AS "Creator", c.name AS "Category", p.likes AS "Likes", p.caption AS "Caption", p.local_video_path FROM posts p JOIN creators cr ON p.creator_id = cr.id JOIN categories c ON p.category_id = c.id WHERE p.local_video_path IS NOT NULL ORDER BY p.video_id DESC LIMIT 30 OFFSET ?''', (offset,)) as cursor:
                rows = await cursor.fetchall()
    return {"data": [{"id": r[0], "creator": r[1], "category": r[2], "likes": f"{r[3]:,}", "caption": r[4], "filename": os.path.basename(r[5])} for r in rows]}
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
    return f"<html><body style='background:#050505; color:white; font-family:sans-serif; padding:50px; text-align:center;'><h2 style='color:#00e5ff;'>🔍 Pattern Recognition Module</h2><p><b>Video ID:</b> {id}</p><p><b>Exact Timestamp:</b> {t}s</p></body></html>"

# ═══════════════════════════════════════════════════════════
# LAYER-5 INTELLIGENCE APIs
# ═══════════════════════════════════════════════════════════

@app.get("/api/semantic_search")
async def api_semantic_search(q: str = "", k: int = 10, mode: str = "siglip"):
    """
    Legacy alias for /api/search/semantic.

    FIXED: the old implementation imported model_manager.WARM_MODELS into
    THIS process — but model_manager runs as a separate process, so the dict
    was always empty (and the import alone would try to pull 7 GPU models
    into the web server). Embedding now goes through a Redis RPC to the
    model_manager process, which owns the warm models.
    """
    if not q.strip():
        return {"error": "Query is empty", "results": []}
    from explorer_backend import _semantic_search_sync
    data = await asyncio.to_thread(_semantic_search_sync, q, k, mode)
    return {"query": q, "mode": mode,
            "error": data.get("reason") if data.get("status") != "ok" else None,
            "results": data.get("results", [])}


@app.get("/api/graph_search")
async def api_graph_search(q: str = "", limit: int = 10):
    """
    Search the Neo4j knowledge graph for entities matching `q`.
    Returns the entity, its type/description, and linked chunk narratives with timestamps.
    """
    if not q.strip():
        return {"error": "Query is empty", "results": []}

    try:
        from tripartite_db import get_neo4j_driver
        driver = get_neo4j_driver()
        if driver is None:
            return {"error": "Neo4j not available — graph search disabled", "results": []}

        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($q) OR toLower(e.description) CONTAINS toLower($q)
                WITH e LIMIT $limit
                OPTIONAL MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
                OPTIONAL MATCH (c)-[:DESCRIBED_BY]->(n:Narrative)
                RETURN e.name AS entity, e.type AS type, e.description AS description,
                       collect(DISTINCT {chunk_id: c.id, start: c.start, end: c.end, narrative: n.text})[..5] AS chunks
                """,
                q=q, limit=limit,
            )
            rows = result.data()

        driver.close()
        return {"query": q, "results": rows}

    except Exception as e:
        return {"error": str(e), "results": []}


@app.get("/api/spatial_proof")
async def api_spatial_proof(msg_id: int = 0, frame_idx: int = 0, query: str = ""):
    """
    Run Grounding DINO + SAM on a specific frame to produce a spatial mask overlay.
    Returns a JPEG image with the matched region highlighted.

    msg_id    — video message ID (matches the videos table)
    frame_idx — frame number (0-indexed, full-res tier)
    query     — text description of object/text to localise
    """
    if not query.strip():
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "query is required"}, status_code=400)

    import glob as _glob
    from fastapi.responses import StreamingResponse
    import io

    # Locate frame file
    frame_dir = os.path.join(VIDEO_DIR, f"frames_{msg_id}")
    pattern = os.path.join(frame_dir, f"frame_{frame_idx:05d}_ts_*.jpg")
    matches = _glob.glob(pattern)
    if not matches:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": f"Frame {frame_idx} not found for video {msg_id}"}, status_code=404)

    frame_path = matches[0]
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(frame_path).convert("RGB")
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": f"Failed to open frame: {e}"}, status_code=500)

    # Get OCR results for this frame from the DB
    import sqlite3 as _sqlite3
    ocr_results = []
    try:
        conn_s = _sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        row = conn_s.execute(
            "SELECT ocr_text FROM frame_notes WHERE msg_id=? AND frame_idx=?",
            (msg_id, frame_idx)
        ).fetchone()
        conn_s.close()
        # EasyOCR format expected: list of (bbox, text, conf)
        # We convert the stored flat string into a minimal structure
        if row and row[0]:
            ocr_results = [([(0,0),(0,0),(0,0),(0,0)], row[0], 0.9)]
    except Exception:
        pass

    # Run spatial proof via Redis RPC in the model_manager process.
    # (Importing model_manager here would try to load 7 GPU models into the
    # web server — the models live in a separate process.)
    import base64
    rpc = await asyncio.to_thread(
        rpc_call_sync, "spatial_proof",
        {"frame_path": frame_path, "query": query, "ocr_results": ocr_results and ocr_results[0][1] or ""},
        30.0)
    if not rpc.get("ok"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": rpc.get("error", "spatial proof RPC failed "
                             "(is model_manager running?)")}, status_code=503)

    try:
        img_bytes = base64.b64decode(rpc["image_b64"])
    except (KeyError, ValueError):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "corrupt spatial proof result"}, status_code=500)

    headers = {
        "X-Spatial-Proof-Success": str(rpc.get("success", False)),
        "X-Spatial-Proof-Message": str(rpc.get("message", ""))[:200],
    }
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg", headers=headers)


@app.get("/", response_class=HTMLResponse)
async def serve_main_ui():
    with open("main_ui.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    init_sqlite_schema()   # authoritative, idempotent full schema (WAL + FTS)
    init_db()              # legacy init kept for safety (all IF NOT EXISTS)
    
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
