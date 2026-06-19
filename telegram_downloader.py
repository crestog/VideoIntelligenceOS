import os
import sqlite3
import re
import builtins
import time
import traceback
import asyncio
import urllib.request
import json
from pyrogram import Client
from pyrogram.errors import FloodWait
import nest_asyncio

# Config imports
from config import API_ID, API_HASH, BOT_TOKEN, CHANNEL_ID, ARCHIVE_DIR, SESSION_PATH, DB_PATH

nest_asyncio.apply()

def custom_print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(f"[{time.strftime('%H:%M:%S')}]", *args, **kwargs)

# Unlocker: Clean up corrupted journal files from previous crashes
for j_file in [f"{SESSION_PATH}.session-journal", f"{DB_PATH}-journal", f"{DB_PATH}-wal"]:
    if os.path.exists(j_file):
        try: os.remove(j_file)
        except: pass

# --- SHARED GLOBALS ---
GLOBAL_STATUS = "⏳ Cloud Server Booting..."
CATEGORY_QUEUE = []

SCAN_TRIGGER = asyncio.Event()
SCAN_TRIGGER.set()

# ==========================================
# 1. DATABASE LOGIC (Flattened)
# ==========================================
def init_db():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            msg_id BIGINT PRIMARY KEY, creator TEXT, category TEXT,
            likes INTEGER, caption TEXT, local_video_path TEXT, status TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ==========================================
# 2. WORKER LOGIC
# ==========================================
def trigger_manual_scan():
    SCAN_TRIGGER.set()
    return "⚡ Scan triggered! Check status box..."

def extract_text(pattern, text):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else "Unknown"

def extract_num(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1).replace(",", "").strip()) if match else 0

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
        custom_print("✅ Telegram Authorization Confirmed.")
    except Exception:
        GLOBAL_STATUS = "⚙️ Auto-Resolving Telegram Cache..."
        custom_print("⏳ AMNESIA DETECTED: Triggering HTTP REST API self-ping loop...")

        amnesia_event = asyncio.Event()
        resolved_msg_id = [None]

        @app_client.on_message()
        async def handler(client, message):
            if message.chat and message.chat.id == CHANNEL_ID:
                resolved_msg_id[0] = message.id
                amnesia_event.set()

        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            payload = json.dumps({"chat_id": CHANNEL_ID, "text": ".", "disable_notification": True}).encode('utf-8')
            req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req)
            custom_print("📡 Ping sent! Waiting for MTProto to echo the Access Hash...")

            await asyncio.wait_for(amnesia_event.wait(), timeout=10.0)
            custom_print("✅ AUTHORIZATION REBUILT AUTOMATICALLY. Ping left as proof.")

        except Exception as e:
            custom_print(f"❌ CRITICAL ERROR: HTTP Auto-resolve failed! ({e})")
            GLOBAL_STATUS = "❌ Bot access denied. Ensure bot is channel admin."
            await asyncio.sleep(10)
            return

    while True:
        if not SCAN_TRIGGER.is_set():
            GLOBAL_STATUS = "💤 Idling. Click 'Scan Telegram' to fetch new videos."
            while not SCAN_TRIGGER.is_set():
                await asyncio.sleep(1)

        SCAN_TRIGGER.clear()
        GLOBAL_STATUS = "⚡ Running Flash Metadata Scan..."
        custom_print("⚡ Starting Scan...")

        try:
            ping = await app_client.send_message(CHANNEL_ID, ".", disable_notification=True)
            latest_id = ping.id
            await ping.delete()

            conn = sqlite3.connect(DB_PATH, timeout=20)
            cursor = conn.cursor()
            missing_ids = [i for i in range(latest_id, 0, -1) if not cursor.execute("SELECT msg_id FROM videos WHERE msg_id = ?", (i,)).fetchone()]
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

                        # FLATTENED SCHEMA INSERT
                        cursor.execute('''
                            INSERT INTO videos (msg_id, creator, category, likes, caption, status) 
                            VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(msg_id) DO UPDATE SET
                            creator=excluded.creator, category=excluded.category, 
                            likes=excluded.likes, caption=excluded.caption
                        ''', (msg.id, creator_name, cat_name, likes, clean_caption, "Metadata_Only"))
                    conn.commit()
                    conn.close()

                    await asyncio.sleep(2)
                except FloodWait as e: await asyncio.sleep(e.value)
                except Exception: pass

            while True:
                conn = sqlite3.connect(DB_PATH, timeout=20)
                cursor = conn.cursor()
                pending = cursor.execute("SELECT msg_id, category FROM videos WHERE status = 'Metadata_Only' ORDER BY msg_id DESC").fetchall()

                if not pending:
                    conn.close()
                    GLOBAL_STATUS = "✅ Sync Complete! Sleeping until next manual scan."
                    custom_print("✅ All queued videos downloaded. Going to sleep.")
                    break

                target_vid = None
                for queued_cat in CATEGORY_QUEUE:
                    target_vid = next((r[0] for r in pending if r[1] == queued_cat), None)
                    if target_vid: break

                if not target_vid: target_vid = pending[0][0]
                conn.close()

                GLOBAL_STATUS = f"🚀 Background downloading Video #{target_vid}..."
                await asyncio.sleep(1.5)
                try:
                    msg = await app_client.get_messages(CHANNEL_ID, target_vid)
                    target_video = os.path.join(ARCHIVE_DIR, f"video_{target_vid}.mp4")
                    await app_client.download_media(msg, file_name=target_video)
                    
                    # FFmpeg ensuring web safe was removed here per your instructions

                    conn = sqlite3.connect(DB_PATH, timeout=20)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE videos SET local_video_path = ?, status = 'Harvested' WHERE msg_id = ?", (target_video, target_vid))
                    conn.commit()
                    conn.close()
                    custom_print(f"✅ SUCCESS: Video #{target_vid} ready!")
                except FloodWait as e: await asyncio.sleep(e.value)
                except Exception as e:
                    custom_print(f"Download Error for {target_vid}: {e}")
                    conn = sqlite3.connect(DB_PATH, timeout=20)
                    conn.execute("UPDATE videos SET status = 'Error' WHERE msg_id = ?", (target_vid,))
                    conn.commit()
                    conn.close()

        except Exception as e:
            custom_print(f"Worker Error: {e}")
            GLOBAL_STATUS = f"❌ Error: {str(e)[:50]}"
            await asyncio.sleep(5)

if __name__ == "__main__":
    init_db()
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(background_downloader())
    except KeyboardInterrupt:
        custom_print("Process manually stopped by user.")
    except Exception as e:
        custom_print(f"Main loop crashed: {e}")
