import os, sqlite3, aiosqlite, redis, re, time, asyncio, urllib.request, json, uvicorn, threading, subprocess
from collections import deque
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
import logging
import aiofiles
from fastapi.staticfiles import StaticFiles
from pyrogram import Client
from pyrogram.errors import FloodWait
import nest_asyncio
from queue_manager import push_job, get_queue_metrics, get_queue_depth, replay_dlq, wait_for_redis
from v17_backend import v17_router
from admin_backend import admin_router
from config import (BASE_DIR, LAKE_DIR, DB_PATH, VIDEO_DIR, SESSION_DIR, STATE_FILE, THUMB_DIR,
                    OMNI_ENABLED, OMNI_DEDUP_SET, OMNI_DASHBOARD_PORT,
                    QUEUE_OMNI_VISION, QUEUE_OMNI_ORACLE, DISK_DL_PAUSE_GB)

nest_asyncio.apply()

logger = logging.getLogger("VIOS")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S'))
    logger.addHandler(ch)

# ═══════════════════════════════════════════════════════════════════════════
# LOG BUS — why the UI used to stall
# ═══════════════════════════════════════════════════════════════════════════
# This process runs TWO event loops: uvicorn owns one in a daemon thread, and
# the Ghost Worker owns the main thread's loop. WebSocket objects are created
# by uvicorn, so their transports belong to uvicorn's loop.
#
# The old code called, from the Ghost Worker's loop:
#     loop.create_task(ws_mgr.broadcast(...))
# i.e. it awaited `ws.send_json()` on transports owned by the *other* loop, once
# per log line, sequentially across every connected client. Three consequences,
# all of which the user sees as "buffering" and "getting stuck":
#
#   1. Writing to a transport from a foreign loop is undefined behaviour. It is
#      the direct source of the `socket.send() raised exception.` flood, and the
#      bare `except: pass` hid every one of them.
#   2. One WS frame per log line. A retry storm emits thousands of lines a
#      second, so the browser spent its main thread parsing JSON instead of
#      responding to clicks.
#   3. The Ghost Worker awaited that I/O, so network stalls and UI stalls fed
#      each other.
#
# The fix inverts the direction. Producers never touch a socket: they append to
# a bounded ring buffer under a plain threading.Lock (no awaits, no loop
# affinity, O(1)). Each WebSocket connection runs its own drain task ON UVICORN'S
# LOOP, keeps a cursor into the ring, and ships whatever is new as ONE batched
# frame a few times a second. Slow or dead clients can no longer back-pressure
# the harvester, and a flood costs the browser 4 frames/sec instead of 4000.
_LOG_RING_SIZE = 400          # lines retained for late joiners
_LOG_FLUSH_INTERVAL = 0.25    # seconds between batched pushes (4 Hz)


class LogBus:
    """Bounded, loop-agnostic fan-out buffer for console lines."""

    def __init__(self, size=_LOG_RING_SIZE):
        self._lock = threading.Lock()
        self._buf = deque(maxlen=size)
        self._seq = 0            # total lines ever published
        # Repeat suppression. A wedged network emits the same line forever;
        # collapsing it keeps the signal ("this is still happening") without
        # the volume, and without inventing a line the producer never wrote.
        self._last_msg = None
        self._last_at = 0.0
        self._repeats = 0

    def publish(self, msg: str):
        now = time.time()
        with self._lock:
            if msg == self._last_msg and (now - self._last_at) < 30.0:
                self._repeats += 1
                # Emit a running tally on a geometric ladder (2,4,8,...) so a
                # storm is visible but bounded.
                if self._repeats & (self._repeats - 1):
                    return
                msg = f"{msg}   (×{self._repeats + 1})"
            else:
                if self._repeats:
                    self._buf.append((self._seq, f"[{time.strftime('%H:%M:%S')}] "
                                      f"↑ previous line repeated "
                                      f"{self._repeats} more time(s)"))
                    self._seq += 1
                self._last_msg = msg
                self._repeats = 0
            self._last_at = now
            self._buf.append((self._seq, f"[{time.strftime('%H:%M:%S')}] {msg}"))
            self._seq += 1

    @property
    def seq(self):
        with self._lock:
            return self._seq

    def since(self, cursor):
        """(lines, new_cursor) for everything published after `cursor`."""
        with self._lock:
            if cursor >= self._seq:
                return [], self._seq
            # A cursor older than the ring means the client fell behind; it
            # resumes from the oldest line still held rather than replaying
            # history it can no longer get.
            return [text for seq, text in self._buf if seq >= cursor], self._seq

    def tail(self, n=120):
        with self._lock:
            lines = [text for _seq, text in self._buf][-n:]
            return lines, self._seq


log_bus = LogBus()


def custom_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logger.info(msg)
    log_bus.publish(msg)


# ── Library log-spam rate limiter ──────────────────────────────────────────
# pyrogram re-logs every MTProto retry at WARNING. When the session's socket
# dies it retries indefinitely, and the boot console filled with hundreds of
# identical `Retrying "channels.GetMessages" due to: [Errno 32] Broken pipe`
# lines. Silencing the logger outright would hide a real fault, so allow the
# first few and then throttle each distinct message to one line per interval.
class _RateLimitFilter(logging.Filter):
    def __init__(self, burst=3, interval=20.0):
        super().__init__()
        self.burst, self.interval = burst, interval
        self._seen = {}
        self._lock = threading.Lock()

    def filter(self, record):
        key = record.getMessage()[:120]
        now = time.time()
        with self._lock:
            count, first_at = self._seen.get(key, (0, now))
            if now - first_at > self.interval:
                count, first_at = 0, now
            self._seen[key] = (count + 1, first_at)
            if len(self._seen) > 512:        # keep the map from growing forever
                self._seen.clear()
        return count < self.burst


for _noisy in ("pyrogram.session.session", "pyrogram.connection.connection",
               "pyrogram.session.auth", "asyncio"):
    logging.getLogger(_noisy).addFilter(_RateLimitFilter())

# --- CREDENTIALS (single source of truth: config.py, env-only) ---
from config import (API_ID, API_HASH, BOT_TOKEN, CHANNEL_ID,
                    missing_telegram_secrets)

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

# ═══════════════════════════════════════════════════════════════════════════
# READ CACHE — the second half of the "buffering" fix
# ═══════════════════════════════════════════════════════════════════════════
# No browser in this repo connects to /ws; every UI polls over HTTP instead.
# main_ui alone fires /api/status every 1.5s and /api/playlist + /api/categories
# every 3s, and each open tab multiplies that. Those three queries were hitting
# SQLite and Redis on every single tick, so a click had to wait behind whatever
# poll was already holding the read lock.
#
# These payloads are all "recent state" — a second of staleness is invisible to
# the user but removes almost all of the duplicated work. Writers bump
# `_feed_epoch` so a finished download shows up immediately instead of waiting
# out the TTL.
_cache_lock = threading.Lock()
_cache: dict = {}
_feed_epoch = 0


def bump_feed_epoch():
    """Invalidate every feed-derived cache entry (call after a DB write)."""
    global _feed_epoch
    with _cache_lock:
        _feed_epoch += 1


def cache_get(key, ttl):
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > time.time() and hit[1] == _feed_epoch:
            return hit[2]
    return None


def cache_put(key, value, ttl):
    with _cache_lock:
        _cache[key] = (time.time() + ttl, _feed_epoch, value)
        if len(_cache) > 256:
            for k in [k for k, v in _cache.items() if v[0] < time.time()]:
                _cache.pop(k, None)
    return value


# How many entries the player feed carries. `get_playlist_data` had no LIMIT at
# all: every 3s poll serialized every row in the table, captions included, and
# the response grew without bound as the archive filled. The player only ever
# renders a window of this, and new posts sort to the front.
PLAYLIST_LIMIT = 400


def init_db():
    """Create lake.db's schema if it is missing.

    The DDL itself lives in lake_schema so the factory reset can re-apply it:
    a reset deletes lake.db from inside this process, and this process is
    deliberately not restarted, so the reset — not the next boot — has to be
    the thing that puts the tables back.
    """
    from lake_schema import ensure_lake_schema
    res = ensure_lake_schema(DB_PATH)
    for note in res["skipped"]:
        logger.warning(f"lake schema — {note}")

async def get_playlist_data(category):
    cache_key = f"playlist:{category}"
    cached = cache_get(cache_key, 3.0)
    if cached is not None:
        return cached
    async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
        base_query = '''SELECT cr.username, c.name, p.likes, p.caption, p.local_video_path, p.video_id FROM posts p JOIN creators cr ON p.creator_id = cr.id JOIN categories c ON p.category_id = c.id WHERE p.local_video_path IS NOT NULL'''
        if category and category != "All Categories":
            base_query += " AND c.name = ? ORDER BY p.video_id DESC LIMIT ?"
            async with conn.execute(base_query, (category, PLAYLIST_LIMIT)) as cursor:
                rows = await cursor.fetchall()
        else:
            base_query += " ORDER BY p.video_id DESC LIMIT ?"
            async with conn.execute(base_query, (PLAYLIST_LIMIT,)) as cursor:
                rows = await cursor.fetchall()

    playlist = []
    for row in rows:
        filename = os.path.basename(row[4]) if row[4] else ""
        playlist.append({'username': row[0], 'name': row[1], 'likes': row[2], 'caption': row[3], 'filename': filename, 'id': row[5]})
    return cache_put(cache_key, playlist, 3.0)

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

def harvest_heartbeat(state, detail=""):
    """Tell the admin panel what the harvester is really doing.

    Synchronous and best-effort. It runs on the Ghost Worker's event loop, so
    it must not block: system_control's Redis handle has a short socket timeout
    and every failure path there returns rather than raising.
    """
    try:
        from system_control import heartbeat
        heartbeat("harvest", state, detail)
    except Exception:
        pass


async def _harvest_paused():
    """True if the harvester should hold. Sleeps a beat before saying so.

    Returning True means "the caller should `continue`" — the sleep is inside
    here so a paused loop does not spin, and so the pause is reported on every
    pass rather than once when it started.
    """
    global GLOBAL_STATUS
    try:
        from system_control import is_paused
    except Exception:
        return False
    if not is_paused("harvest"):
        return False
    GLOBAL_STATUS = "⏸️ Paused by Admin"
    harvest_heartbeat("paused")
    await asyncio.sleep(2)
    return True


async def background_downloader():
    global GLOBAL_STATUS, CATEGORY_QUEUE
    custom_print("\n=======================================")
    custom_print("🚀 GHOST WORKER: SYSTEMS ONLINE")
    custom_print("=======================================\n")

    # Without credentials pyrogram fails several frames deep with an opaque
    # auth error, and the watchdog then restarts this process forever. The web
    # UI, the API and the frame pipeline do not need Telegram, so park the
    # harvester instead of taking the server down with it.
    absent = missing_telegram_secrets()
    if absent:
        GLOBAL_STATUS = f"⚠️ Telegram disabled — missing {', '.join(absent)}"
        custom_print(f"⚠️ Ghost Worker idle: {', '.join(absent)} not set.")
        custom_print("   Add them as Kaggle Secrets and restart the notebook to "
                     "resume channel harvesting. The UI and CV pipeline are unaffected.")
        while True:
            await asyncio.sleep(3600)

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
            custom_print(f"❌ Cannot reach channel {CHANNEL_ID}: {str(e)[:120]}")
            if "peer id invalid" in str(e).lower():
                custom_print("   ↳ Telegram has not cached this channel for the bot. Fix: add the "
                             "bot to the channel AS AN ADMIN, then post any message there once. "
                             "A bot cannot resolve a channel it has never received an update from.")
            await asyncio.sleep(10)
            return

    import shutil
    while True:
        # Admin pause. The harvester used to have no pause at all: the panel
        # could stop the CV engine and the Omniscient workers, but downloads
        # kept running and kept filling the disk that the pause was usually
        # pressed to protect. Checked here and again before each download, so a
        # pause lands within seconds rather than at the end of a scan.
        if await _harvest_paused():
            continue

        # Guard the scratch tier (where videos/frames land), using the shared
        # threshold rather than a local literal — frame_worker pauses at the
        # same boundary, so the two stay in step.
        free_gb = shutil.disk_usage(VIDEO_DIR).free / (1024**3)
        if free_gb < DISK_DL_PAUSE_GB:
            GLOBAL_STATUS = f"⚠️ Storage Critical ({free_gb:.1f}GB left) - Pausing"
            harvest_heartbeat("idle", f"disk critical — {free_gb:.1f} GB free")
            await asyncio.sleep(60)
            continue

        if not SCAN_TRIGGER.is_set():
            GLOBAL_STATUS = "💤 Idling. Background queue is empty."
            harvest_heartbeat("idle", "queue empty")
            while not SCAN_TRIGGER.is_set(): await asyncio.sleep(1)

        SCAN_TRIGGER.clear()

        sync_category_queue(force=True)

        GLOBAL_STATUS = "⚡ Running Flash Metadata Scan across Telegram..."
        custom_print("⚡ Scanning Telegram for new Posts...")

        try:
            # Discover the newest message id by posting a throwaway message and
            # reading its id. This looks roundabout, but a bot account CANNOT
            # read chat history — messages.getHistory is a user-only MTProto
            # method and returns BOT_METHOD_INVALID, so get_chat_history() is
            # not an option here. (messages.getMessages, used by the batch loop
            # below, *is* allowed for bots — that asymmetry is the whole reason
            # for the ping.) The id it leaves behind is recorded in scanned_ids
            # immediately, so the gap never shows up as "missing" again.
            try:
                ping = await app_client.send_message(CHANNEL_ID, ".", disable_notification=True)
                latest_id = ping.id
                await ping.delete()
            except FloodWait as e:
                custom_print(f"⚠️ FloodWait probing newest id: sleeping {e.value}s")
                SCAN_TRIGGER.set()
                await asyncio.sleep(e.value)
                continue
            except Exception as e:
                custom_print(f"❌ Cannot read newest message id: {str(e)[:120]}")
                custom_print("   ↳ The bot must be an ADMIN of the channel with "
                             "permission to post. Retrying in 30s.")
                SCAN_TRIGGER.set()
                await asyncio.sleep(30)
                continue

            async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
                # Close out the ping's own id before computing the gap.
                await conn.execute('INSERT OR IGNORE INTO scanned_ids (video_id) VALUES (?)',
                                   (latest_id,))
                await conn.commit()
                # One set-difference instead of one SELECT per id. The old loop
                # ran `latest_id` queries per scan (tens of thousands on a real
                # channel) just to build this list. Full range is still covered
                # — the whole channel is scanned, we just ask the DB once.
                async with conn.execute("SELECT video_id FROM posts") as cursor:
                    known = {row[0] for row in await cursor.fetchall()}
                async with conn.execute("SELECT video_id FROM scanned_ids") as cursor:
                    known.update(row[0] for row in await cursor.fetchall())
                missing_ids = [i for i in range(latest_id, 0, -1) if i not in known]

            if missing_ids:
                custom_print(f"🔍 {len(missing_ids)} of {latest_id} message ids not yet scanned "
                             f"({len(known)} already covered)")
            else:
                custom_print(f"✅ Channel fully scanned — all {latest_id} ids covered.")

            BATCH_SIZE = 200
            total_batches = (len(missing_ids) + BATCH_SIZE - 1) // BATCH_SIZE
            found_videos = 0
            for i in range(0, len(missing_ids), BATCH_SIZE):
                chunk = missing_ids[i:i + BATCH_SIZE]
                batch_no = i // BATCH_SIZE + 1
                GLOBAL_STATUS = (f"⚡ Scanning batch {batch_no}/{total_batches} "
                                 f"({found_videos} videos found)")
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
                            found_videos += 1
                        # Mark the whole chunk covered, including non-video ids,
                        # so the next scan starts where this one finished.
                        await conn.executemany('INSERT OR IGNORE INTO scanned_ids (video_id) VALUES (?)',
                                               [(cid,) for cid in chunk])
                        await conn.commit()
                    # A long scan should not look like a hang. Report every 10th
                    # batch (~2000 ids) so there is always visible motion.
                    if batch_no % 10 == 0 or batch_no == total_batches:
                        custom_print(f"   🔍 Scanned {batch_no}/{total_batches} batches "
                                     f"— {found_videos} videos found so far")
                    await asyncio.sleep(0.5)
                except FloodWait as e:
                    custom_print(f"⚠️ FloodWait during scan: sleeping {e.value}s")
                    await asyncio.sleep(e.value)
                except Exception as e:
                    # 50 chars truncated the useful half of most Telegram
                    # errors (the method name and reason live at the end).
                    custom_print(f"⚠️ Scan error on batch {batch_no}/{total_batches}: "
                                 f"{type(e).__name__}: {str(e)[:160]}")
                    await asyncio.sleep(5)

            while True:
                # The download loop can run for hours without returning to the
                # outer loop, so the pause is checked here too — otherwise
                # pressing Pause during a long harvest would not take effect
                # until the whole pending list had been downloaded.
                if await _harvest_paused():
                    continue

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
                    # A new playable video must appear in the feed at once, not
                    # after the read cache expires.
                    bump_feed_epoch()

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

                    # Omniscient pipeline: harvested videos ride the DEFAULT
                    # lane (bot uploads own the PRIORITY lane in the engine)
                    if OMNI_ENABLED:
                        try:
                            omni_uuid = f"tg{target_vid}"
                            if not REDIS_CACHE or REDIS_CACHE.sadd(OMNI_DEDUP_SET, omni_uuid):
                                omni_job = {"uuid": omni_uuid, "path": target_video,
                                            "mode": "blitz", "source": "harvest"}
                                push_job(QUEUE_OMNI_VISION, omni_job, is_priority=False)
                                push_job(QUEUE_OMNI_ORACLE, omni_job, is_priority=False)
                                custom_print(f"🔮 Queued #{target_vid} for Omniscient Engine (⚪ DEFAULT)")
                        except Exception as e:
                            custom_print(f"⚠️ Omni queue push failed for #{target_vid}: {str(e)[:60]}")

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
            # This used to swallow the error in silence: it set GLOBAL_STATUS,
            # slept, and looped — and since SCAN_TRIGGER had already been
            # cleared, the worker then parked in the idle branch forever. The
            # console showed "Scanning Telegram for new Posts..." and then
            # nothing at all, with no hint that anything had failed. Always say
            # what broke, and always re-arm the trigger so the cycle retries.
            GLOBAL_STATUS = f"❌ Worker Error: {str(e)[:50]}"
            custom_print(f"❌ Ghost Worker cycle failed: {type(e).__name__}: {str(e)[:160]}")
            custom_print("   ↳ Retrying in 15s.")
            SCAN_TRIGGER.set()
            await asyncio.sleep(15)

# --- FASTAPI APP ---
app = FastAPI(title="Insta-Vault Modular OS")


# When the disk fills, SQLite raises "database or disk is full" from deep inside
# a route and Starlette re-raises it through the ASGI stack — the client gets a
# dropped connection and the console gets an unreadable traceback wall, which is
# exactly what buried the real cause (no space) in the last failed run. These
# handlers turn that into one actionable line plus a normal JSON error.
@app.exception_handler(sqlite3.Error)
async def _sqlite_error_handler(request: Request, exc: sqlite3.Error):
    detail = str(exc)
    if "disk is full" in detail or "database or disk is full" in detail:
        custom_print(f"❌ DISK FULL while serving {request.url.path} — "
                     f"free space on the scratch tier and restart.")
        msg = ("Disk is full. Frames/models live on the scratch tier; free space "
               "there or point VIOS_SCRATCH_DIR at a larger volume.")
    else:
        custom_print(f"❌ DB error on {request.url.path}: {detail[:160]}")
        msg = f"Database error: {detail[:200]}"
    return JSONResponse(status_code=503, content={"ok": False, "error": msg})


@app.exception_handler(OSError)
async def _os_error_handler(request: Request, exc: OSError):
    # errno 28 == ENOSPC
    if getattr(exc, "errno", None) == 28:
        custom_print(f"❌ NO SPACE LEFT while serving {request.url.path}")
        return JSONResponse(status_code=503, content={
            "ok": False,
            "error": "No space left on device. Free the scratch tier and retry."})
    custom_print(f"❌ I/O error on {request.url.path}: {str(exc)[:160]}")
    return JSONResponse(status_code=500,
                        content={"ok": False, "error": str(exc)[:200]})

# StaticFiles raises at construction if the directory is missing, which crashes
# the whole process before uvicorn ever starts — and the watchdog then reboots
# it forever. config creates these, but a mount must never be the thing that
# takes the UI down, so make sure they exist here too.
for _static_dir in (VIDEO_DIR, THUMB_DIR):
    os.makedirs(_static_dir, exist_ok=True)

app.mount("/data", StaticFiles(directory=VIDEO_DIR), name="v17_data")
# THUMB_DIR, not LAKE_DIR/.thumbnails: thumbnails are regenerable, so they live
# on the scratch tier now. This was the last hardcoded copy of the legacy path.
app.mount("/thumbs", StaticFiles(directory=THUMB_DIR), name="v17_thumbs")
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="main_videos")

app.include_router(v17_router)
app.include_router(admin_router)

# The v2 capture plane. Imported defensively because it is the newest and most
# separable part of the system: a missing dependency in the capture tab must
# not take the feed, the explore tab and v17 down with it.
#
# The handler uses `custom_print`, and that is the whole point of this comment.
# It used to call `vios_log`, which this module never imports — so the guard
# raised NameError *while handling* the import failure. An except clause that
# cannot run is worse than no guard at all: one missing wheel
# (python-multipart, yt-dlp) turned "the capture tab is unavailable" into
# ui_server exiting at import, which boot.py's watchdog then restarted forever.
# That is the loop that reads as "the backend does not start".
try:
    from vios.capture.routes import capture_router
    app.include_router(capture_router)
    _CAPTURE_READY = True
except Exception as _capture_exc:      # pragma: no cover - import guard
    _CAPTURE_READY = False
    _CAPTURE_WHY = f"{type(_capture_exc).__name__}: {_capture_exc}"
    custom_print(f"⚠️ capture tab unavailable — {_CAPTURE_WHY}")

# The v2 processing plane. Guarded separately from capture: the two share a
# design and nothing else, and a torch that will not import on this box must
# not cost the operator the capture tab as well.
try:
    from vios.process.routes import process_router
    app.include_router(process_router)
    _PROCESS_READY = True
except Exception as _process_exc:      # pragma: no cover - import guard
    _PROCESS_READY = False
    _PROCESS_WHY = f"{type(_process_exc).__name__}: {_process_exc}"
    custom_print(f"⚠️ process tab unavailable — {_PROCESS_WHY}")


# A tab that failed to import must still answer, or the page it backs shows a
# spinner forever with nothing in the log to explain it. These stand-ins reply
# with the import error itself, so the browser says what the console said.
def _plane_unavailable(name: str, why: str):
    from fastapi.responses import PlainTextResponse

    @app.get(f"/{name}", response_class=PlainTextResponse)
    def _page():                                   # pragma: no cover
        return (f"The {name} tab could not load on this machine.\n\n{why}\n\n"
                f"Everything else on this server is running. Reinstall the "
                f"dependencies (bash setup.sh) and restart to get it back.")

    @app.get(f"/api/{name}/status")
    def _status():                                 # pragma: no cover
        return {"ok": False, "available": False, "error": why}


if not _CAPTURE_READY:
    _plane_unavailable("capture", _CAPTURE_WHY)
if not _PROCESS_READY:
    _plane_unavailable("process", _PROCESS_WHY)


# ── Atlas, mounted rather than launched ───────────────────────────────────
# Atlas was built as its own FastAPI app on its own port with its own tunnel
# (atlas_boot.py), which is why the reader was a separate website you had to
# find a second URL for. It is a sub-application, so it can simply be mounted:
# every Atlas route keeps its own path, one level down, on the URL that is
# already open.
#
# `atlas_boot.py` still works and is unchanged — Atlas on a laptop with the
# channel and no GPU is a real way to run it. Both entry points now share one
# app object, so there is no second copy of the reader to keep in step.
#
# Boot is *not* started here. Atlas's boot thread scans the Telegram channel,
# imports bundles and loads a sentence-transformer; doing that at import time
# would add minutes to every ui_server start, including the starts where nobody
# opens Atlas at all. It is started on the first request to /atlas instead, and
# the interface is designed to be served during its own boot — it renders
# immediately and reports progress in the status pill.
_ATLAS_READY = False
_ATLAS_WHY = ""
try:
    from atlas import server as _atlas_server

    app.mount("/atlas", _atlas_server.app, name="atlas")
    _ATLAS_READY = True
except Exception as _atlas_exc:         # pragma: no cover - import guard
    _ATLAS_WHY = f"{type(_atlas_exc).__name__}: {_atlas_exc}"
    custom_print(f"⚠️ atlas tab unavailable — {_ATLAS_WHY}")

if _ATLAS_READY:
    _atlas_booted = threading.Event()

    @app.middleware("http")
    async def _atlas_lazy_boot(request: Request, call_next):
        """Start Atlas's scan/index the first time somebody opens it."""
        if (request.url.path.startswith("/atlas")
                and not _atlas_booted.is_set()):
            _atlas_booted.set()
            try:
                _atlas_server.start_boot()
                custom_print("🗺️ Atlas: boot started (scanning the channel for "
                             "database bundles)")
            except Exception as exc:
                custom_print(f"⚠️ Atlas boot failed to start: "
                             f"{type(exc).__name__}: {exc}")
        return await call_next(request)
else:
    @app.get("/atlas", response_class=HTMLResponse)
    def _atlas_missing():                          # pragma: no cover
        return HTMLResponse(
            f"<h1>Atlas</h1><p>Atlas could not load on this machine.</p>"
            f"<pre>{_ATLAS_WHY}</pre>", status_code=503)

@app.get("/api/status")
def get_status():
    """Polled every 1.5s per open tab, so it must be nearly free.

    `get_queue_metrics` is a multi-key blocking Redis round-trip and
    `disk_usage` is a syscall against a network-backed mount; neither result
    changes meaningfully inside one second. GLOBAL_STATUS is read live so the
    status line still updates on every tick.
    """
    cached = cache_get("status:payload", 1.0)
    if cached is None:
        try:
            metrics = get_queue_metrics("QUEUE_VISION")
        except Exception:
            metrics = {}
        try:
            import shutil
            disk = shutil.disk_usage(LAKE_DIR)
            free_gb = f"{disk.free / (1024**3):.1f} GB"
        except Exception:
            free_gb = "N/A"
        cached = cache_put("status:payload", {"queue": metrics, "disk_free": free_gb}, 1.0)
    return {"status": GLOBAL_STATUS, **cached}

# ── OMNISCIENT DASHBOARD PROXY ──
# The God-Mode Explorer (Flask) lives inside omni_engine.py on localhost so it
# can share the embedded Qdrant store with the workers. We reverse-proxy it at
# /omni so the workstation's "Omniscient" tab (and cloudflared) can reach it.
_OMNI_DASH_URL = f"http://127.0.0.1:{OMNI_DASHBOARD_PORT}"
_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


@app.get("/omni")
async def omni_root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/omni/")   # trailing slash → relative URLs resolve


@app.api_route("/omni/{path:path}", methods=["GET", "POST"])
async def omni_proxy(request: Request, path: str = ""):
    try:
        import httpx
    except ImportError:
        return HTMLResponse("<h3>httpx not installed — run setup.sh</h3>", status_code=500)
    try:
        client = httpx.AsyncClient(base_url=_OMNI_DASH_URL,
                                   timeout=httpx.Timeout(10.0, read=None))
        fwd_headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in _HOP_HEADERS}
        req = client.build_request(request.method, f"/{path}",
                                   params=request.query_params,
                                   headers=fwd_headers,
                                   content=await request.body())
        resp = await client.send(req, stream=True)
        resp_headers = {k: v for k, v in resp.headers.items()
                        if k.lower() not in _HOP_HEADERS}

        async def _stream():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(_stream(), status_code=resp.status_code,
                                 headers=resp_headers,
                                 media_type=resp.headers.get("content-type"))
    except Exception as e:
        return HTMLResponse(
            "<html><body style='background:#0d1117;color:#c9d1d9;font-family:sans-serif;"
            "padding:50px;text-align:center;'><h2>🔮 Omniscient Engine is still booting…</h2>"
            f"<p>The dashboard will appear once models finish loading.</p>"
            f"<p style='color:#8b949e;font-size:0.8em'>{type(e).__name__}</p>"
            "</body></html>", status_code=503)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """One drain task per client, running on uvicorn's own loop.

    Everything this socket sends originates here, so no other thread ever
    touches the transport. Logs go out as batched arrays on a 4 Hz tick;
    status and queue metrics ride the same tick instead of each holding
    their own timer. `get_queue_metrics` is a blocking Redis call, so it is
    pushed to a worker thread — on the old code a Redis hiccup froze the
    socket, and with it the UI's status bar.
    """
    await ws.accept()

    # Late joiners get the recent tail immediately rather than an empty console.
    backlog, cursor = log_bus.tail(120)
    try:
        await ws.send_json({"type": "log_batch", "lines": backlog})
    except Exception:
        return

    async def pump():
        nonlocal cursor
        last_status = None
        tick = 0
        while True:
            try:
                lines, cursor = log_bus.since(cursor)
                if lines:
                    # Hard cap per frame: a burst larger than this is already
                    # unreadable, and shipping it whole is what locked up the
                    # browser's main thread.
                    if len(lines) > 200:
                        dropped = len(lines) - 200
                        lines = ([f"… {dropped} lines dropped (flood) …"]
                                 + lines[-200:])
                    await ws.send_json({"type": "log_batch", "lines": lines})

                gs = globals().get("GLOBAL_STATUS", "")
                if gs != last_status:
                    last_status = gs
                    await ws.send_json({"type": "status", "message": gs})

                tick += 1
                if tick % 20 == 0:      # every ~5s
                    try:
                        m = await asyncio.to_thread(get_queue_metrics, "QUEUE_VISION")
                        await ws.send_json({"type": "queue", "data": m})
                    except Exception:
                        pass            # Redis down must not kill the socket
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception:
                break
            await asyncio.sleep(_LOG_FLUSH_INTERVAL)

    t = asyncio.create_task(pump())
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        # Always cancel the pump. The old code only did this on a clean
        # WebSocketDisconnect, so any other error left an orphan task looping
        # against a dead socket for the life of the process — one more per
        # browser refresh. That accumulation is the other half of the slowdown.
        t.cancel()

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
    # DISTINCT over a join with no GROUP BY forced a scan of `posts` per call,
    # and main_ui calls this on every feed sync (3s) plus after every switch.
    # The category set changes only when a new category is harvested.
    cats = cache_get("categories:list", 15.0)
    if cats is None:
        async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
            async with conn.execute("SELECT DISTINCT c.name FROM categories c JOIN posts p ON c.id = p.category_id") as cursor:
                cats = sorted([row[0] for row in await cursor.fetchall()])
        cache_put("categories:list", cats, 15.0)
    return {"categories": ["All Categories"] + cats, "queue": CATEGORY_QUEUE}
@app.get("/api/playlist")
async def api_playlist(category: str = "All Categories"): return {"playlist": await get_playlist_data(category)}
@app.get("/api/explore")
async def api_explore(q: str = "", offset: int = 0):
    """Search every piece of metadata a post carries, ids included.

    The old implementation ran `posts_search MATCH ?` against an FTS5 table
    declared with `video_id UNINDEXED` — stored but not searchable — so typing
    an id could never return that video. Worse, `f"{q}*"` is spliced straight
    into FTS5 query syntax: a quote, hyphen, colon or parenthesis raises
    sqlite3.OperationalError, which the global handler turns into a 503 and the
    UI renders as an empty table. That is the "search is not working" report.

    LIKE over the joined columns is the honest fix here. It matches ids,
    creators, categories, captions, filenames and status, it treats the query
    as a literal, and at this table size it is well under a millisecond. Digit
    queries also match the id numerically so "1236" finds tg1236.
    """
    q = (q or "").strip()
    cache_key = f"explore:{q}:{offset}"
    cached = cache_get(cache_key, 3.0)
    if cached is not None:
        return cached

    base = '''SELECT p.video_id, cr.username, c.name, p.likes, p.caption,
                     p.local_video_path, COALESCE(p.status, '')
              FROM posts p
              JOIN creators cr   ON p.creator_id = cr.id
              JOIN categories c  ON p.category_id = c.id
              WHERE p.local_video_path IS NOT NULL'''
    params: list = []
    if q:
        # Escape LIKE wildcards so a literal % or _ in the query means itself.
        esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{esc}%"
        cols = ["CAST(p.video_id AS TEXT)", "cr.username", "c.name",
                "p.caption", "p.local_video_path", "COALESCE(p.status,'')",
                "CAST(p.likes AS TEXT)"]
        base += " AND (" + " OR ".join(
            f"{c} LIKE ? ESCAPE '\\'" for c in cols) + ")"
        params += [like] * len(cols)
        # Exact-id hits sort first, then newest.
        base += " ORDER BY (CAST(p.video_id AS TEXT) = ?) DESC, p.video_id DESC"
        params.append(q)
    else:
        base += " ORDER BY p.video_id DESC"
    base += " LIMIT 30 OFFSET ?"
    params.append(offset)

    async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
        async with conn.execute(base, params) as cursor:
            rows = await cursor.fetchall()

    payload = {"query": q, "count": len(rows), "data": [{
        "id": r[0],
        "creator": r[1],
        "category": r[2],
        "likes": f"{r[3]:,}" if isinstance(r[3], int) else (r[3] or "0"),
        "caption": r[4] or "",
        "status": r[6],
        "uuid": f"tg{r[0]}",
        "filename": os.path.basename(r[5]) if r[5] else "",
    } for r in rows]}
    return cache_put(cache_key, payload, 3.0)
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

@app.get("/", response_class=HTMLResponse)
async def serve_main_ui():
    with open("main_ui.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    init_db()

    # Ghost Worker pushes into Redis as soon as a download finishes — make sure
    # the broker is up first so the very first push doesn't ECONNREFUSE.
    wait_for_redis(label="UI")

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
