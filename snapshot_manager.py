"""
VIOS Snapshot Manager — Database Export/Import via Telegram

The database (lake.db + thumbnails) is the ONLY thing that must survive a
Kaggle session reset — videos live permanently in the channel and are keyed
by message ID, so as long as the DB rows survive, nothing is ever reprocessed.

EXPORT cycle:
  1. `VACUUM INTO` → transactionally-consistent copy of lake.db
     (safe while WAL writers are active; no downtime, no locks held)
  2. Bundle copy + .thumbnails/ + manifest.json into one tar
  3. Compress with zstd (fallback: gzip) — lake.db shrinks ~5-10x
  4. Split into ≤1900 MB parts (bots can upload 2 GB per file via MTProto)
  5. Upload each part to the SAME channel as the videos, captioned
     `#VIOS_SNAPSHOT <utc-ts> part <i>/<n>`
  6. Upload a tiny JSON "pointer" message listing the part message-IDs,
     captioned `#VIOS_SNAPSHOT_MANIFEST <utc-ts>`, and pin it.

IMPORT cycle (fresh Kaggle session):
  1. Read the pinned message (or scan recent messages) for the newest
     `#VIOS_SNAPSHOT_MANIFEST`
  2. Download all listed parts by message ID, concatenate, decompress, untar
  3. Restore lake.db + thumbnails into the DataLake
  4. Rebuild the Redis PROCESSED_VIDEOS_SET from the restored `videos` table
     → the Ghost Worker instantly knows what is already done; zero reprocessing.

Adding new models later: process → write new rows/tables → export again.
Snapshots are append-only in the channel, so every prior state is recoverable.
"""

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import time

from config import (DB_PATH, LAKE_DIR, THUMB_DIR, SNAPSHOT_DIR, SNAPSHOT_CHUNK_MB,
                    SNAPSHOT_TAG, API_ID, API_HASH, BOT_TOKEN, CHANNEL_ID, SQLITE_TIMEOUT)
from logger import vios_log

MANIFEST_TAG = SNAPSHOT_TAG + "_MANIFEST"
CHUNK_BYTES = SNAPSHOT_CHUNK_MB * 1024 * 1024
# Separate session from the Ghost Worker's — two Pyrogram clients must not
# share one session file.
SNAP_SESSION = os.path.join(LAKE_DIR, 'snap_session')


def log(msg, level="INFO"):
    vios_log(msg, "SNAP", level)


# ═══════════════════════════════════════════════════════════
# COMPRESSION — zstd preferred, gzip fallback
# ═══════════════════════════════════════════════════════════
try:
    import zstandard as _zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


def _compress_file(src, dst):
    if _HAS_ZSTD:
        cctx = _zstd.ZstdCompressor(level=9, threads=-1)
        with open(src, 'rb') as fi, open(dst, 'wb') as fo:
            cctx.copy_stream(fi, fo)
    else:
        import gzip
        with open(src, 'rb') as fi, gzip.open(dst, 'wb', compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo)


def _decompress_file(src, dst, codec):
    if codec == 'zstd':
        if not _HAS_ZSTD:
            raise RuntimeError("Snapshot is zstd-compressed but zstandard is not installed")
        dctx = _zstd.ZstdDecompressor()
        with open(src, 'rb') as fi, open(dst, 'wb') as fo:
            dctx.copy_stream(fi, fo)
    else:
        import gzip
        with gzip.open(src, 'rb') as fi, open(dst, 'wb') as fo:
            shutil.copyfileobj(fi, fo)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════
# BUILD — produce the compressed snapshot parts on disk
# ═══════════════════════════════════════════════════════════
def build_snapshot():
    """Create snapshot parts under SNAPSHOT_DIR. Returns (parts, manifest_dict)."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    workdir = os.path.join(SNAPSHOT_DIR, f'snap_{stamp}')
    os.makedirs(workdir, exist_ok=True)

    # 1. Consistent DB copy — VACUUM INTO is transactional and WAL-safe
    db_copy = os.path.join(workdir, 'lake.db')
    log("Creating consistent DB copy (VACUUM INTO)...")
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    try:
        conn.execute("VACUUM INTO ?", (db_copy,))
    finally:
        conn.close()

    # 2. Tar: db + thumbnails + manifest
    tar_path = os.path.join(workdir, 'snapshot.tar')
    log("Bundling DB + thumbnails...")
    with tarfile.open(tar_path, 'w') as tar:
        tar.add(db_copy, arcname='lake.db')
        if os.path.isdir(THUMB_DIR):
            tar.add(THUMB_DIR, arcname='.thumbnails')
    os.remove(db_copy)

    # 3. Compress
    codec = 'zstd' if _HAS_ZSTD else 'gzip'
    comp_path = tar_path + ('.zst' if _HAS_ZSTD else '.gz')
    log(f"Compressing ({codec})...")
    _compress_file(tar_path, comp_path)
    os.remove(tar_path)
    comp_size = os.path.getsize(comp_path)

    # 4. Split into parts
    parts = []
    with open(comp_path, 'rb') as f:
        idx = 0
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            idx += 1
            part_path = os.path.join(workdir, f'snapshot.part{idx:03d}')
            with open(part_path, 'wb') as pf:
                pf.write(chunk)
            parts.append(part_path)
    os.remove(comp_path)

    manifest = {
        "tag": SNAPSHOT_TAG,
        "created_utc": stamp,
        "codec": codec,
        "total_bytes": comp_size,
        "parts": len(parts),
        "part_sha256": [_sha256(p) for p in parts],
        "schema_note": "tar contains lake.db + .thumbnails/",
    }
    with open(os.path.join(workdir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    log(f"Snapshot built: {len(parts)} part(s), {comp_size / (1024*1024):.1f} MB compressed", "SUCCESS")
    return parts, manifest, workdir


# ═══════════════════════════════════════════════════════════
# EXPORT — upload snapshot to the Telegram channel
# ═══════════════════════════════════════════════════════════
async def export_snapshot(progress_cb=None):
    """Build + upload a snapshot. Returns the manifest message ID."""
    from pyrogram import Client

    parts, manifest, workdir = build_snapshot()
    app = Client(SNAP_SESSION, api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)
    await app.start()
    try:
        if not await _ensure_peer(app):
            raise RuntimeError("Channel peer unresolvable — cannot upload snapshot")
        part_msg_ids = []
        for i, part in enumerate(parts, 1):
            cap = f"{SNAPSHOT_TAG} {manifest['created_utc']} part {i}/{len(parts)}"
            log(f"Uploading part {i}/{len(parts)} ({os.path.getsize(part) / (1024*1024):.0f} MB)...")
            msg = await app.send_document(
                CHANNEL_ID, part, caption=cap, force_document=True,
                progress=progress_cb,
            )
            part_msg_ids.append(msg.id)

        manifest["part_msg_ids"] = part_msg_ids
        pointer_path = os.path.join(workdir, 'manifest.json')
        with open(pointer_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        cap = f"{MANIFEST_TAG} {manifest['created_utc']}"
        pointer_msg = await app.send_document(CHANNEL_ID, pointer_path,
                                              caption=cap, force_document=True)
        # Pin so import can find the newest snapshot in O(1)
        try:
            await app.pin_chat_message(CHANNEL_ID, pointer_msg.id, disable_notification=True)
        except Exception as e:
            log(f"Pin failed (non-fatal): {e}", "WARN")

        log(f"Snapshot exported — manifest msg #{pointer_msg.id}", "SUCCESS")
        return pointer_msg.id
    finally:
        await app.stop()
        shutil.rmtree(workdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# IMPORT — restore the newest snapshot from the channel
# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
# PEER RESOLUTION — fresh sessions raise "Peer id invalid"
# ═══════════════════════════════════════════════════════════
async def _ensure_peer(app):
    """
    Make sure the channel peer is in this session's cache. A brand-new
    Pyrogram session (fresh Kaggle boot) knows nothing about CHANNEL_ID, so
    the first get_chat/send_message raises "Peer id invalid: -100...".
    Fix: post via the HTTP Bot API — Telegram then pushes the update to this
    MTProto session, which populates the peer cache. Same trick the Ghost
    Worker uses. Returns True when the peer is usable.
    """
    try:
        await app.resolve_peer(CHANNEL_ID)
        return True
    except Exception:
        pass

    import urllib.request
    got_update = asyncio.Event()

    async def _on_msg(client, message):
        if message.chat and message.chat.id == CHANNEL_ID:
            got_update.set()

    from pyrogram.handlers import MessageHandler
    handler = app.add_handler(MessageHandler(_on_msg))
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = json.dumps({"chat_id": CHANNEL_ID, "text": ".",
                              "disable_notification": True}).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={'Content-Type': 'application/json'})
        resp = json.load(urllib.request.urlopen(req, timeout=10))
        try:
            await asyncio.wait_for(got_update.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        # Clean up the nudge message via HTTP API (works even if MTProto still can't)
        if resp.get("ok"):
            del_url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage"
            del_payload = json.dumps({"chat_id": CHANNEL_ID,
                                      "message_id": resp["result"]["message_id"]}).encode()
            del_req = urllib.request.Request(del_url, data=del_payload,
                                             headers={'Content-Type': 'application/json'})
            try:
                urllib.request.urlopen(del_req, timeout=10)
            except Exception:
                pass
        await app.resolve_peer(CHANNEL_ID)
        return True
    except Exception as e:
        log(f"Peer resolution failed: {e}", "ERROR")
        return False
    finally:
        try:
            app.remove_handler(*handler) if isinstance(handler, tuple) else app.remove_handler(handler)
        except Exception:
            pass


async def _find_latest_manifest(app):
    """Pinned message first; fall back to scanning recent messages."""
    if not await _ensure_peer(app):
        log("Channel peer unresolvable — cannot scan for snapshots", "WARN")
        return None
    try:
        chat = await app.get_chat(CHANNEL_ID)
        pinned = getattr(chat, 'pinned_message', None)
        if pinned and pinned.caption and MANIFEST_TAG in pinned.caption:
            return pinned
    except Exception:
        pass
    # Fallback: walk backwards from the newest message looking for the tag
    try:
        probe = await app.send_message(CHANNEL_ID, ".", disable_notification=True)
        latest_id = probe.id
        await probe.delete()
        for lo in range(latest_id, 0, -200):
            ids = list(range(max(1, lo - 199), lo + 1))[::-1]
            msgs = await app.get_messages(CHANNEL_ID, ids)
            if not isinstance(msgs, list):
                msgs = [msgs]
            for m in msgs:
                if m and m.caption and MANIFEST_TAG in m.caption and m.document:
                    return m
    except Exception as e:
        log(f"Manifest scan failed: {e}", "ERROR")
    return None


async def import_snapshot(progress_cb=None):
    """Restore the latest snapshot. Returns True on success."""
    from pyrogram import Client

    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
        log("lake.db already exists — importing anyway will OVERWRITE it", "WARN")

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    workdir = os.path.join(SNAPSHOT_DIR, 'restore')
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)

    app = Client(SNAP_SESSION, api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)
    await app.start()
    try:
        pointer = await _find_latest_manifest(app)
        if not pointer:
            log("No snapshot manifest found in channel", "WARN")
            return False

        mpath = await app.download_media(pointer, file_name=os.path.join(workdir, 'manifest.json'))
        with open(mpath) as f:
            manifest = json.load(f)
        log(f"Found snapshot {manifest['created_utc']} — {manifest['parts']} part(s)")

        # Download + verify + concatenate parts
        comp_path = os.path.join(workdir, 'snapshot.bin')
        with open(comp_path, 'wb') as out:
            for i, msg_id in enumerate(manifest['part_msg_ids']):
                log(f"Downloading part {i+1}/{manifest['parts']}...")
                msg = await app.get_messages(CHANNEL_ID, msg_id)
                part_path = await app.download_media(
                    msg, file_name=os.path.join(workdir, f'part{i:03d}'),
                    progress=progress_cb)
                got = _sha256(part_path)
                want = manifest['part_sha256'][i]
                if got != want:
                    raise RuntimeError(f"Part {i+1} sha256 mismatch — corrupt download")
                with open(part_path, 'rb') as pf:
                    shutil.copyfileobj(pf, out)
                os.remove(part_path)

        # Decompress + untar
        tar_path = os.path.join(workdir, 'snapshot.tar')
        _decompress_file(comp_path, tar_path, manifest.get('codec', 'zstd'))
        os.remove(comp_path)

        extract_dir = os.path.join(workdir, 'extracted')
        with tarfile.open(tar_path) as tar:
            tar.extractall(extract_dir, filter='data')
        os.remove(tar_path)

        # Restore into DataLake — remove stale WAL/SHM so sqlite doesn't
        # replay old journal pages over the restored file
        for suffix in ('', '-wal', '-shm'):
            p = DB_PATH + suffix
            if os.path.exists(p):
                os.remove(p)
        shutil.move(os.path.join(extract_dir, 'lake.db'), DB_PATH)

        restored_thumbs = os.path.join(extract_dir, '.thumbnails')
        if os.path.isdir(restored_thumbs):
            os.makedirs(THUMB_DIR, exist_ok=True)
            for fn in os.listdir(restored_thumbs):
                shutil.move(os.path.join(restored_thumbs, fn),
                            os.path.join(THUMB_DIR, fn))

        rebuild_dedup_set()
        log(f"Snapshot {manifest['created_utc']} restored — DB is live, zero reprocessing needed", "SUCCESS")
        return True
    finally:
        await app.stop()
        shutil.rmtree(workdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# DEDUP REBUILD — Redis set from the restored DB
# ═══════════════════════════════════════════════════════════
def rebuild_dedup_set():
    """PROCESSED_VIDEOS_SET ← every msg_id in the `videos` table.
    This is what makes 'never reprocess' true across sessions."""
    try:
        from queue_manager import get_redis
        r = get_redis()
        conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        try:
            rows = conn.execute("SELECT msg_id FROM videos").fetchall()
        except sqlite3.OperationalError:
            rows = []  # fresh DB, no videos table yet
        conn.close()
        if rows:
            r.sadd("PROCESSED_VIDEOS_SET", *[str(row[0]) for row in rows])
        log(f"Dedup set rebuilt: {len(rows)} processed videos registered")
        return len(rows)
    except Exception as e:
        log(f"Dedup rebuild failed: {e}", "ERROR")
        return 0


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "export"
    if cmd == "export":
        asyncio.run(export_snapshot())
    elif cmd == "import":
        asyncio.run(import_snapshot())
    elif cmd == "rebuild-dedup":
        rebuild_dedup_set()
    else:
        print("usage: python snapshot_manager.py [export|import|rebuild-dedup]")
