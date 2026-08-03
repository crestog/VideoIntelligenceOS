"""
db_export.py — seal the database into a versioned bundle and upload it to Telegram.

Design comes from the Obsidian vault notes "Database snapshot and restore design"
and "Free storage providers and Telegram risk verified 2026". The second note
reverses the first on one point, and this module follows the reversal:

  * The first note said "don't write an uploader — use rclone with the teldrive
    backend."
  * The second note retracts that for VIOS: the backend is a *fork* of rclone
    (fragile to keep built), teldrive drags in an external Postgres as a single
    point of failure for the part→message mapping, and the project vanished for
    a month in March 2025. Verdict: "Write the thin uploader; keep the
    manifest-in-channel design." teldrive stays useful as prior art only.

The invariant that makes the design work: **a bundle exists if and only if its
manifest message is posted.** Parts are uploaded first and are inert on their
own — a run that dies halfway leaves unreferenced parts, never a corrupt bundle
that restore might believe. The manifest carries every part's message_id and
SHA-256, so the channel is self-describing given only a bot token; no external
metadata store, which is precisely teldrive's trap.

What goes in, and what deliberately does not:

  * index.sqlite.zst  — the harvest DB (posts, creators, categories). Canonical;
                        nothing else can reproduce which reels were downloaded.
  * omnidb.sql.zst    — Postgres dump: frames and chunks, and with them the Qwen
                        narratives. Expensive GPU output, not reproducible
                        without re-running the whole pipeline.
  * Qdrant vectors    — omitted. Derived from the frames by a deterministic
                        encoder pass, and the vault's own size budget puts them
                        at gigabytes. Rebuildable beats replicated.
  * Neo4j graph       — omitted. Projected from the Postgres narratives at
                        ingest; restoring Postgres and re-projecting is cheaper
                        than shipping a JVM store.

Bundles are built under BASE_DIR (Kaggle's OUTPUT tier, 19.5 GB quota) rather
than scratch: a bundle is the one artifact that must outlive the session even if
the upload fails, and OUTPUT is the only tier Kaggle keeps.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time

from config import (BASE_DIR, LAKE_DIR, DB_PATH, SQLITE_TIMEOUT, CHANNEL_ID,
                    API_ID, API_HASH, BOT_TOKEN, missing_telegram_secrets,
                    OMNI_PG_DB, OMNI_PG_USER, OMNI_PG_PASSWORD, OMNI_PG_HOST)
from logger import vios_log

# Derived here rather than imported: ui_server owns the harvester's session path
# and importing it back would make this module depend on the web app. The two
# only need to agree on the directory, not on the file.
EXPORT_SESSION = os.path.join(LAKE_DIR, 'bot_session_export')

# Schema version of the bundle layout itself. Restore refuses a bundle whose
# major version it does not understand, rather than guessing at the layout.
BUNDLE_SCHEMA = 1

EXPORT_DIR = os.path.join(BASE_DIR, "exports")

# One message per part. The vault caps parts at 512 MB — well under Telegram's
# 2 GB free per-file limit, and small parts parallelise restore. 480 MB leaves
# headroom so a part can never land on the wrong side of the cap.
PART_SIZE = 480 * 1024 * 1024

# zstd level 10: within ~2% of 19's ratio on SQLite pages at a fraction of the
# CPU. The Kaggle vCPUs are shared with the encoder workers.
ZSTD_LEVEL = 10

_CHUNK = 4 * 1024 * 1024


# ═══════════════════════════════════════════════════════════
# JOB STATE
#   One export at a time. The UI polls /api/admin/export/status.
# ═══════════════════════════════════════════════════════════
_lock = threading.Lock()
_job: dict = {
    "state": "idle",        # idle | running | done | error
    "stage": "",
    "pct": 0,
    "detail": "",
    "started_at": None,
    "finished_at": None,
    "bundle": None,
    "error": None,
    "log": [],
}


def _set(**kw):
    with _lock:
        _job.update(kw)
        if "stage" in kw:
            line = f"{time.strftime('%H:%M:%S')} · {kw['stage']}"
            if kw.get("detail"):
                line += f" — {kw['detail']}"
            _job["log"] = (_job["log"] + [line])[-40:]


def export_status() -> dict:
    with _lock:
        return dict(_job)


def is_running() -> bool:
    with _lock:
        return _job["state"] == "running"


# ═══════════════════════════════════════════════════════════
# BUNDLE PIECES
# ═══════════════════════════════════════════════════════════
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _snapshot_sqlite(dest: str) -> None:
    """Consistent copy of the harvest DB via VACUUM INTO.

    Not shutil.copy: the harvester writes continuously, and copying a live
    SQLite file can capture a torn page or miss the WAL entirely. VACUUM INTO
    takes a read transaction, so the snapshot is a real point in time, and it
    compacts free pages on the way out.
    """
    con = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    try:
        con.execute("VACUUM INTO ?", (dest,))
    finally:
        con.close()


def _dump_postgres(dest: str) -> bool:
    """pg_dump the Omniscient DB. False (not an exception) when unavailable —
    a machine running with the Omniscient layer disabled still deserves a
    bundle of its harvest DB."""
    if not shutil.which("pg_dump"):
        vios_log("pg_dump not on PATH — bundle will omit Postgres", "EXPORT", "WARN")
        return False
    env = dict(os.environ, PGPASSWORD=OMNI_PG_PASSWORD)
    cmd = ["pg_dump", "-h", OMNI_PG_HOST, "-U", OMNI_PG_USER, "-d", OMNI_PG_DB,
           "--no-owner", "--no-acl", "-f", dest]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    except (subprocess.SubprocessError, OSError) as e:
        vios_log(f"pg_dump failed to launch: {e}", "EXPORT", "WARN")
        return False
    if p.returncode != 0:
        vios_log(f"pg_dump exited {p.returncode}: "
                           f"{(p.stderr or '')[:200]}", "EXPORT", "WARN")
        return False
    return os.path.exists(dest) and os.path.getsize(dest) > 0


def _compress(src: str, dst: str) -> None:
    """Stream src → zstd(dst). Streaming, not one-shot: the Postgres dump can
    exceed the amount of RAM this notebook has left after the models load."""
    import zstandard
    cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=-1)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        cctx.copy_stream(fin, fout, read_size=_CHUNK, write_size=_CHUNK)


def _split(path: str, part_size: int = PART_SIZE) -> list:
    """Split into .partNNN files, or return [path] when it already fits.

    Returns the list of files to upload. A single-part bundle keeps its own
    name so the common case has no reassembly step at all.
    """
    size = os.path.getsize(path)
    if size <= part_size:
        return [path]
    parts, idx = [], 0
    with open(path, "rb") as fin:
        while True:
            part = f"{path}.part{idx:03d}"
            written = 0
            with open(part, "wb") as fout:
                while written < part_size:
                    block = fin.read(min(_CHUNK, part_size - written))
                    if not block:
                        break
                    fout.write(block)
                    written += len(block)
            if written == 0:
                os.remove(part)
                break
            parts.append(part)
            idx += 1
            if written < part_size:
                break
    os.remove(path)          # the joined file is redundant once split
    return parts


# ═══════════════════════════════════════════════════════════
# THE EXPORT
# ═══════════════════════════════════════════════════════════
def _work_dir(seq: str) -> str:
    return os.path.join(EXPORT_DIR, f"bundle-v{BUNDLE_SCHEMA}-{seq}")


def _build_bundle(seq: str) -> dict:
    """Produce the bundle directory and return its manifest (minus upload info)."""
    work = _work_dir(seq)
    os.makedirs(work, exist_ok=True)

    files = []

    _set(stage="Snapshotting SQLite", pct=5, detail="VACUUM INTO")
    raw_sqlite = os.path.join(work, "index.sqlite")
    _snapshot_sqlite(raw_sqlite)
    raw_size = os.path.getsize(raw_sqlite)

    _set(stage="Compressing SQLite", pct=15,
         detail=f"{raw_size / 1048576:.0f} MB → zstd")
    sqlite_zst = raw_sqlite + ".zst"
    _compress(raw_sqlite, sqlite_zst)
    os.remove(raw_sqlite)
    files.append(("index.sqlite.zst", sqlite_zst, raw_size))

    _set(stage="Dumping Postgres", pct=30, detail="frames, chunks, narratives")
    raw_pg = os.path.join(work, "omnidb.sql")
    if _dump_postgres(raw_pg):
        pg_size = os.path.getsize(raw_pg)
        _set(stage="Compressing Postgres dump", pct=42,
             detail=f"{pg_size / 1048576:.0f} MB → zstd")
        pg_zst = raw_pg + ".zst"
        _compress(raw_pg, pg_zst)
        os.remove(raw_pg)
        files.append(("omnidb.sql.zst", pg_zst, pg_size))
    else:
        _set(stage="Postgres skipped", pct=42,
             detail="Omniscient layer unavailable — SQLite only")
        if os.path.exists(raw_pg):
            os.remove(raw_pg)

    _set(stage="Splitting into parts", pct=50)
    entries = []
    for logical, path, uncompressed in files:
        for i, part in enumerate(_split(path)):
            entries.append({
                "file": logical,
                "part_index": i,
                "local_path": part,
                "name": os.path.basename(part),
                "size": os.path.getsize(part),
                "sha256": _sha256(part),
                "uncompressed_size": uncompressed,
            })

    return {
        "schema": BUNDLE_SCHEMA,
        "seq": seq,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": _git_commit(),
        "counts": _row_counts(),
        "work_dir": work,
        "parts": entries,
    }


def _git_commit() -> str:
    """Which code wrote this bundle. Restoring a bundle into an incompatible
    schema is the failure this makes diagnosable."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _row_counts() -> dict:
    """Cheap integrity signal: a restored bundle whose counts do not match its
    manifest is corrupt in a way checksums cannot see."""
    counts = {}
    try:
        con = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        try:
            for table in ("posts", "creators", "categories"):
                try:
                    counts[table] = con.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except sqlite3.Error:
                    pass
            counts["posts_with_file"] = con.execute(
                "SELECT COUNT(*) FROM posts WHERE local_video_path IS NOT NULL"
            ).fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as e:
        vios_log(f"row counts unavailable: {e}", "EXPORT", "WARN")
    return counts


async def _upload_bundle(manifest: dict) -> dict:
    """Upload parts, then the manifest as the commit point, then pin it."""
    from pyrogram import Client

    # A separate session file: the harvester holds its own session open for the
    # lifetime of the process, and two pyrogram Clients on one session file
    # fight over its SQLite lock.
    client = Client(EXPORT_SESSION, api_id=API_ID, api_hash=API_HASH,
                    bot_token=BOT_TOKEN)
    await client.start()
    try:
        total = len(manifest["parts"])
        for n, part in enumerate(manifest["parts"], 1):
            _set(stage=f"Uploading part {n}/{total}", pct=55 + int(35 * n / max(total, 1)),
                 detail=f"{part['name']} · {part['size'] / 1048576:.0f} MB")
            caption = (f"`{part['name']}`\n"
                       f"bundle `{manifest['seq']}` · part {part['part_index']}\n"
                       f"sha256 `{part['sha256'][:16]}…`")
            msg = await client.send_document(
                CHANNEL_ID, part["local_path"], caption=caption,
                file_name=part["name"], disable_notification=True,
                force_document=True)
            part["message_id"] = msg.id
            part.pop("local_path", None)      # a message id, not a local path,
                                              # is what restore needs

        # The commit point. Parts above are inert until this lands, so a run
        # that dies mid-upload leaves orphans rather than a half-bundle that
        # restore would trust.
        _set(stage="Posting manifest", pct=93, detail="commit point")
        man_path = os.path.join(manifest["work_dir"], "manifest.json")
        publishable = {k: v for k, v in manifest.items() if k != "work_dir"}
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(publishable, f, indent=2)

        c = manifest.get("counts", {})
        msg = await client.send_document(
            CHANNEL_ID, man_path, file_name=f"manifest-{manifest['seq']}.json",
            caption=(f"✅ **VIOS bundle `{manifest['seq']}`**\n"
                     f"schema v{manifest['schema']} · code `{manifest['code_commit']}`\n"
                     f"{total} part(s) · "
                     f"{sum(p['size'] for p in manifest['parts']) / 1048576:.0f} MB\n"
                     f"posts `{c.get('posts', '?')}` · "
                     f"with file `{c.get('posts_with_file', '?')}`\n"
                     f"_Restore reads this message first._"),
            disable_notification=True, force_document=True)
        manifest["manifest_message_id"] = msg.id

        # Pinned, so restore finds the newest bundle in one API call instead of
        # walking channel history.
        try:
            await client.pin_chat_message(CHANNEL_ID, msg.id,
                                          disable_notification=True)
            manifest["pinned"] = True
        except Exception as e:
            # A pin needs admin rights; the bundle is already committed and
            # restore can fall back to a history scan.
            manifest["pinned"] = False
            vios_log(f"manifest posted but pin failed: "
                               f"{str(e)[:120]}", "EXPORT", "WARN")
        return manifest
    finally:
        try:
            await client.stop()
        except Exception:
            pass


def _run(keep_local: bool) -> None:
    """Body of the export thread."""
    seq = time.strftime("%Y%m%d-%H%M%S")
    committed = False          # manifest posted → the bundle is real, keep it
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        _set(state="running", stage="Starting", pct=1, detail=f"bundle {seq}",
             started_at=time.time(), finished_at=None, bundle=None, error=None,
             log=[])

        manifest = _build_bundle(seq)
        committed = True       # on disk and self-consistent from here on

        absent = missing_telegram_secrets()
        if absent:
            # The bundle is on disk and valid; only the transport is missing.
            _set(state="done", stage="Built locally", pct=100,
                 detail=f"Telegram disabled ({', '.join(absent)}) — bundle kept "
                        f"at {manifest['work_dir']}",
                 finished_at=time.time(),
                 bundle={k: v for k, v in manifest.items() if k != "parts"})
            vios_log(f"bundle {seq} built but not uploaded: "
                               f"missing {', '.join(absent)}", "EXPORT", "WARN")
            return

        import asyncio
        # Own loop in this thread: the harvester owns the main loop and uvicorn
        # owns another, and neither is safe to schedule a long upload onto.
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            manifest = loop.run_until_complete(_upload_bundle(manifest))
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        work = manifest.pop("work_dir", None)
        if not keep_local and work and os.path.isdir(work):
            # Telegram holds it now, and OUTPUT has a 19.5 GB quota to respect.
            shutil.rmtree(work, ignore_errors=True)

        _set(state="done", stage="Uploaded", pct=100,
             detail=f"{len(manifest['parts'])} part(s) + manifest "
                    f"(message {manifest.get('manifest_message_id')})"
                    + ("" if manifest.get("pinned") else " · pin failed"),
             finished_at=time.time(), bundle=manifest)
        vios_log(f"bundle {seq} uploaded — manifest message "
                           f"{manifest.get('manifest_message_id')}", "EXPORT", "SUCCESS")
    except Exception as e:
        work = _work_dir(seq)
        if not committed and os.path.isdir(work):
            # Failed while building: what is on disk is a partial bundle, worth
            # nothing and sitting in the output tier whose quota ends Kaggle
            # sessions. Running out of space is a likely way to get here, so
            # leaving the debris would make the next attempt fail too.
            shutil.rmtree(work, ignore_errors=True)
            tail = ""
        else:
            # Failed while uploading: the bundle itself is complete and valid.
            # Keep it — the user can retry or download it from the notebook.
            tail = f" · bundle kept at {work}"
        _set(state="error", stage="Failed", detail=(str(e)[:260] + tail),
             error=str(e)[:300], finished_at=time.time())
        vios_log(f"bundle {seq} failed: {e}", "EXPORT", "ERROR")


def start_export(keep_local: bool = False) -> dict:
    """Kick off an export. Returns immediately; poll export_status()."""
    with _lock:
        if _job["state"] == "running":
            return {"ok": False, "error": "An export is already running."}
    t = threading.Thread(target=_run, args=(keep_local,),
                         name="vios-db-export", daemon=True)
    t.start()
    return {"ok": True}


def list_local_bundles() -> list:
    """Bundles still on disk, newest first — what --keep-local left behind."""
    if not os.path.isdir(EXPORT_DIR):
        return []
    out = []
    for name in os.listdir(EXPORT_DIR):
        path = os.path.join(EXPORT_DIR, name)
        if not os.path.isdir(path):
            continue
        try:
            size = sum(os.path.getsize(os.path.join(path, f))
                       for f in os.listdir(path)
                       if os.path.isfile(os.path.join(path, f)))
            out.append({"name": name, "size_mb": round(size / 1048576, 1),
                        "created": os.path.getmtime(path)})
        except OSError:
            pass
    return sorted(out, key=lambda b: b["created"], reverse=True)
