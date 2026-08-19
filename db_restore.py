"""
db_restore.py — rehydrate the database from the newest bundle in the channel.

The other half of db_export.py, and the half that makes the export worth
anything. Kaggle gives every session a fresh container: scratch is wiped, and
PostgreSQL runs from the Debian default data directory on that same ephemeral
disk, so the Omniscient store — every Qwen narrative the GPU produced last run —
is gone the moment the session ends. Export alone is write-only backup. With
restore, the channel becomes the durable tier and each session continues the
database instead of starting one.

The algorithm is the vault's ("Database snapshot and restore design"):

    1. read the pinned manifest from the artifact channel
    2. compare the bundle to what is already local
    3. download parts → reassemble → verify SHA-256 → decompress
    4. load index.sqlite, restore omnidb.sql
    5. every module opens the same paths it always did

Step 5 is the point: nothing downstream knows a restore happened. This module
writes into DB_PATH and into the live Postgres database, so the rest of VIOS
keeps its imports, its connection strings and its assumptions untouched.

Two things this deliberately does NOT do:

  * It does not merge. A bundle replaces what is local. Merging two divergent
    copies of `posts` needs a conflict rule nobody has specified, and the case
    that matters — a fresh container with an empty database — has no conflict to
    resolve. `mode="inspect"` exists so the destructive case (a local database
    that already holds more rows than the bundle) is visible *before* anything
    is overwritten rather than discovered afterwards.
  * It does not restore Qdrant or Neo4j, because the export does not ship them.
    Both are derived: vectors come from an encoder pass over the frames, the
    graph is projected from the Postgres narratives. Restoring Postgres and
    re-deriving is cheaper than shipping them.

Staging happens on SCRATCH_DIR, not the OUTPUT tier: a download is re-fetchable
by definition, so it has no claim on the 19.5 GB that has to survive the
session. The reverse of where export builds, for the same reason.

Transport
─────────
Downloads go over the HTTP Bot API through tg_transport, matching the export
side after MTProto wedged an upload indefinitely (see tg_transport.py). Two
consequences shape this module:

  * The Bot API cannot fetch an arbitrary message by id. So restore never looks
    a part up — schema v2 manifests carry each part's `file_id`, and `getFile`
    goes straight from that to the bytes. Finding the *manifest* still needs one
    lookup, which `getChat`'s `pinned_message` provides.
  * Downloads cap at 20 MB per file, which is why export now ships 18 MB parts.

Schema v1 bundles predate all of this: 480 MB parts addressed by message_id
only, restorable over MTProto alone. They are still accepted, through a fallback
that is wrapped in a hard deadline — the one thing the old code never had.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time

# The four credentials are read as `config.NAME` at each use site, never bound
# here: `from config import BOT_TOKEN` snapshots the value at import and this
# module is imported by the web process at startup, which is before a throttled
# Phase 0 has any chance of being retried. That binding is why a restore said
# "Telegram is not configured" for a whole session over a rate limit that cleared
# in a minute. `missing_telegram_secrets` is a function and recomputes, so
# importing it by name is safe and stays.
import config
from config import (SCRATCH_DIR, DB_PATH, SQLITE_TIMEOUT,
                    missing_telegram_secrets,
                    OMNI_PG_DB, OMNI_PG_USER, OMNI_PG_PASSWORD, OMNI_PG_HOST)
from db_export import BUNDLE_SCHEMA, EXPORT_SESSION, _sha256, _CHUNK, _mb
from logger import vios_log

RESTORE_DIR = os.path.join(SCRATCH_DIR, "restore")

# Layouts this build can read. v1 is MTProto-only (480 MB parts, no file_id);
# v2 adds file_id so the HTTP transport can fetch parts directly. Refusing an
# unknown layout is deliberate — guessing is how a restore writes garbage over
# a working database.
SUPPORTED_SCHEMAS = (1, 2)

# Ceiling on the MTProto fallback. pyrogram's session start retries an
# unreachable DC forever, which is precisely how the exporter used to hang; a
# v1 restore is allowed to be slow but not unbounded.
MTPROTO_DEADLINE = 900.0

# How far back to walk channel history when the manifest is not pinned. Pinning
# needs admin rights the bot may not have, and _upload_bundle records
# manifest["pinned"] = False rather than failing when it cannot pin — so the
# fallback is a normal path, not an error path. 600 messages is several days of
# harvesting; beyond that the bundle is old enough that restoring it would
# discard more than it recovers.
HISTORY_SCAN = 600

# Ids per messages.getMessages call. Telegram caps the request at 200.
_SCAN_BATCH = 200


# ═══════════════════════════════════════════════════════════
# JOB STATE
#   Same shape as db_export's, so the admin panel polls both
#   with one renderer.
# ═══════════════════════════════════════════════════════════
_lock = threading.Lock()
_job: dict = {
    "state": "idle",        # idle | running | ready | done | error
    "mode": "",             # inspect | apply
    "stage": "",
    "pct": 0,
    "detail": "",
    "started_at": None,
    "finished_at": None,
    "plan": None,           # what inspect found: bundle vs local
    "error": None,
    "log": [],
    # Age of the last transferred byte is what distinguishes a slow download
    # from a dead one — the distinction the old export UI could not make.
    "last_progress_at": None,
}

# The manifest the last inspect read, so pressing Restore does not re-download
# it. Cleared once applied — a manifest is a snapshot of the channel at one
# moment and should not be reused across a later session.
_pending: dict | None = None


def _set(**kw):
    with _lock:
        _job.update(kw)
        if "stage" in kw:
            line = f"{time.strftime('%H:%M:%S')} · {kw['stage']}"
            if kw.get("detail"):
                line += f" — {kw['detail']}"
            _job["log"] = (_job["log"] + [line])[-40:]


def restore_status() -> dict:
    """Snapshot for the panel, with `stalled_s` computed the same way export
    computes it: a frozen transfer should look different from a working one."""
    with _lock:
        st = dict(_job)
    if st["state"] == "running" and st.get("last_progress_at"):
        st["stalled_s"] = round(time.time() - st["last_progress_at"], 1)
    else:
        st["stalled_s"] = 0
    return st


def is_running() -> bool:
    with _lock:
        return _job["state"] == "running"


# ═══════════════════════════════════════════════════════════
# FINDING THE BUNDLE
# ═══════════════════════════════════════════════════════════
def _is_manifest(msg) -> bool:
    doc = getattr(msg, "document", None)
    name = getattr(doc, "file_name", "") or ""
    return name.startswith("manifest-") and name.endswith(".json")


async def _scan_for_manifest(client, seq: str | None):
    """Walk the channel backwards looking for a manifest document.

    Not get_chat_history(): messages.getHistory is a user-only MTProto method
    and returns BOT_METHOD_INVALID for a bot account — the same asymmetry the
    harvester's scanner documents. messages.getMessages *is* allowed for bots,
    so the newest id is probed the way the harvester probes it (post a message,
    read its id, delete it) and the walk runs backwards from there in batches.

    This is the fallback path only. The pinned manifest is one API call and
    costs nothing; this exists because pinning needs admin rights the bot may
    not have, and _upload_bundle records pinned=False rather than failing.
    """
    _set(stage="Scanning channel", pct=6,
         detail=f"looking for {'bundle ' + seq if seq else 'the newest manifest'}")

    ping = await client.send_message(config.CHANNEL_ID, ".", disable_notification=True)
    latest = ping.id
    try:
        await ping.delete()
    except Exception:
        pass          # a stray "." in the channel is not worth failing over

    wanted = f"manifest-{seq}.json" if seq else None
    floor = max(1, latest - HISTORY_SCAN)
    hi = latest
    while hi >= floor:
        lo = max(floor, hi - _SCAN_BATCH + 1)
        try:
            msgs = await client.get_messages(config.CHANNEL_ID, list(range(hi, lo - 1, -1)))
        except Exception as e:
            raise RuntimeError(f"Could not read the channel: {str(e)[:150]}")
        for msg in (msgs or []):
            # get_messages returns placeholders for deleted ids, so a null
            # document here is normal rather than an error.
            if msg is None or not _is_manifest(msg):
                continue
            if wanted and msg.document.file_name != wanted:
                continue
            return msg          # descending order, so the first hit is newest
        hi = lo - 1

    raise RuntimeError(
        f"No bundle manifest in the last {HISTORY_SCAN} channel messages. "
        f"Run an export first — a bundle only exists once its manifest "
        f"is posted.")


async def _read_manifest(client, work: str, seq: str | None = None) -> dict:
    """Locate and parse the manifest that commits the bundle to restore.

    Pinned first — that is one API call and is where _upload_bundle puts it.
    Backwards scan second, both because the pin can fail and because asking for
    a specific older `seq` means walking back to find it.
    """
    target = None

    if not seq:
        try:
            chat = await client.get_chat(config.CHANNEL_ID)
            pinned = getattr(chat, "pinned_message", None)
            if pinned is not None and _is_manifest(pinned):
                target = pinned
        except Exception as e:
            # An unpinned channel, or a bot without the rights to read chat
            # info. Neither is fatal; the scan below covers both.
            vios_log(f"pinned manifest unavailable ({str(e)[:100]}) — "
                     f"scanning history", "RESTORE", "WARN")

    if target is None:
        target = await _scan_for_manifest(client, seq)

    _set(stage="Reading manifest", pct=10, detail=target.document.file_name)
    path = os.path.join(work, "manifest.json")
    await client.download_media(target, file_name=path)
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return _validate_manifest(manifest)


def _local_counts() -> dict:
    """What is in the local harvest DB right now, in the manifest's own terms,
    so inspect can say plainly whether restoring would gain or lose rows."""
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
            try:
                counts["posts_with_file"] = con.execute(
                    "SELECT COUNT(*) FROM posts WHERE local_video_path IS NOT NULL"
                ).fetchone()[0]
            except sqlite3.Error:
                pass
        finally:
            con.close()
    except sqlite3.Error:
        pass          # no database yet — the fresh-container case, count of 0
    return counts


def _plan_from(manifest: dict) -> dict:
    """The decision the user is being asked to make, computed rather than
    narrated: which files are in the bundle, how big the download is, and how
    the bundle's row counts compare to the local ones."""
    bundle_counts = manifest.get("counts", {}) or {}
    local = _local_counts()
    files = {}
    for p in manifest["parts"]:
        f = files.setdefault(p["file"], {"parts": 0, "size": 0})
        f["parts"] += 1
        f["size"] += p.get("size", 0)
    has_pg = any(p["file"].startswith("omnidb") for p in manifest["parts"])

    # The one number that decides whether this is recovery or data loss.
    delta = None
    if "posts" in bundle_counts and "posts" in local:
        delta = bundle_counts["posts"] - local["posts"]

    return {
        "seq": manifest.get("seq"),
        "created_at": manifest.get("created_at"),
        "code_commit": manifest.get("code_commit"),
        "schema": manifest.get("schema"),
        "files": [{"name": k, **v} for k, v in files.items()],
        "download_mb": round(sum(p.get("size", 0)
                                 for p in manifest["parts"]) / 1048576, 1),
        "bundle_counts": bundle_counts,
        "local_counts": local,
        "posts_delta": delta,
        "destructive": bool(delta is not None and delta < 0),
        "has_postgres": has_pg,
        "effects": _effects(manifest, has_pg, bundle_counts, local),
    }


def _effects(manifest: dict, has_pg: bool, bundle: dict, local: dict) -> list:
    """Exactly what an apply will and will not touch, as a list the panel
    renders verbatim.

    This exists because "restore the database" hides a lot: it replaces two
    stores, leaves three others alone, and the ones it leaves alone are the
    reason a restored session still has to re-derive vectors and re-download
    video files before it behaves like the session that made the bundle.
    Spelling that out before the click is cheaper than discovering it after.
    """
    posts = bundle.get("posts")
    with_file = bundle.get("posts_with_file")
    out = [
        {"target": "lake.db (harvest index)", "action": "replaced",
         "impact": "yes",
         "detail": f"posts, creators, categories → {posts if posts is not None else '?'} "
                   f"posts. The current index is snapshotted to scratch first, "
                   f"so an unwanted restore can be undone within the session."},
    ]
    if has_pg:
        out.append(
            {"target": "PostgreSQL (omnidb)", "action": "replaced",
             "impact": "yes",
             "detail": "frames, chunks and every Qwen narrative. The schema is "
                       "dropped and replayed, so anything narrated in this "
                       "session and not in the bundle is lost."})
    else:
        out.append(
            {"target": "PostgreSQL (omnidb)", "action": "untouched",
             "impact": "no",
             "detail": "This bundle carries no Postgres dump — it was exported "
                       "from a machine with the Omniscient layer off. Local "
                       "narratives survive."})
    out += [
        {"target": "Qdrant (vectors)", "action": "untouched",
         "impact": "rebuild",
         "detail": "Never shipped in a bundle: vectors are a deterministic "
                   "encoder pass over the frames. Semantic search stays limited "
                   "to whatever this machine has already embedded until the "
                   "encoder re-runs."},
        {"target": "Neo4j (knowledge graph)", "action": "untouched",
         "impact": "rebuild",
         "detail": "Projected from the Postgres narratives at ingest, so it is "
                   "re-derivable rather than replicated. Graph queries reflect "
                   "the local projection, not the restored rows, until it is "
                   "re-projected."},
        {"target": "Video files on disk", "action": "untouched",
         "impact": "re-download",
         "detail": f"A bundle carries the index, not the media. "
                   f"{with_file if with_file is not None else '?'} restored "
                   f"posts point at local paths; on a fresh container those "
                   f"files are absent and the harvester re-fetches them from "
                   f"the channel on demand."},
        {"target": "Redis queues", "action": "untouched",
         "impact": "no",
         "detail": "In-flight jobs and dedup sets are session state. A restore "
                   "does not enqueue work; the pipeline picks up restored rows "
                   "the next time it scans."},
    ]
    return out


# ═══════════════════════════════════════════════════════════
# FETCH AND REASSEMBLE
# ═══════════════════════════════════════════════════════════
def _http_read_manifest(work: str, seq: str | None) -> dict | None:
    """Find and parse the manifest over the HTTP Bot API.

    Only the pinned message is reachable this way — the Bot API has no method
    to walk channel history. That covers the normal case, since export pins
    every manifest it posts for exactly this reason. Returns None when the pin
    is absent or is not the bundle asked for, leaving the MTProto scan to it.
    """
    import tg_transport as tg

    _set(stage="Reading pinned manifest", pct=6, detail="getChat")
    pinned = tg.get_pinned()
    if not pinned:
        return None

    doc = pinned.get("document") or {}
    name = doc.get("file_name") or ""
    if not (name.startswith("manifest-") and name.endswith(".json")):
        vios_log(f"pinned message is not a manifest ({name or 'no document'})"
                 " — falling back to a history scan", "RESTORE", "WARN")
        return None
    if seq and name != f"manifest-{seq}.json":
        # A specific older bundle was asked for; the pin is the newest one.
        return None
    if not doc.get("file_id"):
        return None

    _set(stage="Reading manifest", pct=10, detail=name)
    path = os.path.join(work, "manifest.json")
    tg.download_file(doc["file_id"], path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_manifest(manifest: dict) -> dict:
    got = int(manifest.get("schema", 0))
    if got not in SUPPORTED_SCHEMAS:
        raise RuntimeError(
            f"Bundle schema v{got} but this build understands "
            f"v{', v'.join(str(s) for s in SUPPORTED_SCHEMAS)}. "
            f"Update VIOS (bundle written by code "
            f"{manifest.get('code_commit')}) or export a fresh bundle.")
    if not manifest.get("parts"):
        raise RuntimeError("Manifest lists no parts — the bundle is empty.")
    return manifest


def _http_download_parts(manifest: dict, work: str) -> dict | None:
    """Fetch every part over HTTPS. None if this bundle cannot go that way.

    A part already on disk with the right hash is skipped, so an interrupted
    restore resumes rather than starting over.
    """
    import tg_transport as tg

    parts = manifest["parts"]
    if any(not p.get("file_id") for p in parts):
        return None                     # v1 bundle: message_id only
    oversize = [p for p in parts if p.get("size", 0) > tg.DOWNLOAD_LIMIT]
    if oversize:
        vios_log(f"{len(oversize)} part(s) exceed the Bot API's "
                 f"{tg.DOWNLOAD_LIMIT // 1048576} MB download cap — using "
                 f"MTProto", "RESTORE", "INFO")
        return None

    total = len(parts)
    grand = sum(p.get("size", 0) for p in parts)
    started = time.time()
    done = 0
    got: dict[str, list] = {}

    for n, part in enumerate(parts, 1):
        dest = os.path.join(work, part["name"])
        base = 12 + int(48 * (n - 1) / total)

        if os.path.exists(dest) and os.path.getsize(dest) == part.get("size"):
            _set(stage=f"Verifying part {n}/{total}", pct=base,
                 detail=f"{part['name']} — already on disk")
            if _sha256(dest) == part["sha256"]:
                got.setdefault(part["file"], []).append(
                    (part["part_index"], dest))
                done += part.get("size", 0)
                continue
            os.remove(dest)        # right size, wrong bytes: a truncated resume

        _set(stage=f"Downloading part {n}/{total}", pct=base,
             detail=f"{part['name']} · {_mb(part.get('size', 0))}")

        def progress(recv, size, _base=done, _p=part):
            now = time.time()
            rate = (_base + recv) / max(now - started, 0.001)
            _set(detail=f"{_p['name']} · {_mb(recv)} / {_mb(size)}"
                        f" · {_mb(rate)}/s",
                 last_progress_at=now)
            return True

        tg.download_file(part["file_id"], dest, progress=progress)
        done += part.get("size", 0)

        _set(stage=f"Checking part {n}/{total}", pct=base + 2, detail="sha256")
        actual = _sha256(dest)
        if actual != part["sha256"]:
            # Corruption here would be written straight into the live database
            # a few steps later. Stop while the damage is still a temp file.
            raise RuntimeError(
                f"Checksum mismatch on {part['name']}: expected "
                f"{part['sha256'][:16]}…, got {actual[:16]}…")
        got.setdefault(part["file"], []).append((part["part_index"], dest))

    _set(stage="All parts downloaded", pct=60,
         detail=f"{total} part(s) · {_mb(grand)} verified")
    return {name: [p for _, p in sorted(chunks)]
            for name, chunks in got.items()}


async def _download_parts(client, manifest: dict, work: str) -> dict:
    """Fetch every part, checksum it, and return {logical file: local path}.

    Sequential on purpose. The vault suggests parallel downloads and it would
    be faster, but a bot token that fans out on a channel earns a FloodWait,
    and a restore that trips one stalls the whole session start. A part already
    on disk with the right hash is skipped, so an interrupted restore resumes
    rather than starting over.
    """
    total = len(manifest["parts"])
    got: dict[str, list] = {}

    for n, part in enumerate(manifest["parts"], 1):
        dest = os.path.join(work, part["name"])
        base = 12 + int(48 * (n - 1) / total)
        mb = part.get("size", 0) / 1048576

        if os.path.exists(dest) and os.path.getsize(dest) == part.get("size"):
            _set(stage=f"Verifying part {n}/{total}", pct=base,
                 detail=f"{part['name']} — already on disk")
            if _sha256(dest) == part["sha256"]:
                got.setdefault(part["file"], []).append((part["part_index"], dest))
                continue
            os.remove(dest)        # right size, wrong bytes: a truncated resume

        _set(stage=f"Downloading part {n}/{total}", pct=base,
             detail=f"{part['name']} · {mb:.0f} MB")
        msg = await client.get_messages(config.CHANNEL_ID, part["message_id"])
        if msg is None or getattr(msg, "document", None) is None:
            raise RuntimeError(
                f"Part {part['name']} is missing from the channel "
                f"(message {part['message_id']} deleted?) — bundle "
                f"{manifest['seq']} cannot be restored.")
        await client.download_media(msg, file_name=dest)

        _set(stage=f"Checking part {n}/{total}", pct=base + 2,
             detail="sha256")
        actual = _sha256(dest)
        if actual != part["sha256"]:
            # Corruption here would be written straight into the live database
            # a few steps later. Stop while the damage is still a temp file.
            raise RuntimeError(
                f"Checksum mismatch on {part['name']}: expected "
                f"{part['sha256'][:16]}…, got {actual[:16]}…")
        got.setdefault(part["file"], []).append((part["part_index"], dest))

    return {name: [p for _, p in sorted(chunks)] for name, chunks in got.items()}


def _join(parts: list, dest: str) -> None:
    """Concatenate .partNNN files back into one. A single-part bundle is not
    copied — export kept its own name for exactly this case."""
    if len(parts) == 1 and parts[0] == dest:
        return
    if len(parts) == 1:
        os.replace(parts[0], dest)
        return
    with open(dest, "wb") as out:
        for p in parts:
            with open(p, "rb") as fin:
                shutil.copyfileobj(fin, out, _CHUNK)
            os.remove(p)          # scratch is large but not free


def _decompress(src: str, dst: str) -> None:
    """zstd → plain. Streaming, mirroring _compress: the Postgres dump is
    routinely larger than the RAM left after the models load."""
    import zstandard
    dctx = zstandard.ZstdDecompressor()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        dctx.copy_stream(fin, fout, read_size=_CHUNK, write_size=_CHUNK)


# ═══════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════
def _safety_copy(work: str) -> str | None:
    """Snapshot whatever is local before overwriting it.

    Restore is the one operation in VIOS that destroys data on purpose. It
    lands on scratch and dies with the session, which is the right lifetime: it
    exists to undo a restore the user regrets in the next ten minutes, not to
    be a second backup tier.
    """
    if not os.path.exists(DB_PATH):
        return None
    dest = os.path.join(work, f"pre-restore-{time.strftime('%H%M%S')}.sqlite")
    try:
        con = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        try:
            con.execute("VACUUM INTO ?", (dest,))
        finally:
            con.close()
        return dest
    except sqlite3.Error as e:
        vios_log(f"pre-restore snapshot failed: {e}", "RESTORE", "WARN")
        return None


def _load_sqlite(src: str) -> int:
    """Copy the restored database over the live one via SQLite's backup API.

    Not a file move. Every module in VIOS opens DB_PATH per call and several
    workers are mid-query at any moment; swapping the file underneath them
    leaves those connections reading a deleted inode, and on a WAL database it
    orphans the -wal/-shm sidecars. backup() does the replacement page by page
    inside the destination's own locking, so open connections either wait or
    see the new content — never a mixture.
    """
    source = sqlite3.connect(src, timeout=SQLITE_TIMEOUT)
    target = sqlite3.connect(DB_PATH, timeout=max(SQLITE_TIMEOUT, 60))
    try:
        source.backup(target)
        # VACUUM INTO writes a rollback-journal file, so the restored pages
        # arrive without WAL. Put it back: the whole UI reads concurrently with
        # the harvester's writes and depends on WAL not blocking them.
        target.execute("PRAGMA journal_mode=WAL;")
        target.execute("PRAGMA synchronous=NORMAL;")
        try:
            return target.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        except sqlite3.Error:
            return 0
    finally:
        source.close()
        target.close()


def _load_postgres(dump: str) -> str:
    """Replace the Omniscient database with the dump. Returns "" on success, or
    a short reason it was skipped.

    A reason rather than a bool because there are three distinct ways this does
    not happen — no psql, a schema that will not drop, a dump that will not
    replay — and reporting all of them as "skipped" sends the user looking for
    a missing binary that is right there.

    DROP SCHEMA first: pg_dump emits CREATE TABLE, which fails outright against
    the tables ensure_services() already created at boot. Dropping is safe in a
    way it would not be elsewhere — this database holds only derived pipeline
    output, and the dump about to be replayed is a superset of it.
    """
    if not shutil.which("psql"):
        vios_log("psql not on PATH — Postgres half of the bundle skipped",
                 "RESTORE", "WARN")
        return "psql not installed"
    env = dict(os.environ, PGPASSWORD=OMNI_PG_PASSWORD)
    base = ["psql", "-h", OMNI_PG_HOST, "-U", OMNI_PG_USER, "-d", OMNI_PG_DB,
            "-q", "-v", "ON_ERROR_STOP=1"]
    try:
        # A DROP waits behind every open transaction on these tables, so the
        # engine's own connections can stall this. Timing out and saying so
        # beats hanging the restore for an hour.
        wipe = subprocess.run(
            base + ["-c", "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"],
            env=env, capture_output=True, text=True, timeout=300)
        if wipe.returncode != 0:
            vios_log(f"could not clear Postgres schema: "
                     f"{(wipe.stderr or '')[:200]}", "RESTORE", "WARN")
            return "existing schema would not drop"
        load = subprocess.run(base + ["-f", dump], env=env,
                              capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        vios_log("psql timed out — another connection is holding a lock on the "
                 "Omniscient tables", "RESTORE", "WARN")
        return "timed out waiting on a table lock"
    except (subprocess.SubprocessError, OSError) as e:
        vios_log(f"psql failed to launch: {e}", "RESTORE", "WARN")
        return "psql would not run"
    if load.returncode != 0:
        # ON_ERROR_STOP means the schema is now partially loaded. Say so — a
        # half-restored Postgres reads as a working one until a query hits the
        # missing table.
        vios_log(f"psql restore exited {load.returncode}: "
                 f"{(load.stderr or '')[:300]}", "RESTORE", "ERROR")
        raise RuntimeError(
            f"Postgres restore failed partway: {(load.stderr or '')[:200]}")
    return ""


# ═══════════════════════════════════════════════════════════
# THE RESTORE
# ═══════════════════════════════════════════════════════════
async def _fetch_mtproto(mode: str, seq: str | None, work: str,
                         manifest: dict | None) -> tuple:
    """MTProto half, for what the Bot API cannot reach: unpinned manifests and
    v1 bundles with parts over the download cap."""
    from pyrogram import Client

    # Same separate session file the exporter used: the harvester holds its own
    # open for the life of the process and two Clients on one session file
    # fight over its SQLite lock. Export and restore never run together — the
    # admin routes refuse — so sharing one between them is fine.
    client = Client(EXPORT_SESSION, api_id=config.API_ID, api_hash=config.API_HASH,
                    bot_token=config.BOT_TOKEN)
    await client.start()
    try:
        if manifest is None:
            manifest = await _read_manifest(client, work, seq)
        if mode == "inspect":
            return manifest, {}
        return manifest, await _download_parts(client, manifest, work)
    finally:
        try:
            await client.stop()
        except Exception:
            pass


def _fetch(mode: str, seq: str | None, work: str) -> tuple:
    """Get the manifest, and for an apply the parts too.

    HTTPS first because it is bounded and proven from these containers;
    MTProto only for what it cannot do, and never without a deadline.
    """
    manifest = None
    try:
        manifest = _http_read_manifest(work, seq)
        if manifest is not None:
            manifest = _validate_manifest(manifest)
    except Exception as e:
        vios_log(f"HTTP manifest read failed ({str(e)[:140]}) — trying MTProto",
                 "RESTORE", "WARN")

    if manifest is not None:
        if mode == "inspect":
            return manifest, {}
        files = _http_download_parts(manifest, work)
        if files is not None:
            return manifest, files
        _set(stage="Falling back to MTProto",
             detail="parts exceed the Bot API download cap")

    # Anything still unresolved needs MTProto. Bounded, unlike the code this
    # replaced: a wedged session start now fails the job instead of parking it.
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            asyncio.wait_for(_fetch_mtproto(mode, seq, work, manifest),
                             timeout=MTPROTO_DEADLINE))
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Telegram did not respond within {int(MTPROTO_DEADLINE / 60)} "
            f"minutes over MTProto. The bundle is still in the channel; "
            f"re-run the restore.")
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _outcome(plan: dict, loaded: list, snapshot: str | None) -> dict:
    """What the restore actually did, read back from the live stores.

    `_effects` is a forecast written before anything moves; this is the
    measurement taken afterwards. They are separate on purpose — the question
    "what will happen" and the question "what happened" have different answers
    when Postgres declines to drop a locked schema, and reporting the forecast
    as the result is how a half-restored database looks finished.
    """
    after = _local_counts()
    bundle = plan.get("bundle_counts", {}) or {}
    return {
        "loaded": loaded,
        "counts_after": after,
        # The check that matters: the live database should now agree with the
        # manifest. A mismatch means something did not land.
        "matches_bundle": (after.get("posts") == bundle.get("posts")
                           if "posts" in bundle and "posts" in after else None),
        "snapshot": snapshot,
        "next_steps": [
            "The UI reads lake.db per request, so restored posts appear on the "
            "next page load — no restart needed.",
            "Video files are not in a bundle. Restored posts whose files are "
            "absent get re-downloaded from the channel by the harvester.",
            "Qdrant and Neo4j were not touched. Semantic search and graph "
            "queries reflect this machine's own derivation until the encoder "
            "and the projector re-run over the restored rows.",
        ] + ([f"The pre-restore database is at {snapshot} (scratch — it dies "
              f"with this session)."] if snapshot else []),
    }


def _run(mode: str, seq: str | None) -> None:
    """Body of the restore thread."""
    global _pending
    work = os.path.join(RESTORE_DIR, seq or "latest")
    try:
        os.makedirs(work, exist_ok=True)
        _set(state="running", mode=mode, pct=1, started_at=time.time(),
             finished_at=None, error=None, plan=None, log=[],
             stage="Contacting Telegram",
             detail="reading the pinned manifest" if mode == "inspect"
                    else "preparing to restore")

        absent = missing_telegram_secrets()
        if absent:
            raise RuntimeError(
                f"Telegram is not configured ({', '.join(absent)}) — the "
                f"channel is where bundles live, so restore needs it.")

        manifest, files = _fetch(mode, seq, work)

        plan = _plan_from(manifest)

        if mode == "inspect":
            _pending = manifest
            note = ("Restoring would REPLACE a larger local database"
                    if plan["destructive"] else
                    "Local database is empty or older — restoring recovers rows")
            _set(state="ready", stage="Bundle found", pct=100, plan=plan,
                 detail=f"bundle {plan['seq']} · {plan['download_mb']} MB · {note}",
                 finished_at=time.time())
            return

        # ── Reassemble ──
        _set(stage="Reassembling", pct=62,
             detail=f"{sum(len(v) for v in files.values())} part(s)")
        ready = {}
        for logical, parts in files.items():
            joined = os.path.join(work, logical)
            _join(parts, joined)
            plain = joined[:-4] if joined.endswith(".zst") else joined + ".out"
            _set(stage=f"Decompressing {logical}", pct=68)
            _decompress(joined, plain)
            os.remove(joined)
            ready[logical] = plain

        # ── Load ──
        snapshot = _safety_copy(work)
        loaded = []

        sqlite_src = ready.get("index.sqlite.zst")
        if sqlite_src:
            _set(stage="Loading harvest database", pct=80,
                 detail="page-level copy over lake.db")
            rows = _load_sqlite(sqlite_src)
            loaded.append(f"{rows} posts")
            os.remove(sqlite_src)

        pg_src = ready.get("omnidb.sql.zst")
        if pg_src:
            _set(stage="Restoring Postgres", pct=90,
                 detail="frames, chunks and narratives")
            why = _load_postgres(pg_src)
            loaded.append("Omniscient store" if not why
                          else f"Postgres skipped ({why})")
            os.remove(pg_src)

        _pending = None
        tail = f" · previous database kept at {snapshot}" if snapshot else ""
        plan["outcome"] = _outcome(plan, loaded, snapshot)
        _set(state="done", stage="Restored", pct=100, plan=plan,
             detail=f"bundle {plan['seq']} → {', '.join(loaded) or 'nothing'}"
                    + tail,
             finished_at=time.time())
        vios_log(f"bundle {plan['seq']} restored: {', '.join(loaded)}",
                 "RESTORE", "SUCCESS")

    except Exception as e:
        _set(state="error", stage="Failed", detail=str(e)[:280],
             error=str(e)[:300], finished_at=time.time())
        vios_log(f"restore failed: {e}", "RESTORE", "ERROR")


def start_restore(mode: str = "inspect", seq: str | None = None) -> dict:
    """Kick off an inspect or an apply. Returns immediately; poll
    restore_status().

    `inspect` is non-destructive and reads only the manifest: it exists so the
    admin panel can show what a restore would do — which bundle, how much to
    download, how its row counts compare to the local ones — before the user
    authorises overwriting anything.
    """
    if mode not in ("inspect", "apply"):
        return {"ok": False, "error": f"Unknown restore mode {mode!r}."}
    with _lock:
        if _job["state"] == "running":
            return {"ok": False, "error": "A restore is already running."}
    try:
        import db_export
        if db_export.is_running():
            # Both drive the same pyrogram session file, and an apply would be
            # dumping the very database the export is reading.
            return {"ok": False,
                    "error": "An export is running — wait for it to finish."}
    except Exception:
        pass

    # An apply always re-reads the manifest from the channel rather than
    # trusting whatever inspect cached, because the two can be minutes apart
    # and a newer bundle may have landed in between. Pinning the sequence is
    # how the user asks for the exact bundle they were shown.
    if mode == "apply" and seq is None and _pending:
        seq = _pending.get("seq")

    t = threading.Thread(target=_run, args=(mode, seq),
                         name="vios-db-restore", daemon=True)
    t.start()
    return {"ok": True, "mode": mode, "seq": seq}
