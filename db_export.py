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

Transport
─────────
Uploading went through pyrogram (MTProto) until it wedged on Kaggle — the panel
sat at "Uploading part 1/2 · 72%" forever. Nothing in the path carried a
timeout, so a stalled upload blocked the export thread permanently, and the 72%
was a per-part constant rather than a measurement, so the bar could not show
that nothing was moving. See tg_transport.py for the full account.

Uploads now go over the HTTP Bot API through tg_transport, where every call has
a deadline and reports real byte counts. That caps a document at 50 MB and a
download at 20 MB, so PART_SIZE drops to 18 MB and each part is both uploadable
and downloadable over plain HTTPS. Each part records its `file_id`, which is
what lets restore fetch it without the message lookup the Bot API does not
offer.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time

# The four Telegram credentials were imported here and never used — the upload
# goes through tg_transport, which reads them itself. Dropped rather than
# rewritten, because `from config import BOT_TOKEN` snapshots the value at import
# and an unused snapshot is a trap waiting for the first person who needs one.
from config import (BASE_DIR, LAKE_DIR, DB_PATH, SQLITE_TIMEOUT,
                    missing_telegram_secrets,
                    OMNI_PG_DB, OMNI_PG_USER, OMNI_PG_PASSWORD, OMNI_PG_HOST)
from logger import vios_log

# Derived here rather than imported: ui_server owns the harvester's session path
# and importing it back would make this module depend on the web app. The two
# only need to agree on the directory, not on the file.
EXPORT_SESSION = os.path.join(LAKE_DIR, 'bot_session_export')

# Schema version of the bundle layout itself. Restore refuses a bundle whose
# major version it does not understand, rather than guessing at the layout.
#   v1 — parts addressed by message_id only (MTProto restore)
#   v2 — parts also carry file_id, so restore works over the HTTP Bot API
BUNDLE_SCHEMA = 2

EXPORT_DIR = os.path.join(BASE_DIR, "exports")

# One message per part. The old 480 MB cap suited MTProto's 2 GB per-file
# limit; the Bot API caps uploads at 50 MB and *downloads* at 20 MB, and the
# download side is the binding one — a part restore cannot fetch is a part that
# might as well not be in the channel. 18 MB leaves headroom under 20.
PART_SIZE = max(1, int(os.environ.get("VIOS_PART_MB", "18"))) * 1024 * 1024

# zstd level 10: within ~2% of 19's ratio on SQLite pages at a fraction of the
# CPU. The Kaggle vCPUs are shared with the encoder workers.
ZSTD_LEVEL = 10

_CHUNK = 4 * 1024 * 1024

# Local record of a built bundle, written before the first byte is uploaded so
# an interrupted run can resume instead of rebuilding. Holds local paths, which
# is exactly what the published manifest must not.
LOCAL_MANIFEST = "_local.json"


# ═══════════════════════════════════════════════════════════
# JOB STATE
#   One export at a time. The UI polls /api/admin/export/status.
# ═══════════════════════════════════════════════════════════
_lock = threading.Lock()
_cancel = threading.Event()
_job: dict = {
    "state": "idle",        # idle | running | done | error | cancelled
    "stage": "",
    "pct": 0,
    "detail": "",
    "started_at": None,
    "finished_at": None,
    "bundle": None,
    "error": None,
    "log": [],
    # Byte-level transfer state. The old panel showed a per-part constant and
    # so could not distinguish "slow" from "wedged"; these are measured.
    "sent_bytes": 0,
    "total_bytes": 0,
    "part_sent": 0,
    "part_total": 0,
    "rate_bps": 0,
    "eta_s": None,
    "last_progress_at": None,
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
    """Snapshot for the panel.

    `stalled_s` is computed here rather than stored: it is the age of the last
    byte, and the whole reason this rewrite exists is that a frozen transfer
    used to be indistinguishable from a working one.
    """
    with _lock:
        st = dict(_job)
    if st["state"] == "running" and st.get("last_progress_at"):
        st["stalled_s"] = round(time.time() - st["last_progress_at"], 1)
    else:
        st["stalled_s"] = 0
    st["cancelling"] = _cancel.is_set() and st["state"] == "running"
    return st


def is_running() -> bool:
    with _lock:
        return _job["state"] == "running"


def cancel_export() -> dict:
    """Ask a running export to stop.

    Cooperative: the flag is read between stages and inside the upload
    progress callback, so a stop lands within a chunk rather than instantly.
    That is enough — every call it could be sitting in now has a deadline.
    """
    if not is_running():
        return {"ok": False, "error": "No export is running."}
    _cancel.set()
    _set(stage="Cancelling", detail="stopping after the current chunk")
    return {"ok": True}


class _Cancelled(RuntimeError):
    """Raised on the export thread once the cancel flag is seen."""


def _check_cancel():
    if _cancel.is_set():
        raise _Cancelled("Export cancelled from the panel.")


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
    """Produce the bundle directory and return its manifest (minus upload info).

    Cancellation is checked between stages rather than inside them: VACUUM INTO
    and pg_dump are single blocking calls with no callback to hook, so the
    honest granularity is "after each stage".
    """
    work = _work_dir(seq)
    os.makedirs(work, exist_ok=True)

    files = []

    _check_cancel()
    _set(stage="Snapshotting SQLite", pct=5, detail="VACUUM INTO")
    raw_sqlite = os.path.join(work, "index.sqlite")
    _snapshot_sqlite(raw_sqlite)
    raw_size = os.path.getsize(raw_sqlite)

    _check_cancel()
    _set(stage="Compressing SQLite", pct=15,
         detail=f"{raw_size / 1048576:.0f} MB → zstd")
    sqlite_zst = raw_sqlite + ".zst"
    _compress(raw_sqlite, sqlite_zst)
    os.remove(raw_sqlite)
    files.append(("index.sqlite.zst", sqlite_zst, raw_size))

    _check_cancel()
    _set(stage="Dumping Postgres", pct=30, detail="frames, chunks, narratives")
    raw_pg = os.path.join(work, "omnidb.sql")
    if _dump_postgres(raw_pg):
        pg_size = os.path.getsize(raw_pg)
        _check_cancel()
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

    _check_cancel()
    _set(stage="Splitting into parts", pct=48,
         detail=f"{PART_SIZE // 1048576} MB each")
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
    _set(stage="Bundle built", pct=50,
         detail=f"{len(entries)} part(s) · "
                f"{_mb(sum(e['size'] for e in entries))}")

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


def _upload_bundle(manifest: dict) -> dict:
    """Upload parts, then the manifest as the commit point, then pin it.

    Synchronous, over the HTTP Bot API. The previous version was async
    pyrogram and had no deadline anywhere in it; every call here is bounded by
    tg_transport, so this either completes, raises, or is cancelled.

    Resumable: a part that already carries a message_id was uploaded by an
    earlier attempt and is skipped. Re-posting it would leave an orphan
    consuming channel space and would not make the bundle any more complete.
    """
    import tg_transport as tg

    total = len(manifest["parts"])
    grand = sum(p["size"] for p in manifest["parts"])
    done_bytes = sum(p["size"] for p in manifest["parts"] if p.get("message_id"))
    _set(total_bytes=grand, sent_bytes=done_bytes,
         last_progress_at=time.time())

    # Throughput is measured over what *this* run transfers. Bytes inherited
    # from a resumed attempt cost no time now, so counting them would report a
    # rate the link never achieved and an ETA that is always too optimistic.
    started = time.time()
    resume_base = done_bytes

    for n, part in enumerate(manifest["parts"], 1):
        _check_cancel()

        if part.get("message_id"):
            _set(stage=f"Part {n}/{total} already uploaded", pct=_pct(done_bytes, grand),
                 detail=f"{part['name']} · resuming after it")
            continue

        base_done = done_bytes
        _set(stage=f"Uploading part {n}/{total}", pct=_pct(base_done, grand),
             detail=f"{part['name']} · {_mb(part['size'])}",
             part_sent=0, part_total=part["size"],
             last_progress_at=time.time())

        def progress(sent, part_total, _base=base_done, _part=part):
            """Called per chunk by the transport. Returning False cancels."""
            if _cancel.is_set():
                return False
            now = time.time()
            elapsed = max(now - started, 0.001)
            moved = _base + sent
            rate = (moved - resume_base) / elapsed
            remain = (grand - moved) / rate if rate > 1 else None
            _set(pct=_pct(moved, grand),
                 sent_bytes=moved, part_sent=sent, part_total=part_total,
                 rate_bps=int(rate), last_progress_at=now,
                 eta_s=int(remain) if remain is not None else None,
                 detail=f"{_part['name']} · {_mb(sent)} / {_mb(part_total)}"
                        f" · {_mb(rate)}/s")
            return True

        sent = tg.send_document(
            part["local_path"],
            caption=(f"{part['name']}\n"
                     f"bundle {manifest['seq']} · part {part['part_index']}\n"
                     f"sha256 {part['sha256'][:16]}…"),
            file_name=part["name"], progress=progress)

        part["message_id"] = sent["message_id"]
        part["file_id"] = sent["file_id"]
        done_bytes += part["size"]
        _set(sent_bytes=done_bytes, pct=_pct(done_bytes, grand))
        # Record progress on disk immediately: a crash after this point must
        # not re-upload a part that is already in the channel.
        _save_local(manifest)

    _check_cancel()

    # The commit point. Parts above are inert until this lands, so a run that
    # dies mid-upload leaves orphans rather than a half-bundle restore trusts.
    _set(stage="Posting manifest", pct=97, detail="commit point")
    man_path = os.path.join(manifest["work_dir"], "manifest.json")
    publishable = {
        k: v for k, v in manifest.items() if k != "work_dir"
    }
    publishable["parts"] = [{k: v for k, v in p.items() if k != "local_path"}
                            for p in manifest["parts"]]
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(publishable, f, indent=2)

    c = manifest.get("counts", {})
    posted = tg.send_document(
        man_path, file_name=f"manifest-{manifest['seq']}.json",
        caption=(f"✅ VIOS bundle {manifest['seq']}\n"
                 f"schema v{manifest['schema']} · code {manifest['code_commit']}\n"
                 f"{total} part(s) · {_mb(grand)}\n"
                 f"posts {c.get('posts', '?')} · "
                 f"with file {c.get('posts_with_file', '?')}\n"
                 f"Restore reads this message first."))
    manifest["manifest_message_id"] = posted["message_id"]
    manifest["manifest_file_id"] = posted["file_id"]

    # Pinned, so restore finds the newest bundle in one getChat instead of
    # walking channel history — which the Bot API cannot do at all.
    _set(stage="Pinning manifest", pct=99)
    manifest["pinned"] = tg.pin_message(posted["message_id"])
    if not manifest["pinned"]:
        vios_log("manifest posted but pin failed — restore will need the "
                 "MTProto history scan to find it", "EXPORT", "WARN")
    return manifest


def _pct(done: int, total: int) -> int:
    """Build phase owns 0-50, upload owns 50-97. Monotonic and derived from
    bytes, so the bar stops moving exactly when the transfer does."""
    if total <= 0:
        return 50
    return 50 + min(47, int(47 * done / total))


def _mb(n: float) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / 1073741824:.2f} GB"
    if n >= 1024 * 1024:
        return f"{n / 1048576:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{int(n)} B"


def _save_local(manifest: dict) -> None:
    """Persist the in-progress manifest beside the parts it describes."""
    try:
        with open(os.path.join(manifest["work_dir"], LOCAL_MANIFEST),
                  "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except OSError as e:
        vios_log(f"could not write local manifest: {e}", "EXPORT", "WARN")


def _run(keep_local: bool, resume_seq: str | None = None) -> None:
    """Body of the export thread."""
    seq = resume_seq or time.strftime("%Y%m%d-%H%M%S")
    committed = False          # bundle on disk and self-consistent
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        _set(state="running", stage="Starting", pct=1, detail=f"bundle {seq}",
             started_at=time.time(), finished_at=None, bundle=None, error=None,
             log=[], sent_bytes=0, total_bytes=0, part_sent=0, part_total=0,
             rate_bps=0, eta_s=None, last_progress_at=time.time())

        manifest = _load_local(seq) if resume_seq else None
        if manifest:
            done = sum(1 for p in manifest["parts"] if p.get("message_id"))
            _set(stage="Resuming bundle", pct=50,
                 detail=f"{done}/{len(manifest['parts'])} part(s) already in "
                        f"the channel")
        else:
            manifest = _build_bundle(seq)
        committed = True

        absent = missing_telegram_secrets()
        if absent:
            # The bundle is on disk and valid; only the transport is missing.
            _save_local(manifest)
            _set(state="done", stage="Built locally", pct=100,
                 detail=f"Telegram disabled ({', '.join(absent)}) — bundle kept "
                        f"at {manifest['work_dir']}",
                 finished_at=time.time(),
                 bundle={k: v for k, v in manifest.items() if k != "parts"})
            vios_log(f"bundle {seq} built but not uploaded: "
                     f"missing {', '.join(absent)}", "EXPORT", "WARN")
            return

        # Reachability first. Without this the failure mode is a 403 after the
        # bundle has already been built and the first part pushed.
        import tg_transport as tg
        _set(stage="Checking Telegram", pct=51, detail="getMe · getChat")
        health = tg.probe()
        if not health["ok"]:
            raise RuntimeError(health["error"] or "Telegram is unreachable.")
        _set(stage="Telegram ready", pct=52,
             detail=f"@{health['bot']} → {health['channel']}")

        _save_local(manifest)
        manifest = _upload_bundle(manifest)

        work = manifest.pop("work_dir", None)
        if not keep_local and work and os.path.isdir(work):
            # Telegram holds it now, and OUTPUT has a 19.5 GB quota to respect.
            shutil.rmtree(work, ignore_errors=True)
        elif work:
            _save_local(dict(manifest, work_dir=work))

        for p in manifest["parts"]:
            p.pop("local_path", None)

        _set(state="done", stage="Uploaded", pct=100,
             detail=f"{len(manifest['parts'])} part(s) + manifest "
                    f"(message {manifest.get('manifest_message_id')})"
                    + ("" if manifest.get("pinned") else " · pin failed"),
             finished_at=time.time(), bundle=manifest,
             part_sent=0, part_total=0, eta_s=0)
        vios_log(f"bundle {seq} uploaded — manifest message "
                 f"{manifest.get('manifest_message_id')}", "EXPORT", "SUCCESS")

    except _Cancelled as e:
        # The parts already posted stay posted: they are inert without a
        # manifest, and keeping them lets a resume skip re-uploading them.
        work = _work_dir(seq)
        _set(state="cancelled", stage="Cancelled", finished_at=time.time(),
             detail=f"{e} Parts already uploaded are kept — start another "
                    f"export to resume bundle {seq}.")
        vios_log(f"bundle {seq} cancelled by operator", "EXPORT", "WARN")
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
    finally:
        _cancel.clear()


def _load_local(seq: str) -> dict | None:
    """Re-read a half-uploaded bundle so a retry resumes rather than rebuilds.

    Every part it names must still be on disk; a bundle whose files were
    cleaned up cannot be resumed and rebuilding is the honest answer.
    """
    path = os.path.join(_work_dir(seq), LOCAL_MANIFEST)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    for p in manifest.get("parts", []):
        if not p.get("message_id") and not os.path.exists(p.get("local_path", "")):
            return None
    manifest["work_dir"] = _work_dir(seq)
    return manifest


def resumable_bundle() -> str | None:
    """Newest bundle on disk that was built but never committed to the channel.

    A bundle with a manifest_message_id is finished; anything else is an
    interrupted upload the panel can offer to resume.
    """
    for b in list_local_bundles():
        seq = b["name"].replace(f"bundle-v{BUNDLE_SCHEMA}-", "")
        man = _load_local(seq)
        if man and not man.get("manifest_message_id"):
            return seq
    return None


def start_export(keep_local: bool = False, resume: bool = True) -> dict:
    """Kick off an export. Returns immediately; poll export_status()."""
    with _lock:
        if _job["state"] == "running":
            return {"ok": False, "error": "An export is already running."}
    try:
        import db_restore
        if db_restore.is_running():
            return {"ok": False,
                    "error": "A restore is running — wait for it to finish."}
    except Exception:
        pass
    _cancel.clear()
    seq = resumable_bundle() if resume else None
    t = threading.Thread(target=_run, args=(keep_local, seq),
                         name="vios-db-export", daemon=True)
    t.start()
    return {"ok": True, "resumed": seq}


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
