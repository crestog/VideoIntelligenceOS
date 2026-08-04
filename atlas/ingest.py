"""
Channel scan and bundle import.

*"Automatically download or whatever it do all the database files uploaded in
channel, by scanning entire channel."*

The channel is the only durable tier — Kaggle's disk is wiped between sessions,
so every bundle the harvester ever exported lives there and nowhere else. This
module turns that channel back into a database.

Three things make it awkward, and each shapes the design:

**A bundle is a set of messages, not a file.** The HTTP Bot API refuses to
download anything over 20 MB, so the exporter splits at 18 MB and posts each
part as its own message. A manifest message, posted last, lists every part with
its message_id, its file_id and its SHA-256. The manifest is the commit point:
parts without one are debris from a run that died, and are ignored here.

**Bots cannot list history.** `messages.getHistory` returns BOT_METHOD_INVALID.
So the scan cannot ask "what is in this channel" — it has to walk message ids
backwards from the newest and look at each one. `tgchannel.newest_message_id()`
finds the top by posting a throwaway message and reading its id.

**The newest bundle is the one you want first.** A full backwards walk over a
channel with thousands of messages takes a while, and the whole point is a site
that is usable immediately. So there are two paths: the pinned manifest, which
the exporter pins on every successful export and which lands in one API call,
and the full scan, which runs afterwards to pick up history. The site is live
after the first one.

Import is a merge, not a restore. Every bundle is a full snapshot of the machine
at one moment; importing all of them into one database means later snapshots
overwrite rows they share and contribute rows they added, and re-importing one
changes nothing. That is what makes "scan the entire channel" safe to run twice.
"""

import json
import os
import sqlite3
import threading
import time

from . import config, pgdump, reflect, tgchannel
from .tgchannel import log

# ── Progress, readable from the API while a scan is running ───────────────
_LOCK = threading.RLock()
_STATE = {
    "phase": "idle",          # idle | probing | fast | scanning | importing | done | error
    "detail": "",
    "scanned": 0,
    "scan_total": 0,
    "found": 0,
    "imported": 0,
    "skipped": 0,
    "failed": 0,
    "bytes_done": 0,
    "bytes_total": 0,
    "current": "",
    "started_at": 0.0,
    "finished_at": 0.0,
    "error": "",
    "running": False,
}


def _set(**kw):
    with _LOCK:
        _STATE.update(kw)


def status() -> dict:
    with _LOCK:
        s = dict(_STATE)
    s["elapsed"] = round(
        (s["finished_at"] or time.time()) - s["started_at"], 1
    ) if s["started_at"] else 0.0
    s["log"] = tgchannel.recent_log(40)
    return s


# ══════════════════════════════════════════════════════════════════════════
# ATLAS'S OWN BOOKKEEPING
# ══════════════════════════════════════════════════════════════════════════
_META_DDL = (
    "CREATE TABLE IF NOT EXISTS atlas_meta "
    "(key TEXT PRIMARY KEY, value TEXT)",

    # One row per bundle ever imported, so a re-scan can skip work it has
    # already done and the UI can show where the data came from.
    "CREATE TABLE IF NOT EXISTS bundles "
    "(seq TEXT PRIMARY KEY, manifest_id INTEGER, schema INTEGER, "
    "created_at TEXT, code_commit TEXT, parts INTEGER, bytes INTEGER, "
    "counts TEXT, imported_at REAL, status TEXT, note TEXT)",
)


def connect(path: str = None) -> sqlite3.Connection:
    """Open atlas.db with the pragmas this workload wants.

    WAL because the indexer writes while the server reads; a 64 MB page cache
    because search touches the moment table constantly and Kaggle has the RAM;
    `foreign_keys` deliberately off because imported dumps arrive in whatever
    order the channel held them.
    """
    conn = sqlite3.connect(path or config.DB_PATH, timeout=60.0,
                           check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def ensure_meta(conn: sqlite3.Connection) -> None:
    for ddl in _META_DDL:
        conn.execute(ddl)
    conn.commit()


def meta_get(conn: sqlite3.Connection, key: str, default=None):
    try:
        row = conn.execute("SELECT value FROM atlas_meta WHERE key=?",
                           (key,)).fetchone()
    except sqlite3.Error:
        return default
    return row[0] if row else default


def meta_set(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute("INSERT OR REPLACE INTO atlas_meta(key, value) VALUES (?,?)",
                 (key, str(value)))
    conn.commit()


def imported_seqs(conn: sqlite3.Connection) -> set:
    try:
        return {r[0] for r in conn.execute(
            "SELECT seq FROM bundles WHERE status='ok'")}
    except sqlite3.Error:
        return set()


def bundle_rows(conn: sqlite3.Connection) -> list:
    """What the Sources tab shows: every bundle, newest first."""
    try:
        cur = conn.execute(
            "SELECT seq, manifest_id, schema, created_at, code_commit, parts, "
            "bytes, counts, imported_at, status, note FROM bundles "
            "ORDER BY COALESCE(manifest_id, 0) DESC")
    except sqlite3.Error:
        return []
    out = []
    for r in cur.fetchall():
        try:
            counts = json.loads(r[7] or "{}")
        except (ValueError, TypeError):
            counts = {}
        out.append({"seq": r[0], "manifest_id": r[1], "schema": r[2],
                    "created_at": r[3], "code_commit": r[4], "parts": r[5],
                    "bytes": r[6], "counts": counts, "imported_at": r[8],
                    "status": r[9], "note": r[10]})
    return out


# ══════════════════════════════════════════════════════════════════════════
# DOWNLOAD AND REASSEMBLE
# ══════════════════════════════════════════════════════════════════════════
def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_part(part: dict, dest: str) -> str:
    """Download one part, preferring the cheap transport. Returns "" on success.

    HTTP first because it needs no session file and no login — the bot token is
    enough, and an 18 MB part is inside the 20 MB ceiling by design. MTProto is
    the fallback for parts from a v1 bundle (no file_id recorded) and for the
    case where the file_id has expired.
    """
    if os.path.exists(dest) and part.get("sha256"):
        if os.path.getsize(dest) == part.get("size") and \
                _sha256(dest) == part["sha256"]:
            return ""                      # already here from an earlier run

    file_id = part.get("file_id")
    if file_id:
        try:
            if tgchannel.http_download(file_id, dest):
                return ""
        except Exception as e:
            log(f"part {part.get('name')} — HTTP failed ({e}), trying MTProto")

    mid = part.get("message_id")
    if not mid:
        return "no file_id and no message_id"
    if not tgchannel.mtproto_ready():
        return f"HTTP unavailable and MTProto not logged in ({tgchannel.mtproto_error()})"
    try:
        got = tgchannel.download_by_id(mid, dest)
        return "" if got else "download returned nothing"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _join(parts: list, out_path: str) -> None:
    """Concatenate .partNNN files in index order."""
    with open(out_path, "wb") as out:
        for p in sorted(parts, key=lambda x: x.get("part_index", 0)):
            with open(p["local_path"], "rb") as fin:
                while True:
                    block = fin.read(4 * 1024 * 1024)
                    if not block:
                        break
                    out.write(block)


def _decompress(src: str, dst: str) -> str:
    """zstd → plain. Returns "" on success.

    Two paths because Kaggle's image is not guaranteed to carry the Python
    binding, and installing one mid-boot is a worse failure than shelling out
    to the `zstd` binary that the image does ship.
    """
    try:
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            dctx.copy_stream(fin, fout, read_size=1 << 20, write_size=1 << 20)
        return ""
    except ImportError:
        pass
    except Exception as e:
        return f"zstandard failed: {type(e).__name__}: {e}"

    import shutil
    import subprocess
    exe = shutil.which("zstd")
    if not exe:
        return "no zstd: neither the python module nor the binary is available"
    try:
        r = subprocess.run([exe, "-d", "-f", "-o", dst, src],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            return f"zstd exit {r.returncode}: {(r.stderr or '')[:200]}"
        return ""
    except (subprocess.SubprocessError, OSError) as e:
        return f"zstd: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════
# IMPORT ONE BUNDLE
# ══════════════════════════════════════════════════════════════════════════
def _import_sqlite(payload: str, conn: sqlite3.Connection) -> dict:
    """Merge every table of a decompressed lake.db snapshot into atlas.db.

    ATTACH rather than row-by-row Python: SQLite copies between two attached
    databases inside its own engine, which is both faster and shorter.

    The merge is per-table INSERT OR REPLACE when the source table has a real
    primary key, INSERT OR IGNORE when it does not. Neither can double a row on
    a second import of the same bundle, which is the property that matters —
    `scan the entire channel` is expected to be run repeatedly.
    """
    counts = {}
    conn.execute("ATTACH DATABASE ? AS src", (payload,))
    try:
        src_tables = [r[0] for r in conn.execute(
            "SELECT name FROM src.sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND COALESCE(sql,'') "
            "NOT LIKE '%VIRTUAL TABLE%'")]

        for t in src_tables:
            if reflect._FTS_SHADOW.search(t.lower()):
                continue
            src_cols = [r[1] for r in conn.execute(
                f'PRAGMA src.table_info("{t}")')]
            if not src_cols:
                continue
            has_pk = any(r[5] for r in conn.execute(
                f'PRAGMA src.table_info("{t}")'))

            if not _table_exists(conn, t):
                ddl = conn.execute(
                    "SELECT sql FROM src.sqlite_master WHERE type='table' "
                    "AND name=?", (t,)).fetchone()
                if not ddl or not ddl[0]:
                    continue
                conn.execute(ddl[0])
            else:
                _add_missing(conn, t, payload, t)

            dst_cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')}
            shared = [c for c in src_cols if c in dst_cols]
            if not shared:
                continue
            cols_sql = ", ".join(f'"{c}"' for c in shared)
            verb = "INSERT OR REPLACE" if has_pk else "INSERT OR IGNORE"
            before = _count(conn, t)
            conn.execute(f'{verb} INTO main."{t}" ({cols_sql}) '
                         f'SELECT {cols_sql} FROM src."{t}"')
            counts[t] = _count(conn, t) - before
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE src")
    return counts


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone() is not None


def _count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return conn.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()[0]
    except sqlite3.Error:
        return 0


def _add_missing(conn: sqlite3.Connection, dst: str, _payload: str,
                 src: str) -> None:
    """Widen a destination table to fit a newer snapshot's extra columns."""
    have = {r[1] for r in conn.execute(f'PRAGMA table_info("{dst}")')}
    for r in conn.execute(f'PRAGMA src.table_info("{src}")'):
        name, decl = r[1], (r[2] or "TEXT")
        if name in have:
            continue
        try:
            conn.execute(f'ALTER TABLE main."{dst}" ADD COLUMN "{name}" {decl}')
        except sqlite3.Error:
            pass


def import_bundle(manifest: dict, manifest_id: int,
                  conn: sqlite3.Connection) -> dict:
    """Download, verify, decompress and merge one bundle. Returns a result dict.

    Verification is not optional. A truncated part concatenated with its
    siblings produces a file that zstd will refuse or, worse, that SQLite will
    open with a corrupt page. Checking each part's SHA-256 against the manifest
    catches it before either happens.
    """
    seq = str(manifest.get("seq") or manifest_id)
    work = os.path.join(config.BUNDLE_DIR, f"seq-{seq}")
    os.makedirs(work, exist_ok=True)

    parts = manifest.get("parts") or []
    if not parts:
        return {"ok": False, "seq": seq, "note": "manifest lists no parts"}

    total_bytes = sum(int(p.get("size") or 0) for p in parts)
    _set(current=f"bundle {seq}", bytes_total=total_bytes, bytes_done=0)

    # ── fetch + verify every part ──
    by_file = {}
    for n, part in enumerate(parts, 1):
        name = part.get("name") or f"part{n:03d}"
        dest = os.path.join(work, name)
        _set(detail=f"bundle {seq} — part {n}/{len(parts)} · {name}")

        err = _fetch_part(part, dest)
        if err:
            return {"ok": False, "seq": seq, "note": f"part {name}: {err}"}

        want = part.get("sha256")
        if want:
            got = _sha256(dest)
            if got != want:
                os.remove(dest)
                return {"ok": False, "seq": seq,
                        "note": f"part {name}: checksum mismatch "
                                f"({got[:12]}… vs {want[:12]}…)"}

        entry = dict(part)
        entry["local_path"] = dest
        by_file.setdefault(part.get("file") or "index.sqlite.zst",
                           []).append(entry)
        with _LOCK:
            _STATE["bytes_done"] += int(part.get("size") or 0)

    # ── join, decompress, merge ──
    merged = {}
    for logical, group in by_file.items():
        _set(detail=f"bundle {seq} — assembling {logical}")
        joined = os.path.join(work, logical)
        if len(group) == 1 and group[0]["local_path"] == joined:
            pass
        else:
            _join(group, joined)

        plain = joined[:-4] if joined.endswith(".zst") else joined + ".plain"
        err = _decompress(joined, plain)
        if err:
            return {"ok": False, "seq": seq, "note": f"{logical}: {err}"}

        _set(detail=f"bundle {seq} — merging {os.path.basename(plain)}")
        try:
            if plain.endswith(".sql"):
                got = pgdump.load_dump(plain, conn, prefix="omni_")
            else:
                got = _import_sqlite(plain, conn)
            for k, v in got.items():
                merged[k] = merged.get(k, 0) + v
        except Exception as e:
            return {"ok": False, "seq": seq,
                    "note": f"{logical}: {type(e).__name__}: {e}"}
        finally:
            # The payload is reconstructible from the channel; the parts are
            # what cost bandwidth. Drop the big intermediates either way —
            # /kaggle/temp is not large enough to hold every bundle twice.
            for f in (joined, plain):
                try:
                    if os.path.exists(f) and os.path.getsize(f) > 8 * 1024 * 1024:
                        os.remove(f)
                except OSError:
                    pass

    conn.execute(
        "INSERT OR REPLACE INTO bundles(seq, manifest_id, schema, created_at, "
        "code_commit, parts, bytes, counts, imported_at, status, note) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (seq, manifest_id, manifest.get("schema"), manifest.get("created_at"),
         manifest.get("code_commit"), len(parts), total_bytes,
         json.dumps(manifest.get("counts") or {}), time.time(), "ok", ""))
    conn.commit()
    delta = ", ".join(f"{k} +{v}" for k, v in sorted(merged.items()) if v)
    log(f"bundle {seq} imported — {delta or 'no new rows'}")
    return {"ok": True, "seq": seq, "rows": merged}


def _record_failure(conn: sqlite3.Connection, seq: str, manifest_id: int,
                    note: str) -> None:
    try:
        conn.execute(
            "INSERT OR REPLACE INTO bundles(seq, manifest_id, parts, bytes, "
            "counts, imported_at, status, note) VALUES (?,?,0,0,'{}',?,?,?)",
            (seq, manifest_id, time.time(), "failed", note[:400]))
        conn.commit()
    except sqlite3.Error:
        pass


# ══════════════════════════════════════════════════════════════════════════
# THE SCAN
# ══════════════════════════════════════════════════════════════════════════
def _manifest_from_message(msg, work_dir: str):
    """Return a manifest dict if this message is one, else None."""
    info = tgchannel.message_document(msg)
    if not info or not tgchannel.looks_like_manifest(info):
        return None
    try:
        return tgchannel.read_manifest_document(info, work_dir)
    except Exception as e:
        log(f"message {info.get('message_id')} looked like a manifest "
            f"but would not parse ({e})")
        return None


def scan_and_import(full: bool = True, max_messages: int = 0,
                    on_bundle=None) -> dict:
    """Find every bundle in the channel and import the ones not yet held.

    Two phases, in this order on purpose:

      1. The pinned manifest. The exporter pins each successful export, so this
         is the newest bundle and it costs one API call. The site becomes
         useful here — usually within a few seconds of boot.
      2. The backwards walk, if `full`. Batches of ids from the newest down to
         1, looking for manifests. This is the "scan entire channel" part and
         is slow by nature; it runs after the site is already up.

    Idempotent: bundles already recorded as imported are skipped without being
    downloaded, so calling this on every boot costs one `getChat` plus the walk.
    """
    if _STATE["running"]:
        return {"ok": False, "note": "a scan is already running"}

    _set(phase="probing", running=True, error="", started_at=time.time(),
         finished_at=0.0, scanned=0, found=0, imported=0, skipped=0, failed=0,
         detail="checking channel access", current="")

    conn = connect()
    ensure_meta(conn)
    seen = imported_seqs(conn)
    work = config.BUNDLE_DIR

    try:
        probe = tgchannel.probe()
        if not probe.get("ok"):
            raise RuntimeError(probe.get("error") or "channel unreachable")
        log(f"channel ok — bot @{probe.get('bot')} on "
            f"{probe.get('channel') or config.CHANNEL_ID}")

        found_ids = []

        # ── phase 1: the pinned manifest ──
        _set(phase="fast", detail="reading pinned manifest")
        pinned = probe.get("pinned_message_id")
        if pinned:
            # The Bot API's getChat already returned the pinned message inline,
            # and a manifest is a few KB — well inside the 20 MB getFile cap. So
            # this whole path runs over plain HTTPS with no session and no
            # login, which is why the site can be useful seconds after boot.
            # MTProto is only consulted if that shape carried no document.
            msg = tgchannel.pinned_message()
            if not tgchannel.looks_like_manifest(tgchannel.message_document(msg)):
                if tgchannel.mtproto_ready():
                    got = tgchannel.get_messages([pinned])
                    msg = got[0] if got else msg
            man = None
            try:
                man = _manifest_from_message(msg, work)
            except Exception as e:
                log(f"pinned manifest unreadable ({e}) — falling back to scan")
            if man:
                found_ids.append(pinned)
                seq = str(man.get("seq") or pinned)
                if seq in seen:
                    _set(skipped=_STATE["skipped"] + 1)
                    log(f"pinned bundle {seq} already imported")
                else:
                    _set(phase="importing", found=_STATE["found"] + 1)
                    res = import_bundle(man, pinned, conn)
                    if res.get("ok"):
                        _set(imported=_STATE["imported"] + 1)
                        seen.add(res["seq"])
                        if on_bundle:
                            on_bundle(conn, res)
                    else:
                        _set(failed=_STATE["failed"] + 1)
                        _record_failure(conn, seq, pinned, res.get("note", ""))
                        log(f"pinned bundle {seq} failed — {res.get('note')}")

        if not full:
            _set(phase="done", running=False, finished_at=time.time(),
                 detail="pinned bundle only")
            return status()

        # ── phase 2: the backwards walk ──
        if not tgchannel.mtproto_ready():
            note = ("full channel scan needs MTProto (bots cannot list "
                    f"history) — {tgchannel.mtproto_error()}")
            log(note)
            _set(phase="done", running=False, finished_at=time.time(),
                 detail=note)
            return status()

        head = tgchannel.newest_message_id()
        if not head:
            raise RuntimeError("could not determine the newest message id")
        floor = 1
        if max_messages:
            floor = max(1, head - int(max_messages) + 1)
        _set(phase="scanning", scan_total=head - floor + 1,
             detail=f"walking messages {head} → {floor}")
        log(f"scanning {head - floor + 1} message ids for manifests")

        BATCH = 190          # get_messages tolerates 200; leave headroom
        mid = head
        while mid >= floor:
            ids = list(range(max(floor, mid - BATCH + 1), mid + 1))
            try:
                msgs = tgchannel.get_messages(ids)
            except Exception as e:
                log(f"batch {ids[0]}–{ids[-1]} failed ({e}) — continuing")
                msgs = []
            with _LOCK:
                _STATE["scanned"] += len(ids)

            for msg in reversed(msgs or []):
                info = tgchannel.message_document(msg)
                if not info or info.get("message_id") in found_ids:
                    continue
                man = _manifest_from_message(msg, work)
                if not man:
                    continue
                m_id = info.get("message_id")
                found_ids.append(m_id)
                seq = str(man.get("seq") or m_id)
                _set(found=_STATE["found"] + 1)
                if seq in seen:
                    _set(skipped=_STATE["skipped"] + 1)
                    continue
                _set(phase="importing")
                res = import_bundle(man, m_id, conn)
                if res.get("ok"):
                    _set(imported=_STATE["imported"] + 1)
                    seen.add(res["seq"])
                    if on_bundle:
                        on_bundle(conn, res)
                else:
                    _set(failed=_STATE["failed"] + 1)
                    _record_failure(conn, seq, m_id, res.get("note", ""))
                    log(f"bundle {seq} failed — {res.get('note')}")
                _set(phase="scanning")

            mid -= BATCH

        meta_set(conn, "last_scan", time.time())
        meta_set(conn, "last_scan_head", head)
        _set(phase="done", running=False, finished_at=time.time(),
             current="", detail=f"{_STATE['found']} bundle(s) in channel · "
                                f"{_STATE['imported']} imported · "
                                f"{_STATE['skipped']} already held")
        log(f"scan complete — {_STATE['found']} manifest(s), "
            f"{_STATE['imported']} imported, {_STATE['failed']} failed")
    except Exception as e:
        _set(phase="error", running=False, finished_at=time.time(),
             error=f"{type(e).__name__}: {e}",
             detail="scan stopped — see error")
        log(f"scan aborted — {type(e).__name__}: {e}")
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    return status()


def scan_in_background(full: bool = True, max_messages: int = 0,
                       on_bundle=None) -> bool:
    if _STATE["running"]:
        return False
    t = threading.Thread(target=scan_and_import,
                         args=(full, max_messages, on_bundle),
                         name="atlas-scan", daemon=True)
    t.start()
    return True
