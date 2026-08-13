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

**Two lanes post to this channel, and only one of them posts bundles.** The
harvester writes bundles: snapshots of the capture ledger, manifests, parts.
The GPU plane writes *shards* — `vios-evidence-*.jsonl.gz`, one gzipped JSONL
per batch of claims, each complete on its own, none of them pinned and none of
them mentioned in any manifest. Nothing about that format resembles a bundle, so
a reader that knew only manifests walked past every transcript, every caption
and every OCR line the models ever produced. `import_shard` is the other half,
and the walk offers each message to both readers.

Shards are additive where bundles are snapshots, so their merge rule is the
mirror image: never overwrite, only fill. See `_dedup_columns` and `_enrich`.
"""

import json
import os
import re
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
    "assets": 0,              # per-video asset manifests read into `parts`
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


# ══════════════════════════════════════════════════════════════════════════
# IMPORT ONE EVIDENCE SHARD
# ══════════════════════════════════════════════════════════════════════════
# A shard is the other lane's format, and it is nothing like a bundle: no
# manifest, no parts, no SQLite. One gzipped JSONL file, first line a header,
# every line after it a row tagged with the table it came from. It is complete
# on its own and it is *additive* — bundles are snapshots that overwrite, shards
# are batches that accumulate.
#
# Atlas replays them without knowing what the GPU plane's schema is, because it
# cannot: that schema is defined in another program, on another branch of the
# tree, and changes when a pass is added. So the table shapes are inferred from
# the rows themselves. What Atlas *does* have to get right is the declared
# column types — reflect.py reads them to decide what is a timestamp and what is
# searchable text, and a table built with every column untyped reflects as one
# where nothing is a number and everything is prose.
_SHARD_MAGIC = "vios-evidence-shard"

# Long unbroken hex or base64: an embedding, a thumbnail, a signature. Atlas
# cannot read it, cannot search it and has no code that consumes it — the
# vectors it searches with are its own, from its own encoder, in its own space.
# Kept, it is the largest thing in the database and the only useless one.
_OPAQUE = re.compile(r"^[A-Za-z0-9+/=]{256,}$")


def _sql_type(values: list) -> str:
    """The narrowest SQLite type that holds every value seen for a column, or
    "" when the column was empty in this shard.

    Not cosmetic. `_is_numeric` and `_is_texty` in reflect.py read the declared
    type and nothing else, so a `t0` column created untyped is not a timestamp,
    a claim's text is not searchable, and the whole shard lands in the database
    correct and completely invisible.

    The empty case is why this returns "" rather than defaulting to TEXT. A
    video is written the moment it is downloaded, before anything has measured
    it, so the first shard carries `duration: null` for every row. Typed TEXT on
    that evidence, the column then stores the 30.0 a later shard measured as the
    *string* `"30.0"` — which is not a duration, does not compare, and turns the
    moment ribbon into a blank strip. The caller leaves such a column out
    entirely, so the first shard that knows something creates it correctly.
    """
    seen = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            seen.add(int)
        elif isinstance(v, (int, float)):
            seen.add(type(v))
        else:
            seen.add(str)
    if not seen:
        return ""
    if seen == {int}:
        return "INTEGER"
    if seen <= {int, float}:
        return "REAL"
    return "TEXT"


def _is_opaque(values: list) -> bool:
    """Is this column a payload rather than a fact? Judged on the values, since
    the name is whatever the other program chose to call it."""
    long_blobs = 0
    real = 0
    for v in values:
        if v is None or v == "":
            continue
        real += 1
        if isinstance(v, str) and _OPAQUE.match(v):
            long_blobs += 1
    return real > 0 and long_blobs == real


def _dedup_columns(order: list, rows: list) -> list:
    """The smallest column set whose values identify a row, or [].

    This is how a re-imported shard stays a no-op at the row level rather than
    only at the file level. There is no primary key to copy — JSONL carries
    values, not constraints — so it is measured instead: try each identifier-
    shaped column alone, then identifier-led pairs. Every table in the evidence
    schema lands on its real key this way (`uid` for claims, `video_key` for
    videos, `video_key + idx` for shots) without Atlas naming any of them.

    **A key must contain an identifier.** Uniqueness alone is not evidence of a
    key, because it is measured on one batch: a shard holding one video's two
    shots has a unique `idx` — 0 and 1 — and an index built on that silently
    swallows every other video's opening shot for the rest of the archive. A
    column that is unique only because the sample was narrow is a coincidence;
    requiring the identifier is what tells the two apart.

    Empty when nothing qualifies, which is not a failure — the shard-level skip
    still holds, and a table with genuine duplicate rows should keep them.
    """
    scalar = [c for c in order
              if all(v is None or isinstance(v, (int, float, bool))
                     or (isinstance(v, str) and len(v) <= 96)
                     for v in (r.get(c) for r in rows))]
    ident = [c for c in scalar
             if reflect._norm(c).endswith(("id", "key", "uid", "hash"))]
    if not ident:
        return []
    rest = [c for c in scalar if c not in ident]

    def unique(cols):
        keys = set()
        for r in rows:
            k = tuple(_scalar(r.get(c)) for c in cols)
            if any(v is None for v in k) or k in keys:
                return False
            keys.add(k)
        return True

    for c in ident:
        if unique([c]):
            return [c]
    for a in ident:
        for b in ident + rest:
            if b != a and unique([a, b]):
                return [a, b]
    return []


def read_shard(path: str) -> tuple:
    """Parse a shard file into (header, {table: [rows]}).

    A shard truncated by a session that died mid-upload is not an error — every
    line before the tear is still good evidence. Parsing stops at the first
    unreadable line and keeps what came before, the same way the process plane's
    own reader does.
    """
    import gzip                                       # noqa: PLC0415
    tables, torn = {}, 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        header = json.loads(fh.readline() or "{}")
        if header.get("_") != _SHARD_MAGIC:
            raise ValueError("not an evidence shard")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                torn = 1
                break
            t = rec.pop("t", "")
            if t and isinstance(rec, dict):
                tables.setdefault(str(t), []).append(rec)
    header["_torn"] = torn
    return header, tables


def _ensure_shard_table(conn: sqlite3.Connection, table: str,
                        types: dict, keys: list) -> list:
    """Create or widen the destination table. Returns the columns it will take.

    Widening rather than recreating matters because shards arrive in any order
    and a later pass adds columns: shard 4 may carry a `confidence` the schema
    did not have when shard 1 built the table.
    """
    if not _table_exists(conn, table):
        cols_sql = ", ".join(f'"{c}" {t}' for c, t in types.items())
        conn.execute(f'CREATE TABLE "{table}" ({cols_sql})')
        if keys:
            idx = ("ux_" + table + "_" + "_".join(keys))[:60]
            key_sql = ", ".join('"%s"' % k for k in keys)
            conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{idx}" '
                         f'ON "{table}" ({key_sql})')
        return list(types)

    have = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
    for c, t in types.items():
        if c not in have:
            try:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" {t}')
                have.add(c)
            except sqlite3.Error:
                pass
    return [c for c in types if c in have]


def import_shard(info: dict, conn: sqlite3.Connection, work_dir: str) -> dict:
    """Download one evidence shard and replay it into atlas.db."""
    seq = "shard:" + tgchannel.shard_seq(info)
    msg_id = info.get("message_id")
    dest = os.path.join(work_dir, f"shard-{msg_id}.jsonl.gz")
    _set(current=seq, detail=f"{seq} — downloading")

    if not tgchannel.fetch_document(info, dest):
        size = int(info.get("file_size") or 0)
        return {"ok": False, "seq": seq,
                "note": (f"could not download ({size / 1048576:.1f} MB — over "
                         f"the Bot API's 20 MB cap and MTProto is not "
                         f"available)" if size > config.HTTP_DOWNLOAD_LIMIT
                         else "could not download")}
    try:
        header, tables = read_shard(dest)
    except (OSError, ValueError) as exc:
        return {"ok": False, "seq": seq, "note": f"unreadable: {exc}"}

    rows_in = sum(len(v) for v in tables.values())
    _set(detail=f"{seq} — replaying {rows_in} row(s)")
    merged, dropped = {}, []

    for table, rows in tables.items():
        if not rows or reflect._norm(table) in reflect._ATLAS_OWN or \
                reflect._FTS_SHADOW.search(table.lower()):
            # Atlas's own tables are not a landing zone. A shard that happened
            # to carry a table called `moments` would otherwise overwrite the
            # index with the raw rows the index is built from.
            dropped.append(f"{table} (reserved name)")
            continue

        order, values = [], {}
        for r in rows:
            for c in r:
                if c not in values:
                    order.append(c)
                    values[c] = []
        for c in order:
            values[c] = [r.get(c) for r in rows]

        types = {}
        for c in order:
            if _is_opaque(values[c]):
                dropped.append(f"{table}.{c} (opaque)")
                continue
            t = _sql_type(values[c])
            if not t:
                # Empty in this shard, so there is nothing to learn about it and
                # nothing to store. Left out, not guessed at — see `_sql_type`.
                continue
            types[c] = t
        if not types:
            continue

        keys = _dedup_columns(list(types), rows)
        cols = _ensure_shard_table(conn, table, types, keys)
        if not cols:
            continue

        before = conn.total_changes
        col_sql = ", ".join('"%s"' % c for c in cols)
        conn.executemany(
            f'INSERT OR IGNORE INTO "{table}" ({col_sql}) '
            f'VALUES ({", ".join("?" * len(cols))})',
            [[_scalar(r.get(c)) for c in cols] for r in rows])
        added = conn.total_changes - before
        merged[table] = merged.get(table, 0) + added

        # Rows the unique index rejected are not necessarily redundant: a video
        # first written with no duration, then re-written once it was measured,
        # is the same row carrying more. Fill the blanks it can, overwrite
        # nothing. Only runs when there were duplicates, so a shard of all-new
        # claims — the common case, and the big one — pays nothing for it.
        if keys and added < len(rows):
            _enrich(conn, table, cols, keys, rows)

    conn.commit()
    # `counts` is what the shard *holds*, not what this run happened to add.
    # A bundle's row records its manifest's counts for the same reason: the
    # Sources tab is answering "what arrived in this file", and a shard imported
    # twice would otherwise rewrite its own history to say it contributed
    # nothing. What this run added is the delta, and it goes in the note.
    delta = ", ".join(f"{k} +{v}" for k, v in sorted(merged.items()) if v)
    held = {t: len(rows) for t, rows in tables.items()}
    conn.execute(
        "INSERT OR REPLACE INTO bundles(seq, manifest_id, schema, created_at, "
        "code_commit, parts, bytes, counts, imported_at, status, note) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (seq, msg_id, header.get("schema"), str(header.get("at") or ""),
         header.get("component") or "", 1,
         int(info.get("file_size") or os.path.getsize(dest)),
         json.dumps(held), time.time(), "ok",
         ("truncated — kept what was readable. "
          if header.get("_torn") else "")
         + (delta or "nothing new")
         + (" · dropped " + ", ".join(dropped[:6]) if dropped else ""))[:400])
    conn.commit()

    try:
        os.remove(dest)
    except OSError:
        pass

    log(f"{seq} imported — {delta or 'no new rows'}"
        + (f" · dropped {len(dropped)} opaque column(s)" if dropped else ""))
    return {"ok": True, "seq": seq, "rows": merged, "shard": True}


def _scalar(v):
    """JSON holds nested objects; SQLite does not. Anything structured is
    stored as the JSON it already was, which reflect.py then treats as text."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return int(v)
    return v


def _enrich(conn: sqlite3.Connection, table: str, cols: list, keys: list,
            rows: list) -> None:
    """Fill NULL columns of rows that already existed. Never overwrites."""
    fill = [c for c in cols if c not in keys]
    if not fill:
        return
    sets = ", ".join(f'"{c}" = COALESCE("{c}", ?)' for c in fill)
    where = " AND ".join(f'"{k}" = ?' for k in keys)
    conn.executemany(
        f'UPDATE "{table}" SET {sets} WHERE {where}',
        [[_scalar(r.get(c)) for c in fill] + [_scalar(r.get(k)) for k in keys]
         for r in rows])


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


_ASSET_SUFFIX = "-manifest.json"


def _looks_like_asset_manifest(info: dict) -> bool:
    """Cheap name test before spending a download.

    Deliberately narrow. The channel carries three kinds of `.json` — export
    bundle manifests, per-video records, and these — and a wrong guess costs a
    download plus a parse per message across a whole-channel walk. The
    `<key>-manifest.json` name is written by `assets.manifest_name` and nothing
    else produces it.
    """
    name = str(info.get("file_name") or "")
    return name.endswith(_ASSET_SUFFIX) and len(name) > len(_ASSET_SUFFIX)


def _asset_manifest(conn: sqlite3.Connection, info: dict,
                    work_dir: str) -> bool:
    """Import a per-video asset manifest into `parts`. True if rows landed.

    Failure is quiet and returns False: a manifest that will not parse costs
    that video its instant playback, and the byte-range path still plays it.
    Losing the scan over it would cost every video.
    """
    if not _looks_like_asset_manifest(info):
        return False
    dest = os.path.join(work_dir, f"asset-{info.get('message_id')}.json")
    try:
        if not tgchannel.fetch_document(info, dest):
            return False
        with open(dest, "r", encoding="utf-8", errors="replace") as f:
            man = json.load(f)
    except Exception as e:
        log(f"asset manifest {info.get('message_id')} would not read ({e})")
        return False
    finally:
        try:
            os.remove(dest)
        except OSError:
            pass
    if not isinstance(man, dict) or not man.get("key"):
        return False
    try:
        from . import index as index_mod
        index_mod.ensure_schema(conn)
        n = index_mod.record_parts(conn, man)
    except Exception as e:
        log(f"asset manifest for {man.get('key')} would not store ({e})")
        return False
    if n:
        # The canonical key, not the manifest's own spelling — this line is how
        # the operator checks that clips landed where `/api/clip` looks for them.
        log(f"asset set for {reflect.normalize_key(man['key'])} — "
            f"{len(man.get('chunks') or [])} clip(s)")
    return bool(n)


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
         assets=0, detail="checking channel access", current="")

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
                m_id = info.get("message_id")

                # Two lanes post to this one channel and their messages are
                # interleaved by time, so every message is offered to both
                # readers rather than the walk being run twice. A message can
                # only be one of them: a manifest is a `.json` naming parts, a
                # shard is a `.jsonl.gz` that is already the whole thing.
                if tgchannel.looks_like_shard(info):
                    found_ids.append(m_id)
                    seq = "shard:" + tgchannel.shard_seq(info)
                    _set(found=_STATE["found"] + 1)
                    if seq in seen:
                        _set(skipped=_STATE["skipped"] + 1)
                        continue
                    _set(phase="importing")
                    res = import_shard(info, conn, work)
                    if res.get("ok"):
                        _set(imported=_STATE["imported"] + 1)
                        seen.add(res["seq"])
                        if on_bundle:
                            on_bundle(conn, res)
                    else:
                        _set(failed=_STATE["failed"] + 1)
                        _record_failure(conn, seq, m_id, res.get("note", ""))
                        log(f"{seq} failed — {res.get('note')}")
                    _set(phase="scanning")
                    continue

                man = _manifest_from_message(msg, work)
                if not man:
                    # Not a bundle manifest. It may still be a per-video asset
                    # manifest — the index that makes clip playback possible —
                    # which lives in the same channel and is also a .json.
                    if _asset_manifest(conn, info, work):
                        found_ids.append(m_id)
                        with _LOCK:
                            _STATE["assets"] = _STATE.get("assets", 0) + 1
                    continue
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
             current="", detail=f"{_STATE['found']} bundle(s) and shard(s) "
                                f"in channel · {_STATE['imported']} imported · "
                                f"{_STATE['skipped']} already held · "
                                f"{_STATE['assets']} asset set(s)")
        log(f"scan complete — {_STATE['found']} bundle(s)/shard(s), "
            f"{_STATE['imported']} imported, {_STATE['failed']} failed, "
            f"{_STATE['assets']} asset set(s)")
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
