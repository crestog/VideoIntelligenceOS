"""
The moment index.

Search accuracy is decided here, not in the ranker. A ranker can only reorder
what the index gave it, so this file's job is to turn a pile of database rows
into passages that are actually retrievable.

Four decisions do most of the work:

**One table, every source.** Narratives live in Postgres, speech in one SQLite
table, OCR and object labels in another, captions in a third. Searching them
separately means merging incomparable scores later. They are copied into a
single `moments` table with a `source` tag, so ranking compares like with like
and the UI can show which kind of evidence matched.

**Short fragments are merged into passages.** A transcript row is often three
words — "yeah exactly that" — and both BM25 and a dense encoder do badly with
it: there is no context to weigh, and the vector lands nowhere useful. Adjacent
rows from the same video and source are greedily merged up to a target length,
which is why a query matches a sentence somebody said across two subtitle
segments. Long rows (a full narrative) are left alone, because they are already
passages.

**Duplicate text is collapsed.** The pipeline can emit the same narrative for a
window twice — the harvester's own schema notes call this out. Five identical
rows would win five ranks in the candidate list and crowd out real results, so
the same (video, source, text) is stored once.

**Every video gets a precomputed summary row.** `video_index` holds the title,
duration, caption, creator, poster path and per-source moment counts for each
video. Result cards need all of it, and doing those joins per query is the
difference between a 15 ms search and a 300 ms one.

**One video, one row, decided before anything is read.** Every key that arrives
here is resolved through `atlas.identity` first, and `identity.refresh` runs at
the top of `rebuild`. This is the rule that stops the archive counting the same
reel twice: the passages a video's transcript produced under `DZDNyKgv70R` and
the ones its frame notes produced under `38` land on one row, because the map
that says those are the same post is built before the first `SELECT`. The keys
this file writes come from `identity.canonical_keys` and nowhere else.

Nothing here names a source table. The list comes from `reflect.text_sources()`,
so a column added upstream shows up as searchable moments on the next build.
"""

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid

from . import config, identity, reflect
from .tgchannel import log

# ── Passage shaping ───────────────────────────────────────────────────────
# A merged passage aims for this many characters. 320 is a compromise found by
# what the retrievers want: bge-small truncates at 512 tokens (~2000 chars) so
# longer would still embed, but a passage that spans 40 seconds of video stops
# being a *moment* — you would jump to it and not see what matched.
TARGET_CHARS = 320
MAX_CHARS = 900
# Rows further apart than this are different moments even if both are short.
MERGE_GAP_S = 14.0
# A row longer than this is already a passage; never merge it into a neighbour.
STANDALONE_CHARS = 180
# A point-in-time row (a frame note) is given this much width so it can be
# played and so overlap logic has something to work with.
POINT_WIDTH_S = 2.5

_MOMENT_DDL = (
    "CREATE TABLE IF NOT EXISTS moments ("
    "  id INTEGER PRIMARY KEY,"
    "  video_key TEXT NOT NULL,"
    "  t_start REAL,"
    "  t_end REAL,"
    "  source TEXT,"
    "  src_table TEXT,"
    "  weight REAL,"
    "  text TEXT NOT NULL,"
    "  text_hash TEXT,"
    "  UNIQUE(video_key, source, text_hash))",

    "CREATE INDEX IF NOT EXISTS moments_by_video ON moments(video_key, t_start)",
    "CREATE INDEX IF NOT EXISTS moments_by_source ON moments(source)",

    # One row per video: everything a result card shows, precomputed.
    #
    # The identity columns at the end are not decoration. `shortcode` and `url`
    # are what makes a card provably about *one* Instagram post; `aliases` says
    # which other spellings of that post the archive has seen, so a card can
    # admit "also message 38" instead of quietly becoming a second video; and
    # `collections` is the many-to-many, flattened here for display only — the
    # queryable copy lives in `video_collection`.
    #
    # `messages` is the readable half of `aliases`. A card that wants to say
    # "also at message 10" should not have to guess which of `10`, `tg10`,
    # `msg_10` and `frames_10` was a message id; `[10, 40]` is the answer, and
    # `msg_id` alone cannot hold it because a reel uploaded twice sits at two.
    "CREATE TABLE IF NOT EXISTS video_index ("
    "  video_key TEXT PRIMARY KEY,"
    "  msg_id INTEGER,"
    "  title TEXT,"
    "  caption TEXT,"
    "  creator TEXT,"
    "  category TEXT,"
    "  duration REAL,"
    "  width INTEGER,"
    "  height INTEGER,"
    "  fps REAL,"
    "  size_mb REAL,"
    "  likes INTEGER,"
    "  created_at REAL,"
    "  local_path TEXT,"
    "  poster TEXT,"
    "  moment_count INTEGER,"
    "  sources TEXT,"
    "  has_speech INTEGER,"
    "  has_narrative INTEGER,"
    "  text_len INTEGER,"
    "  shortcode TEXT,"
    "  url TEXT,"
    "  aliases TEXT,"
    "  messages TEXT,"
    "  collections TEXT,"
    "  twin_of TEXT,"
    "  is_stub INTEGER)",

    "CREATE INDEX IF NOT EXISTS video_by_created ON video_index(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS video_by_moments ON video_index(moment_count DESC)",

    # One row per channel message that belongs to a video's *asset set*: the
    # 2-second clips, the metadata json, the manifest itself. Filled by
    # atlas.ingest while it walks the channel — the same walk that imports
    # bundles — so a video's clips are discoverable by message id without
    # opening a single Telegram session.
    #
    # `kind` is one of: video | clip | meta | frames | manifest | record.
    # A clip row carries its time range so playback can pick the exact
    # segment covering a moment. `file_id` is the cheap HTTP route and
    # `msg_id` the permanent MTProto one; Atlas tries file_id first and
    # falls back to msg_id exactly like every other fetch here.
    "CREATE TABLE IF NOT EXISTS parts ("
    "  video_key TEXT NOT NULL,"
    "  kind TEXT NOT NULL,"
    "  seq INTEGER,"
    "  msg_id INTEGER,"
    "  file_id TEXT,"
    "  name TEXT,"
    "  bytes INTEGER,"
    "  sha256 TEXT,"
    "  t_start REAL,"
    "  t_end REAL,"
    "  chunk_seconds REAL,"
    "  UNIQUE(msg_id))",

    "CREATE INDEX IF NOT EXISTS parts_by_video ON parts(video_key, kind, seq)",
    "CREATE INDEX IF NOT EXISTS parts_by_time ON parts(video_key, t_start)",
)

# External-content FTS5: the text lives in `moments` and is not duplicated
# here. Porter stemming so "running" finds "ran"; unicode61 with diacritic
# folding so "cafe" finds "café"; `detail=full` keeps phrase queries working.
_FTS_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS moments_fts USING fts5("
    "  text, content='moments', content_rowid='id',"
    "  tokenize=\"porter unicode61 remove_diacritics 2\")",
)

_LOCK = threading.RLock()

# Columns added to `video_index` after it shipped. `CREATE TABLE IF NOT EXISTS`
# is a no-op on an existing table, so a database built before identity existed
# would keep the old twenty columns and every write below would fail on the
# first unknown name. Listed here rather than inferred so the migration is a
# thing you can read.
_VIDEO_INDEX_ADDED = (
    ("shortcode", "TEXT"), ("url", "TEXT"), ("aliases", "TEXT"),
    ("messages", "TEXT"),
    ("collections", "TEXT"), ("twin_of", "TEXT"), ("is_stub", "INTEGER"),
)


def _add_columns(conn: sqlite3.Connection, table: str, columns) -> int:
    """Add missing columns to an existing table. Idempotent."""
    try:
        have = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
    except sqlite3.Error:
        return 0
    if not have:
        return 0
    added = 0
    for name, decl in columns:
        if name in have:
            continue
        try:
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {decl}')
            added += 1
        except sqlite3.Error as e:
            log(f"could not add {table}.{name}: {e}")
    return added


_STATE = {
    "phase": "idle",        # idle | reading | writing | fts | embedding | done | error
    "detail": "",
    "moments": 0,
    "videos": 0,
    "embedded": 0,
    "embed_total": 0,
    "started_at": 0.0,
    "finished_at": 0.0,
    "error": "",
    "running": False,
    "lexical_ready": False,
    "dense_ready": False,
}


def _set(**kw):
    with _LOCK:
        _STATE.update(kw)


def status() -> dict:
    with _LOCK:
        s = dict(_STATE)
    s["elapsed"] = round((s["finished_at"] or time.time()) - s["started_at"], 1) \
        if s["started_at"] else 0.0
    return s


# ══════════════════════════════════════════════════════════════════════════
# TEXT HYGIENE
# ══════════════════════════════════════════════════════════════════════════
_WS = re.compile(r"\s+")
_JSONISH = re.compile(r'^\s*[\[{]')


def clean_text(value) -> str:
    """Normalise one cell into something worth indexing.

    Object lists arrive as JSON — `["person","bicycle"]` — because that is how
    the CV worker stored them. Indexed raw, the brackets and quotes become
    tokens and a search for `person` competes with punctuation. Unwrapping them
    into words is the difference between object labels helping and hurting.

    Opaque handles are removed for the mirror-image reason. The legacy harvest
    wrote Telegram file_ids into `categories.name`, so a category passage reads
    `BAACAgUAAyEGAAMBCGCQ-wADEmp3oIU2fPqhaznC2nu0-W1-RulYAAK…, liked posts` — 80
    characters no query will ever contain, tokenised into a dozen nonsense terms
    that dilute every score in the table they came from. They are dropped rather
    than the whole cell, because the rest of that cell is real.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return ""
    s = str(value).strip()
    if not s:
        return ""

    if _JSONISH.match(s):
        try:
            obj = json.loads(s)
            s = _flatten_json(obj)
        except (ValueError, TypeError):
            pass

    if _OPAQUE.search(s):
        s = _OPAQUE.sub(" ", s)
        s = re.sub(r"\s*,\s*,+", ", ", s).strip(" ,")

    s = _WS.sub(" ", s).strip()
    if len(s) > MAX_CHARS:
        s = s[:MAX_CHARS].rsplit(" ", 1)[0]
    # A cell holding one number or one token of punctuation is not content.
    if len(s) < 2 or s.isdigit():
        return ""
    return s


# A base64-ish run long enough, and mixed enough, to be a machine handle: 24+
# characters from the base64url alphabet carrying at least one digit and one
# capital. No word, name or sentence reaches that shape.
_OPAQUE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]{24,})"
    r"(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*[A-Z])[A-Za-z0-9_-]{24,}")


def _flatten_json(obj, depth: int = 0) -> str:
    """Turn nested JSON into a readable phrase, keeping labels and dropping
    scores. `[{"label":"dog","conf":0.9}]` becomes `dog`."""
    if depth > 4:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return ""
    if isinstance(obj, list):
        return ", ".join(p for p in (_flatten_json(o, depth + 1)
                                     for o in obj) if p)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            if str(k).lower() in ("conf", "confidence", "score", "prob", "id",
                                  "bbox", "box", "xyxy", "index", "idx"):
                continue
            piece = _flatten_json(v, depth + 1)
            if piece:
                parts.append(piece)
        return ", ".join(parts)
    return ""


def _hash(text: str) -> str:
    return hashlib.sha1(text.lower().encode("utf-8", "replace")).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════
# PASSAGE BUILDING
# ══════════════════════════════════════════════════════════════════════════
def _as_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def build_passages(rows: list) -> list:
    """Merge short adjacent rows into passages. Returns [(t0, t1, text)].

    `rows` is [(t_start, t_end, text)] for one video and one source, in any
    order. Rows with no timestamp keep None and are emitted as-is: a caption
    describes the whole video, so giving it a fake position would put a false
    marker on the timeline.
    """
    timed, untimed = [], []
    for t0, t1, text in rows:
        if not text:
            continue
        a, b = _as_float(t0), _as_float(t1)
        if a is None:
            untimed.append(text)
        else:
            if b is None or b <= a:
                b = a + POINT_WIDTH_S
            timed.append((a, b, text))

    out = []

    timed.sort(key=lambda r: (r[0], r[1]))
    buf = []          # [(a, b, text)] pending merge
    buf_chars = 0

    def flush():
        nonlocal buf, buf_chars
        if not buf:
            return
        joined = " ".join(t for _, _, t in buf).strip()
        if joined:
            out.append((buf[0][0], max(b for _, b, _ in buf), joined))
        buf, buf_chars = [], 0

    for a, b, text in timed:
        long_enough = len(text) >= STANDALONE_CHARS
        if long_enough:
            # Already a passage. Flush whatever was accumulating and emit alone.
            flush()
            out.append((a, b, text))
            continue
        if buf:
            gap = a - buf[-1][1]
            if gap > MERGE_GAP_S or buf_chars + len(text) > MAX_CHARS:
                flush()
        buf.append((a, b, text))
        buf_chars += len(text) + 1
        if buf_chars >= TARGET_CHARS:
            flush()
    flush()

    # Untimed text for a video is one passage per distinct string; merging
    # captions from different rows would invent sentences nobody wrote.
    #
    # Emitted last, and the same string is dropped outright when a placed passage
    # already carries it. Both halves of that matter because `moments` is
    # `UNIQUE(video_key, source, text_hash)` and the writer is `INSERT OR IGNORE`,
    # so of two identical texts the *first* one inserted is the one that survives
    # — and while this loop ran first, the survivor was always the copy that
    # cannot seek. It is not a hypothetical: a shot-scoped motion reading and the
    # whole-reel reading of the same pass both say `gentle`, one of them at
    # 0.0–95.0s and one of them nowhere, and the timeline got the second.
    placed = {text for _, _, text in out}
    for text in untimed:
        if text not in placed:
            out.append((None, None, text))

    return out


# ══════════════════════════════════════════════════════════════════════════
# THE BUILD
# ══════════════════════════════════════════════════════════════════════════
def ensure_schema(conn: sqlite3.Connection) -> bool:
    """Create the moment tables. Returns True if FTS5 is usable."""
    for ddl in _MOMENT_DDL:
        conn.execute(ddl)
    _add_columns(conn, "video_index", _VIDEO_INDEX_ADDED)
    identity.ensure(conn)
    # `rebuild` records its fingerprint in atlas_meta on the last four
    # statements it runs. That table belongs to the ingest path, so a database
    # that reached the indexer without going through a bundle import — a folder
    # adopted locally, a shard replayed straight in — had every passage built
    # and then lost the lot to "no such table: atlas_meta" at the finish line.
    from .ingest import ensure_meta          # noqa: PLC0415  (cycle at import)
    ensure_meta(conn)
    fts = True
    for ddl in _FTS_DDL:
        try:
            conn.execute(ddl)
        except sqlite3.Error as e:
            log(f"fts5 unavailable ({e}) — search will use LIKE, which is "
                f"slower and cannot rank")
            fts = False
    conn.commit()
    return fts


# ══════════════════════════════════════════════════════════════════════════
# ASSET PARTS — the clip index behind instant playback
# ══════════════════════════════════════════════════════════════════════════
def record_parts(conn: sqlite3.Connection, manifest: dict) -> int:
    """Store one video's asset manifest as `parts` rows. Returns rows written.

    `INSERT OR REPLACE` keyed on `msg_id`: a message is one asset, forever, so
    re-importing the same manifest is a no-op and a manifest that was rebuilt
    after a partial upload replaces the stale rows rather than doubling them.

    The key is normalised, because the producer and the reader do not spell it
    the same way. The manifest carries the capture *ledger's* key — `up_1234`
    for a hand-uploaded video — while `/api/clip` and `/api/clips` normalise
    whatever the page sends, and `video_index` is keyed off `posts.video_id`,
    the bare message id. Stored raw, every clip lands under a key nothing ever
    asks for and the routes keep answering 204 with a full `parts` table.
    `UNIQUE(msg_id)` means there can only be one spelling, so it is this one.
    """
    key = reflect.normalize_key(manifest.get("key") or "")
    if not key:
        return 0
    span = manifest.get("chunk_seconds")
    rows = []

    video = manifest.get("video") or {}
    if video.get("msg_id"):
        rows.append((key, "video", 0, int(video["msg_id"]),
                     video.get("file_id", ""), video.get("name", ""),
                     int(video.get("bytes") or 0), video.get("sha256", ""),
                     0.0, manifest.get("duration"), span))

    for c in (manifest.get("chunks") or []):
        if not c.get("msg_id"):
            continue
        rows.append((key, "clip", int(c.get("i") or 0), int(c["msg_id"]),
                     c.get("file_id", ""), c.get("name", ""),
                     int(c.get("bytes") or 0), c.get("sha256", ""),
                     float(c.get("t0") or 0.0), float(c.get("t1") or 0.0),
                     span))

    for a in (manifest.get("assets") or []):
        if not a.get("msg_id"):
            continue
        rows.append((key, str(a.get("kind") or "asset"), 0, int(a["msg_id"]),
                     a.get("file_id", ""), a.get("name", ""),
                     int(a.get("bytes") or 0), a.get("sha256", ""),
                     None, None, span))

    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO parts (video_key, kind, seq, msg_id, file_id, "
        "name, bytes, sha256, t_start, t_end, chunk_seconds) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def clip_at(conn: sqlite3.Connection, video_key: str, t: float) -> dict:
    """The clip covering timestamp `t`, or {}.

    Clip boundaries come from the muxer, not from `t // seconds`, because
    segmenting cuts on keyframes and a reel's GOP is rarely exactly the
    requested length. So this is a range query, and the `<=`/`>` asymmetry is
    what stops a `t` that lands exactly on a boundary matching two clips.
    """
    try:
        row = conn.execute(
            "SELECT video_key, kind, seq, msg_id, file_id, name, bytes, "
            "t_start, t_end FROM parts "
            "WHERE video_key=? AND kind='clip' AND t_start<=? AND t_end>? "
            "ORDER BY seq LIMIT 1", (str(video_key), float(t), float(t))
        ).fetchone()
    except sqlite3.Error:
        return {}
    if not row:
        # Past the last clip — a `t` at or beyond the end should still play the
        # final clip rather than nothing, which is what a search hit on the last
        # second of a video asks for.
        try:
            row = conn.execute(
                "SELECT video_key, kind, seq, msg_id, file_id, name, bytes, "
                "t_start, t_end FROM parts WHERE video_key=? AND kind='clip' "
                "ORDER BY seq DESC LIMIT 1", (str(video_key),)).fetchone()
        except sqlite3.Error:
            return {}
    if not row:
        return {}
    cols = ("video_key", "kind", "seq", "msg_id", "file_id", "name", "bytes",
            "t_start", "t_end")
    return dict(zip(cols, row))


def clips_for(conn: sqlite3.Connection, video_key: str,
              t0: float = None, t1: float = None) -> list:
    """Every clip for a video, optionally limited to a time window."""
    sql = ("SELECT seq, msg_id, file_id, name, bytes, t_start, t_end "
           "FROM parts WHERE video_key=? AND kind='clip'")
    args = [str(video_key)]
    if t0 is not None:
        sql += " AND t_end > ?"
        args.append(float(t0))
    if t1 is not None:
        sql += " AND t_start < ?"
        args.append(float(t1))
    sql += " ORDER BY seq"
    try:
        cur = conn.execute(sql, args)
    except sqlite3.Error:
        return []
    cols = ("seq", "msg_id", "file_id", "name", "bytes", "t_start", "t_end")
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def part_of(conn: sqlite3.Connection, video_key: str, kind: str) -> dict:
    """One non-clip asset for a video (`meta`, `manifest`, `frames`), or {}."""
    try:
        row = conn.execute(
            "SELECT msg_id, file_id, name, bytes FROM parts "
            "WHERE video_key=? AND kind=? ORDER BY msg_id DESC LIMIT 1",
            (str(video_key), str(kind))).fetchone()
    except sqlite3.Error:
        return {}
    if not row:
        return {}
    return dict(zip(("msg_id", "file_id", "name", "bytes"), row))


def has_clips(conn: sqlite3.Connection, video_key: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM parts WHERE video_key=? AND kind='clip' LIMIT 1",
            (str(video_key),)).fetchone()
    except sqlite3.Error:
        return False
    return bool(row)


def keys_with_clips(conn: sqlite3.Connection) -> set:
    """Every video that has a clip index, for the UI to badge instant playback."""
    try:
        return {r[0] for r in conn.execute(
            "SELECT DISTINCT video_key FROM parts WHERE kind='clip'")}
    except sqlite3.Error:
        return set()


def _collect(conn: sqlite3.Connection) -> dict:
    """Read every text source into {(video_key, source): {rows, table}}.

    Grouped by video and source because that is the unit passages merge within:
    two consecutive subtitle lines belong together, a subtitle line and an OCR
    hit at the same second do not.
    """
    buckets = {}
    specs = reflect.text_sources(conn)
    log(f"indexing {len(specs)} text source(s): " +
        ", ".join(sorted({f"{s['table']}.{s['text']}" for s in specs})))

    for spec in specs:
        _set(detail=f"reading {spec['table']}.{spec['text']}")
        try:
            cur = conn.execute(spec["sql"])
        except sqlite3.Error as e:
            log(f"skipped {spec['table']}.{spec['text']} — {e}")
            continue

        source = spec["source"]
        n = 0
        while True:
            batch = cur.fetchmany(5000)
            if not batch:
                break
            for key, t0, t1, raw in batch:
                vk = reflect.normalize_key(key)
                if not vk:
                    continue
                text = clean_text(raw)
                if not text:
                    continue
                b = buckets.setdefault((vk, source),
                                       {"rows": [], "table": spec["table"]})
                b["rows"].append((t0, t1, text))
                n += 1
        if n:
            log(f"  {spec['table']}.{spec['text']} → {n} row(s) as {source}")
    return buckets


def _video_metadata(conn: sqlite3.Connection) -> dict:
    """Best-effort per-video metadata, tolerating a moved schema.

    Every lookup is guarded: this runs against whatever the channel happened to
    contain, which may predate half these columns. A missing table costs that
    field, not the build.
    """
    meta = {}

    def absorb(sql, mapping):
        try:
            cur = conn.execute(sql)
        except sqlite3.Error:
            return
        names = [d[0] for d in cur.description]
        for row in cur.fetchall():
            r = dict(zip(names, row))
            vk = reflect.normalize_key(r.get(mapping["key"]))
            if not vk:
                continue
            slot = meta.setdefault(vk, {})
            for dest, src in mapping["fields"].items():
                val = r.get(src)
                if val not in (None, "") and slot.get(dest) in (None, ""):
                    slot[dest] = val

    tables = set(reflect.tables(conn))

    if "videos" in tables:
        cols = {c["name"] for c in reflect.columns(conn, "videos")}
        want = {"msg_id": "msg_id", "title": "title", "duration": "duration_sec",
                "width": "width", "height": "height", "fps": "fps",
                "size_mb": "file_size_mb", "created_at": "created_at",
                "local_path": "abs_path", "poster": "thumb"}
        sel = {d: s for d, s in want.items() if s in cols}
        if "msg_id" in cols and sel:
            absorb(f'SELECT {", ".join(sorted(set(sel.values())))} FROM videos',
                   {"key": "msg_id", "fields": sel})

    if "posts" in tables:
        cols = {c["name"] for c in reflect.columns(conn, "posts")}
        if "video_id" in cols:
            pieces = ["p.video_id"]
            fields = {}
            for dest, src in (("caption", "caption"), ("likes", "likes"),
                              ("local_path", "local_video_path")):
                if src in cols:
                    pieces.append(f"p.{src}")
                    fields[dest] = src
            if "creator_id" in cols and "creators" in tables:
                pieces.append("(SELECT username FROM creators WHERE "
                              "id = p.creator_id) AS creator")
                fields["creator"] = "creator"
            if "category_id" in cols and "categories" in tables:
                pieces.append("(SELECT name FROM categories WHERE "
                              "id = p.category_id) AS category")
                fields["category"] = "category"
            absorb(f'SELECT {", ".join(pieces)} FROM posts p',
                   {"key": "video_id", "fields": fields})

    # The new capture/process plane's own row per video. It is the only writer
    # that measures a video with ffprobe, and for anything it captured it is the
    # only source of a `msg_id` at all — a video keyed by shortcode has no digits
    # to fall back on, so without this it has no metadata row, no duration and no
    # way to be fetched. Absorbed after `videos`/`posts` so the legacy harvest
    # index still wins where both know a field.
    if "video" in tables:
        cols = {c["name"] for c in reflect.columns(conn, "video")}
        want = {"msg_id": "msg_id", "duration": "duration", "width": "width",
                "height": "height", "fps": "fps", "creator": "uploader",
                "created_at": "taken_at"}
        sel = {d: s for d, s in want.items() if s in cols}
        if "video_key" in cols and sel:
            picked = sorted(set(sel.values()) | {"video_key"})
            absorb(f'SELECT {", ".join(picked)} FROM video',
                   {"key": "video_key", "fields": sel})

    # The Omniscient side knows a path for videos the harvest index may not.
    for t in ("omni_chunks", "omni_frames"):
        if t not in tables:
            continue
        cols = {c["name"] for c in reflect.columns(conn, t)}
        if "video_uuid" in cols and "video_path" in cols:
            absorb(f"SELECT video_uuid, video_path FROM {t} "
                   f"WHERE video_path IS NOT NULL GROUP BY video_uuid",
                   {"key": "video_uuid", "fields": {"local_path": "video_path"}})
    return meta


def rebuild(conn: sqlite3.Connection, embed: bool = True) -> dict:
    """Rebuild `moments` and `video_index` from whatever is in the database.

    A full rebuild rather than an incremental one: the whole table is a few
    hundred thousand rows of text, it rebuilds in seconds, and incremental
    updates against a schema that can change underneath you are how indexes
    drift out of sync with their source. Cheap and always correct beats clever.

    Identity is settled *first*, before a single source row is read. That
    ordering is the fix for the duplication defect, not a tidiness preference:
    `_collect` and `_video_metadata` both reduce their keys through
    `reflect.normalize_key`, and until the alias map is installed that function
    can only see what is in the characters of a string. Nothing in `38` says it
    is `DZDNyKgv70R`, so with the map built afterwards the passages of one reel
    split across two rows and Home counted 62 videos where there are 30.
    """
    if _STATE["running"]:
        return {"ok": False, "note": "an index build is already running"}
    _set(phase="reading", running=True, error="", started_at=time.time(),
         finished_at=0.0, moments=0, videos=0, detail="resolving identity")

    try:
        has_fts = ensure_schema(conn)

        ident = identity.refresh(conn, media_dir=getattr(config, "MEDIA_DIR", ""),
                                 ledger_path=getattr(config, "LEDGER_PATH", ""))
        report = ident["audit"]
        log(f"identity — {report['videos']} video(s), {report['aliases']} alias(es)"
            f", {report['unresolved']} unresolved"
            + (f", {len(report['conflicts'])} conflict(s)"
               if report["conflicts"] else "")
            + (f", {report['twins']} twin(s)" if report["twins"] else "")
            + (f", {report['reuploads']} re-upload(s)"
               if report.get("reuploads") else ""))
        if not report["ok"]:
            for t in report["tables"]:
                if not t["ok"]:
                    log(f"  {t['table']}.{t['column']} — {t['unresolved']} key(s) "
                        f"name no video, e.g. {', '.join(t['examples'])}")
        _set(detail="reading sources")

        buckets = _collect(conn)

        _set(phase="writing", detail="building passages")
        conn.execute("DELETE FROM moments")
        if has_fts:
            try:
                conn.execute("INSERT INTO moments_fts(moments_fts) "
                             "VALUES('delete-all')")
            except sqlite3.Error:
                conn.execute("DROP TABLE IF EXISTS moments_fts")
                has_fts = ensure_schema(conn)

        weights = config.SOURCE_WEIGHT
        # A moment belongs to a video. If the key a source row carries names no
        # video even after identity has resolved it, there is nothing for the
        # passage to be *about*: no card to show it on, no file to play, no
        # duration to seek within. Those rows used to be written anyway, which is
        # how `moments` came to hold nine hundred keys for thirty videos — most
        # of them channel message ids from a scan log.
        #
        # Guarded on a non-empty canonical set for the same reason
        # `_build_video_index` is: an archive with no `video` table has no set to
        # check against, and dropping everything would be worse than keeping it.
        canonical = identity.canonical_keys(conn)
        rows_out = []
        per_video = {}
        orphaned = {}
        for (vk, source), bucket in buckets.items():
            if canonical and vk not in canonical:
                orphaned[vk] = orphaned.get(vk, 0) + len(bucket["rows"])
                continue
            w = weights.get(source, 1.0)
            for t0, t1, text in build_passages(bucket["rows"]):
                rows_out.append((vk, t0, t1, source, bucket["table"], w,
                                 text, _hash(text)))
                slot = per_video.setdefault(vk, {"sources": {}, "chars": 0})
                slot["sources"][source] = slot["sources"].get(source, 0) + 1
                slot["chars"] += len(text)
        if orphaned:
            log(f"skipped {sum(orphaned.values())} row(s) under "
                f"{len(orphaned)} key(s) that name no video: "
                + ", ".join(sorted(orphaned)[:6])
                + (" …" if len(orphaned) > 6 else ""))

        _set(detail=f"writing {len(rows_out)} passage(s)")
        conn.executemany(
            "INSERT OR IGNORE INTO moments"
            "(video_key, t_start, t_end, source, src_table, weight, text, "
            " text_hash) VALUES (?,?,?,?,?,?,?,?)", rows_out)
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0]
        _set(moments=total)

        if has_fts:
            _set(phase="fts", detail="building full-text index")
            conn.execute("INSERT INTO moments_fts(rowid, text) "
                         "SELECT id, text FROM moments")
            conn.execute("INSERT INTO moments_fts(moments_fts) "
                         "VALUES('optimize')")
            conn.commit()
        _set(lexical_ready=True)

        _set(phase="writing", detail="summarising videos")
        _build_video_index(conn, per_video)
        _set(videos=conn.execute(
            "SELECT COUNT(*) FROM video_index").fetchone()[0])

        from .ingest import meta_set
        meta_set(conn, "index_fingerprint", reflect.fingerprint(conn))
        meta_set(conn, "index_built_at", time.time())
        meta_set(conn, "index_moments", total)
        meta_set(conn, "index_has_fts", int(has_fts))

        # The generation this table belongs to. `moments.id` is reassigned by
        # every rebuild — the DELETE above frees the rowids and the INSERT hands
        # them out again in a different order — so a vector file built for an
        # earlier generation still has the right *shape* while pointing every hit
        # at the wrong passage. Nothing in the file itself says which table it
        # was made from, so this id is what says it, and both the writer and the
        # reader check it.
        build_id = uuid.uuid4().hex[:12]
        meta_set(conn, "index_build_id", build_id)

        log(f"index built — {total} passage(s) across {_STATE['videos']} video(s)"
            + ("" if has_fts else " (no fts5)"))

        _set(phase="done", running=False, finished_at=time.time(),
             detail=f"{total} passage(s) · {_STATE['videos']} video(s)")
        result = {"ok": True, "moments": total, "videos": _STATE["videos"],
                  "fts": has_fts, "build_id": build_id}
    except Exception as e:
        _set(phase="error", running=False, finished_at=time.time(),
             error=f"{type(e).__name__}: {e}", detail="index build failed")
        log(f"index build failed — {type(e).__name__}: {e}")
        return {"ok": False, "note": f"{type(e).__name__}: {e}"}

    if embed:
        start_embedding(conn_path=config.DB_PATH, build_id=build_id)
    return result


def _build_video_index(conn: sqlite3.Connection, per_video: dict) -> None:
    meta = _video_metadata(conn)
    spans = {}
    for vk, t_end in conn.execute(
            "SELECT video_key, MAX(t_end) FROM moments GROUP BY video_key"):
        spans[vk] = t_end

    # The keys of this table are the archive's videos, and the only thing
    # entitled to say what those are is `identity`. Taking the union of whatever
    # `_collect` and `_video_metadata` happened to produce is what put 62 rows
    # here: `moments` had a row under `38` and another under `DZDNyKgv70R`, both
    # spellings landed in the union, and Home counted the spellings.
    #
    # Now the union is *resolved* and then intersected with the canonical set.
    # Anything left over is evidence naming a video that has no `video` row —
    # which identity already tried to adopt, so reaching here means it is
    # genuinely unidentifiable, and a card for it would be a card for nothing.
    canonical = identity.canonical_keys(conn)
    res = identity.Resolver(conn)
    keys, dropped = set(), {}
    for raw in set(per_video) | set(meta) | set(spans):
        vk = res(raw) or raw
        # An archive with no `video` table at all — a bare legacy lake, or a
        # restore that has not replayed its shards yet — has no canonical set to
        # check against. Filtering on an empty set would produce an empty index,
        # so in that case every resolved key stands. Better a card that might be
        # a duplicate than no cards.
        if not canonical or vk in canonical:
            keys.add(vk)
        else:
            dropped[raw] = vk
    if dropped:
        log(f"video_index — {len(dropped)} key(s) named no video and were left "
            f"out: {', '.join(sorted(dropped)[:6])}")

    ident = identity.bulk(conn)
    _blank = {"aliases": [], "messages": [], "collections": [], "twins": []}

    # A resolved key may have arrived under several spellings, and each of them
    # carries its own share of the evidence. They are folded together here, so
    # a video whose transcript came in as `38` and whose narrative came in as
    # its shortcode gets one row with the sum of both, not two rows with half
    # each and not one row that silently keeps whichever was read last.
    folded_meta, folded_counts = {}, {}
    for raw, m in meta.items():
        vk = res(raw) or raw
        slot = folded_meta.setdefault(vk, {})
        for k, v in m.items():
            if v not in (None, "") and slot.get(k) in (None, ""):
                slot[k] = v
    for raw, p in per_video.items():
        vk = res(raw) or raw
        slot = folded_counts.setdefault(vk, {"sources": {}, "chars": 0})
        for s, n in p["sources"].items():
            slot["sources"][s] = slot["sources"].get(s, 0) + n
        slot["chars"] += p["chars"]

    # `msg_id` says where in the channel this video's file is, and on that one
    # question the capture plane outranks the legacy harvest. Both know a number
    # for the same reel, and they are not always the same number: reel
    # `DZDNyKgv70R` was uploaded twice, so the harvest recorded message 10 and
    # the capture ledger recorded 40. Either plays, but only the ledger's is the
    # one the manifest, the `parts` rows and the clip routes were built around,
    # so a card holding the other silently loses instant playback.
    authoritative = {}
    try:
        for k, mid in conn.execute(
                "SELECT video_key, msg_id FROM video "
                "WHERE msg_id IS NOT NULL"):
            authoritative[res(k) or str(k)] = mid
    except sqlite3.Error:
        pass

    conn.execute("DELETE FROM video_index")
    rows = []
    for vk in keys:
        m = folded_meta.get(vk, {})
        p = folded_counts.get(vk, {"sources": {}, "chars": 0})
        srcs = p["sources"]
        duration = _as_float(m.get("duration"))
        if not duration:
            # No metadata row for this video, but its moments know how far in
            # they go. A ribbon needs a length; this is the honest lower bound.
            duration = _as_float(spans.get(vk)) or 0.0
        # `msg_id` is where the video *is*, not what it is called. The old
        # fallback `_int(vk)` was the same conflation one level down: it read a
        # numeric key as a message id, which is exactly how a message id came to
        # be used as an identity in the first place.
        ids = ident.get(vk) or _blank
        rows.append((
            vk, _int(authoritative.get(vk)) or _int(m.get("msg_id")),
            m.get("title"),
            m.get("caption"), m.get("creator"), m.get("category"),
            duration, _int(m.get("width")), _int(m.get("height")),
            _as_float(m.get("fps")), _as_float(m.get("size_mb")),
            _int(m.get("likes")), _as_float(m.get("created_at")),
            m.get("local_path"), m.get("poster"),
            sum(srcs.values()), json.dumps(srcs),
            1 if srcs.get("speech") else 0,
            1 if srcs.get("narrative") else 0,
            p["chars"],
            "" if identity.is_upload(vk) else vk,
            identity.canonical_url(vk),
            json.dumps(sorted(ids["aliases"])),
            json.dumps(sorted(ids["messages"])),
            json.dumps(sorted(ids["collections"])),
            json.dumps(sorted(ids["twins"])) if ids["twins"] else "",
            1 if identity.is_stub(conn, vk) else 0))
    conn.executemany(
        "INSERT OR REPLACE INTO video_index(video_key, msg_id, title, caption, "
        "creator, category, duration, width, height, fps, size_mb, likes, "
        "created_at, local_path, poster, moment_count, sources, has_speech, "
        "has_narrative, text_len, shortcode, url, aliases, messages, "
        "collections, twin_of, is_stub) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def _int(v):
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════
# Written as a flat float32 file rather than into SQLite. Search needs every
# vector as one contiguous matrix to multiply against; pulling 200k BLOBs out of
# SQLite and stacking them per query would cost more than the search itself.
_EMBED_THREAD = None


def vector_state() -> dict:
    try:
        with open(config.VECTOR_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def start_embedding(conn_path: str = None, build_id: str = "") -> bool:
    """Kick off the dense index in the background.

    Deliberately not blocking. Lexical search is already live at this point, so
    the site is usable while this runs; when it finishes, the ranker starts
    fusing dense results in and nobody has to reload anything.

    `build_id` is the index generation these vectors will describe. It travels
    with the thread so the thread can check, at the last possible moment, that
    the table it read is still the table on disk.
    """
    global _EMBED_THREAD
    if _EMBED_THREAD is not None and _EMBED_THREAD.is_alive():
        return False
    _EMBED_THREAD = threading.Thread(
        target=_embed_all, args=(conn_path or config.DB_PATH, build_id),
        name="atlas-embed", daemon=True)
    _EMBED_THREAD.start()
    return True


def _embed_all(db_path: str, build_id: str = "") -> None:
    import sqlite3 as _sq
    try:
        from .encoder import get_encoder
        enc = get_encoder()
    except Exception as e:
        log(f"dense index skipped — encoder unavailable ({type(e).__name__}: "
            f"{e}). Lexical search is unaffected.")
        _set(dense_ready=False, detail="lexical only — no encoder")
        return
    if enc is None:
        _set(dense_ready=False, detail="lexical only — no encoder")
        return

    conn = _sq.connect(db_path, timeout=60.0, check_same_thread=False)
    try:
        rows = conn.execute(
            "SELECT id, text FROM moments ORDER BY id").fetchall()
    finally:
        conn.close()
    if not rows:
        return

    _set(phase="embedding", embed_total=len(rows), embedded=0,
         detail=f"encoding {len(rows)} passage(s)")
    try:
        import numpy as np
    except ImportError:
        log("dense index skipped — numpy missing")
        return

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    dim = config.EMBED_DIM
    vecs = np.zeros((len(rows), dim), dtype=np.float32)

    batch = config.EMBED_BATCH
    t0 = time.time()
    for i in range(0, len(rows), batch):
        chunk = [r[1] for r in rows[i:i + batch]]
        try:
            out = enc.encode_passages(chunk)
        except Exception as e:
            log(f"encoder failed at {i} ({type(e).__name__}: {e}) — "
                f"keeping the {i} vectors already made")
            vecs = vecs[:i]
            ids = ids[:i]
            break
        vecs[i:i + len(chunk)] = out
        _set(embedded=min(i + batch, len(rows)))
        if i and i % (batch * 20) == 0:
            done = i + len(chunk)
            rate = done / max(0.001, time.time() - t0)
            _set(detail=f"encoding {done}/{len(rows)} · {rate:.0f}/s")

    if len(ids) == 0:
        return

    # Encoding takes minutes; a rebuild that started while it ran has already
    # reassigned every `moments.id` these vectors are keyed by. Writing them now
    # would leave a well-formed dense index that maps hits to the wrong
    # passages — search's worst failure mode, because it looks like it works.
    # The next build's own embed pass replaces them, so dropping these costs a
    # cycle of dense search and nothing else.
    if build_id:
        try:
            check = _sq.connect(db_path, timeout=60.0, check_same_thread=False)
            try:
                from .ingest import meta_get
                now_id = meta_get(check, "index_build_id", "")
            finally:
                check.close()
        except Exception:                                   # noqa: BLE001
            now_id = build_id       # cannot tell; the reader checks again
        if now_id and now_id != build_id:
            log("dense vectors discarded — a newer index build superseded "
                "this one")
            _set(dense_ready=False, phase="done", finished_at=time.time(),
                 detail="dense index superseded mid-build")
            return

    tmp_v = config.VECTOR_PATH + ".tmp"
    tmp_i = config.VECTOR_PATH + ".ids.tmp"
    vecs.tofile(tmp_v)
    ids.tofile(tmp_i)
    os.replace(tmp_v, config.VECTOR_PATH)
    os.replace(tmp_i, config.VECTOR_PATH + ".ids")
    with open(config.VECTOR_META, "w", encoding="utf-8") as f:
        json.dump({"dim": dim, "count": int(len(ids)),
                   "model": config.EMBED_MODEL, "built_at": time.time(),
                   "build_id": build_id}, f)

    _set(dense_ready=True, phase="done", finished_at=time.time(),
         detail=f"dense index ready — {len(ids)} vector(s) in "
                f"{time.time() - t0:.0f}s")
    log(f"dense index ready — {len(ids)} vectors, {time.time() - t0:.0f}s")

    from . import search
    search.reload_vectors(expect=build_id)

    # The map is a projection of exactly these vectors, so the moment they land
    # is the moment it can be drawn — and the moment any previously built map
    # became a picture of an older archive. Building it here rather than on the
    # first click keeps opening the tab instant, and it runs in its own thread
    # so the encoder finishing is not held up by a projection.
    try:
        from . import maps
        maps.start_build(db_path)
    except Exception as e:                                  # noqa: BLE001
        log(f"map build could not start — {type(e).__name__}: {e}")
