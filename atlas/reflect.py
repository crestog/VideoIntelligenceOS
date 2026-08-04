"""
Schema reflection.

The requirement this file exists for: *"even if I change things, add or subtract
things from the database, it should still work, or require very very minimum
changes in code."*

The usual way to read a database is to write its column names into the program.
That works exactly until the schema moves. Add `frame_notes.pose_summary` and
the search index silently ignores it. Rename `chunks.description` and search
returns nothing, with no error anywhere. Drop a table and the UI throws.

So Atlas never names a column outside this module. It asks three questions at
runtime and derives everything else:

  1. What tables are here?              → `tables()`
  2. What does each column mean?        → role inference below
  3. Has any of that changed?           → `fingerprint()`

Role inference is the interesting part. Rather than matching a fixed list, each
column is scored against a small vocabulary of *roles* — the key that identifies
a video, the timestamps that place a row on the timeline, and the free text a
human would search. A new TEXT column is picked up as searchable content
automatically, because the rule is "text that is not an id, a path, or an
enum", not "one of these seven names". A renamed column is found as long as the
new name still reads like what it is.

Two rules here are stricter than they first look, and both are stricter on
purpose:

  A key must be *named* like a video key. The tempting fallback — "any single
  numeric primary key is the key" — attaches `categories(id=3, name='fitness')`
  to video 3. That is not a crash, it is worse: search quietly returns the wrong
  video. When the identity of a row is unknown, the row is not indexed.

  A blocked column suffix must be a whole token. Matching raw string suffixes
  drops `objects` for ending in "ts" and `format` for ending in "at". Names are
  split on separators and case changes, and only the final token is judged.

Nothing here raises on an unfamiliar schema. An unrecognised table becomes a
browsable table with no moments in it, which is the correct degradation: you can
still see the data, you just cannot search text Atlas could not identify as text.
"""

import hashlib
import re
import sqlite3

# ── Role vocabularies ─────────────────────────────────────────────────────
# Ordered by confidence: when a table offers several candidates, the earlier
# name wins. Matched against the *normalised* column name (lowercase,
# non-alphanumerics stripped), so `video_uuid`, `videoUUID` and `video uuid`
# are the same string by the time they get here.
_KEY_NAMES = ("videouuid", "videokey", "mediakey", "msgid", "videoid",
              "messageid", "postid", "reelid", "clipid", "itemid",
              "video", "uuid", "mediaid")
_START_NAMES = ("startt", "startsec", "starttime", "tstart", "start",
                "tssec", "timestamp", "ts", "time", "sec", "seconds",
                "offset", "position")
_END_NAMES = ("endt", "endsec", "endtime", "tend", "end", "stop", "until")

# Wall-clock columns. A row's insert time is not a position in a video, and
# treating it as one puts every moment at t=1.75 billion seconds.
_NOT_TIMELINE = {"createdat", "updatedat", "insertedat", "modifiedat",
                 "date", "datetime", "epoch", "fetchedat", "importedat"}

# Final-token blocklist for content columns: identifiers, filesystem paths,
# enums and formatted duplicates of a numeric column. Indexing these makes
# search worse — a query for "kitchen" should not match because a file happens
# to live in /kaggle/temp/kitchen/.
_NOT_CONTENT_TOKEN = {
    "id", "ids", "path", "paths", "uuid", "url", "uri", "link", "href",
    "at", "on", "ts", "time", "date", "status", "state", "mode", "kind",
    "type", "hash", "sha256", "md5", "checksum", "sig", "signature",
    "ext", "mime", "mimetype", "filename", "dir", "folder", "thumb",
    "flag", "version", "rev", "idx", "index", "seq", "num", "count",
}
# Whole names that are not content regardless of how they tokenise.
_NOT_CONTENT_EXACT = {"durationstr", "firstframe", "folderid", "thumb",
                      "localvideopath", "abspath", "videopath", "filepath"}

# What kind of evidence a column carries. Keyed by (table, column) with both
# sides normalised, because the same column name means different things in
# different tables: `chunks.description` is a vision model narrating a
# five-second window; `frame_notes.description` describes one still frame.
_SOURCE_MAP = {
    ("chunks", "description"):      "narrative",
    ("transcripts", "text"):        "speech",
    ("framenotes", "description"):  "visual",
    ("framenotes", "objects"):      "visual",
    ("framenotes", "ocrtext"):      "ocr",
    ("posts", "caption"):           "caption",
    ("videos", "title"):            "meta",
    ("creators", "username"):       "meta",
    ("categories", "name"):         "meta",
}

# Fallback for a column nobody has met before. Checked in order, first hit wins.
_SOURCE_HINTS = (
    ("ocr", "ocr"), ("subtitle", "speech"), ("transcript", "speech"),
    ("speech", "speech"), ("audio", "speech"), ("dialog", "speech"),
    ("caption", "caption"), ("narrat", "narrative"), ("summar", "narrative"),
    ("descri", "narrative"), ("story", "narrative"),
    ("object", "visual"), ("label", "visual"), ("tag", "visual"),
    ("scene", "visual"), ("pose", "visual"), ("action", "visual"),
    ("title", "meta"), ("name", "meta"), ("author", "meta"),
)

# Atlas's own tables. They are derived from the others, so indexing them would
# feed search its own output back to it.
_ATLAS_OWN = {"moments", "moments_fts", "bundles", "atlas_meta", "ingest_log",
              "video_index", "graph_nodes", "graph_edges"}

_FTS_SHADOW = re.compile(r"_(data|idx|content|docsize|config)$")
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _tokens(name: str) -> list:
    """Split a column name into words, on separators and on camelCase."""
    return [t.lower() for t in _TOKEN_SPLIT.split(name or "") if t]


# ══════════════════════════════════════════════════════════════════════════
# RAW INTROSPECTION
# ══════════════════════════════════════════════════════════════════════════
def tables(conn: sqlite3.Connection) -> list:
    """Every real table a human would want to look at, in a stable order.

    Virtual tables are excluded by reading `sql`, not by name: sqlite_master
    reports an fts5 index with type='table', so `posts_search` looks exactly
    like a real table until you notice its DDL says CREATE VIRTUAL TABLE.
    """
    try:
        rows = conn.execute(
            "SELECT name, type, COALESCE(sql,'') FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name").fetchall()
    except sqlite3.Error:
        return []
    out = []
    for name, _kind, sql in rows:
        low = name.lower()
        if "CREATE VIRTUAL TABLE" in (sql or "").upper():
            continue
        if _FTS_SHADOW.search(low):     # fts5 keeps four shadow tables per index
            continue
        if low in _ATLAS_OWN:
            continue
        out.append(name)
    return out


def columns(conn: sqlite3.Connection, table: str) -> list:
    """[{name, type, notnull, pk}] for one table. Empty list if it is gone."""
    try:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error:
        return []
    return [{"name": r[1], "type": (r[2] or "").upper(),
             "notnull": bool(r[3]), "pk": bool(r[5])} for r in rows]


def row_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    except sqlite3.Error:
        return 0


def fingerprint(conn: sqlite3.Connection) -> str:
    """A hash of the schema's shape.

    The indexer stores this alongside the moment table. When it differs, the
    schema moved and the index is rebuilt — which is how a new column becomes
    searchable without anyone editing code or pressing anything.
    """
    parts = []
    for t in tables(conn):
        cols = ",".join(f"{c['name']}:{c['type']}" for c in columns(conn, t))
        parts.append(f"{t}({cols})")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════
# ROLE INFERENCE
# ══════════════════════════════════════════════════════════════════════════
def _is_numeric(col: dict) -> bool:
    t = col["type"]
    return any(k in t for k in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC"))


def _is_texty(col: dict) -> bool:
    # A column with no declared type is TEXT in practice — pgdump only emits
    # BLOB for bytea, and an untyped SQLite column accepts anything.
    t = col["type"]
    return (not t) or ("CHAR" in t) or ("TEXT" in t) or ("CLOB" in t)


def key_column(cols: list) -> str:
    """The column that identifies which video a row belongs to, or "".

    Preference order matters more than it looks: `chunks` has both `video_uuid`
    and `chunk_id`, and picking `chunk_id` would give every row its own video.

    There is deliberately no "any numeric primary key" fallback — see the module
    docstring. A table whose key cannot be named is left unindexed rather than
    joined to whatever video happens to share its row id.
    """
    by_norm = {_norm(c["name"]): c["name"] for c in cols}
    for want in _KEY_NAMES:
        if want in by_norm:
            return by_norm[want]
    return ""


def time_columns(cols: list) -> tuple:
    """(start, end) column names, either of which may be "".

    A row with a start and no end is a point on the timeline, not a bug: a
    frame note happens at an instant. The moment builder gives those a small
    window rather than dropping them.
    """
    numeric = {_norm(c["name"]): c["name"] for c in cols
               if _is_numeric(c) or not c["type"]}
    start = end = ""
    for want in _START_NAMES:
        if want in numeric and want not in _NOT_TIMELINE:
            start = numeric[want]
            break
    for want in _END_NAMES:
        if want in numeric and want not in _NOT_TIMELINE:
            end = numeric[want]
            break
    if _norm(start) in _NOT_TIMELINE:
        start = ""
    if _norm(end) in _NOT_TIMELINE:
        end = ""
    return start, end


def content_columns(cols: list) -> list:
    """Text columns a person would actually want to search.

    This is the rule that makes new columns work for free: anything text-typed
    that is not an identifier, a path, a timestamp or a short enum is content.
    """
    out = []
    for c in cols:
        if not _is_texty(c) or c["pk"]:
            continue
        n = _norm(c["name"])
        if n in _NOT_CONTENT_EXACT:
            continue
        toks = _tokens(c["name"])
        if not toks:
            continue
        # Judge the final token only. `ocr_text` ends in "text" (content);
        # `local_video_path` ends in "path" (not). A single-token name is
        # judged whole, so `status` is out and `objects` is in.
        if toks[-1] in _NOT_CONTENT_TOKEN:
            continue
        out.append(c["name"])
    return out


def source_label(table: str, column: str) -> str:
    """Which kind of evidence a column carries, for weighting and for colour."""
    t = _norm(table)
    if t.startswith("omni"):
        t = t[4:]
    key = (t, _norm(column))
    if key in _SOURCE_MAP:
        return _SOURCE_MAP[key]
    n = _norm(column)
    for needle, label in _SOURCE_HINTS:
        if needle in n:
            return label
    return "meta"


# ══════════════════════════════════════════════════════════════════════════
# DIMENSION JOINS
# ══════════════════════════════════════════════════════════════════════════
def _plural_forms(stem: str) -> tuple:
    forms = [stem, stem + "s", stem + "es"]
    if stem.endswith("y"):
        forms.append(stem[:-1] + "ies")
    return tuple(forms)


def dimension_links(conn: sqlite3.Connection, table: str, cols: list) -> list:
    """Lookup tables this table points at, found by naming convention.

    `posts.creator_id` → `creators.id`, `posts.category_id` → `categories.id`.
    Without this, a creator's name lives in a table whose own key is a category
    id, so it can never be indexed safely on its own — and searching for a
    creator by name would silently return nothing.

    The convention is the only signal available: pg_dump's plain format carries
    foreign keys as ALTER TABLE statements that this pipeline skips, and
    lake.db never declared them.
    """
    present = {_norm(t): t for t in tables(conn)}
    links = []
    for c in cols:
        toks = _tokens(c["name"])
        if len(toks) < 2 or toks[-1] != "id":
            continue
        stem = "".join(toks[:-1])
        for form in _plural_forms(stem):
            target = present.get(form)
            if not target or target == table:
                continue
            tcols = columns(conn, target)
            tnames = {_norm(x["name"]): x["name"] for x in tcols}
            if "id" not in tnames:
                continue
            texts = content_columns(tcols)
            if not texts:
                continue
            links.append({"table": target, "local": c["name"],
                          "remote": tnames["id"], "texts": texts})
            break
    return links


# ══════════════════════════════════════════════════════════════════════════
# THE CATALOG
# ══════════════════════════════════════════════════════════════════════════
def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def text_sources(conn: sqlite3.Connection) -> list:
    """Every (table, key, time, text) the indexer should read, as ready SQL.

    The indexer holds no table names at all — it walks this list and runs the
    `sql` on each spec, which always yields exactly four columns:
    (key, start, end, text). Adding a table to the bundle adds rows to search;
    dropping one removes them; neither is a code change.
    """
    specs = []
    for table in tables(conn):
        cols = columns(conn, table)
        if not cols:
            continue
        key = key_column(cols)
        if not key:
            continue                       # nothing to attach a moment to
        start, end = time_columns(cols)
        s_expr = f"t.{_q(start)}" if start else "NULL"
        e_expr = f"t.{_q(end)}" if end else "NULL"

        for text_col in content_columns(cols):
            specs.append({
                "table": table, "key": key, "start": start, "end": end,
                "text": text_col, "source": source_label(table, text_col),
                "via": None,
                "sql": (f"SELECT t.{_q(key)}, {s_expr}, {e_expr}, "
                        f"t.{_q(text_col)} FROM {_q(table)} t "
                        f"WHERE t.{_q(text_col)} IS NOT NULL "
                        f"AND TRIM(t.{_q(text_col)}) <> ''"),
            })

        # Dimension text, pulled onto the parent's key so a creator's name is
        # searchable against the videos they made rather than against nothing.
        for link in dimension_links(conn, table, cols):
            for text_col in link["texts"]:
                specs.append({
                    "table": link["table"], "key": key, "start": "", "end": "",
                    "text": text_col,
                    "source": source_label(link["table"], text_col),
                    "via": f'{table}.{link["local"]}',
                    "sql": (f"SELECT t.{_q(key)}, NULL, NULL, d.{_q(text_col)} "
                            f"FROM {_q(table)} t "
                            f"JOIN {_q(link['table'])} d "
                            f"  ON t.{_q(link['local'])} = d.{_q(link['remote'])} "
                            f"WHERE d.{_q(text_col)} IS NOT NULL "
                            f"AND TRIM(d.{_q(text_col)}) <> ''"),
                })
    return specs


def describe(conn: sqlite3.Connection, samples: int = 0) -> dict:
    """Everything the Data tab needs to render a database it has never seen.

    Roles are returned alongside the raw columns so the UI can mark which column
    is the key and which carry searchable text — the same inference search uses,
    shown to the person looking at it.
    """
    out = {"fingerprint": fingerprint(conn), "tables": []}
    for table in tables(conn):
        cols = columns(conn, table)
        key = key_column(cols)
        start, end = time_columns(cols)
        content = set(content_columns(cols))
        entry = {
            "name": table,
            "rows": row_count(conn, table),
            "key": key,
            "start": start,
            "end": end,
            "indexed": bool(key and content),
            "columns": [{
                "name": c["name"],
                "type": c["type"] or "TEXT",
                "pk": c["pk"],
                "role": ("key" if c["name"] == key else
                         "start" if c["name"] == start else
                         "end" if c["name"] == end else
                         "content" if c["name"] in content else "field"),
                "source": (source_label(table, c["name"])
                           if c["name"] in content else None),
            } for c in cols],
        }
        if samples:
            try:
                cur = conn.execute(
                    f'SELECT * FROM {_q(table)} LIMIT {int(samples)}')
                names = [d[0] for d in cur.description]
                entry["sample"] = [dict(zip(names, r)) for r in cur.fetchall()]
            except sqlite3.Error:
                entry["sample"] = []
        out["tables"].append(entry)
    return out


# ══════════════════════════════════════════════════════════════════════════
# KEYS
# ══════════════════════════════════════════════════════════════════════════
_DIGITS = re.compile(r"\d+")


def normalize_key(value) -> str:
    """Reduce any spelling of a video's identity to its digits.

    Postgres says `tg1234`, lake.db says `1234` in `posts.video_id` and again
    in `videos.msg_id`, and a manifest says `"1234"`. They are the same reel.
    Normalising to the digits is what lets a narrative from one table and a
    transcript from another land on the same video without a mapping table that
    would need maintaining every time a producer changes its id format.
    """
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(int(value))
    m = _DIGITS.search(str(value))
    return m.group(0) if m else str(value).strip()
