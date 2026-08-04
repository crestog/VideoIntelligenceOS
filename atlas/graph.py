"""
The graph.

*"Show the graph database also... graph nodes and edges and everything should be
clickable and opened as I click... all information in database, exact videos it
is referring to, instantly playable."*

A relational database already is a graph — it just hides it. `posts.creator_id`
is an edge; `frame_notes.objects` is a fan of edges to every object seen; a
hashtag in a caption is an edge to everything else that used it. None of that is
visible in a table view, and all of it is exactly what a person means when they
say they want to explore.

So this module reads the graph *out* of the schema rather than being told it.
Four derivations, none of which names a table:

**Videos** are the anchor. One node per row of `video_index`, which is the one
table Atlas builds itself, so it is always there and always has a message id —
which is what makes every video node instantly playable.

**Dimensions** come from `reflect.dimension_links`. A column called
`<something>_id` pointing at a table called `<something>s` is a foreign key by
convention, and the convention is all there is: pg_dump's plain format carries
real keys as ALTER TABLE statements the importer skips. Each referenced row
becomes a node labelled with its own text, and the link becomes an edge. Add a
`mood_id` column and a `moods` table tomorrow and mood nodes appear with no code
change.

**Tags** come from list-shaped text. A column whose values parse as a JSON array,
or read as a short comma-separated list, is a set of things rather than a
sentence — `frame_notes.objects` is the obvious one, but the test is structural,
not by name. Each distinct item becomes a node, and every video whose row
mentioned it gets an edge. This is where the graph stops being an org chart and
starts being useful: it connects videos that share nothing but a coffee cup.

**Hashtags** are mined from every text column, because a caption's `#hashtag` is
an author-supplied label and throwing it away would be silly.

Everything is precomputed into two ordinary tables and read back through indexed
lookups, so expanding a node is a millisecond, not a scan. The graph is rebuilt
whenever the index is, which means it tracks the schema for free.

Edges carry a `ref` — the table, column and value they came from. That is what
makes an edge clickable: the interface can ask "why are these two connected?"
and get back the actual database rows that justify it.
"""

import json
import re
import sqlite3
import threading
import time

from . import reflect
from .tgchannel import log

# ── Limits ────────────────────────────────────────────────────────────────
# A graph is only explorable if it is finite. These caps are generous enough
# that nothing a person would look for is missing, and tight enough that the
# build stays a few seconds and the tables stay small.
MAX_TAGS_PER_COLUMN = 6000
MIN_TAG_COUNT       = 2       # a token seen once connects nothing
MAX_TOKEN_LEN       = 48
SAMPLE_ROWS         = 240     # how many values decide if a column is a list
FANOUT              = 60      # default neighbours returned per expansion

_DDL = (
    "CREATE TABLE IF NOT EXISTS graph_nodes ("
    "  id TEXT PRIMARY KEY,"
    "  kind TEXT NOT NULL,"        # video | dim | tag | hashtag
    "  label TEXT NOT NULL,"
    "  sub TEXT,"                  # the group: creators, objects, …
    "  weight REAL DEFAULT 0,"     # degree, for size and for ranking
    "  meta TEXT)",                # JSON, whatever that kind carries

    "CREATE INDEX IF NOT EXISTS graph_nodes_kind ON graph_nodes(kind, weight DESC)",
    "CREATE INDEX IF NOT EXISTS graph_nodes_sub  ON graph_nodes(sub, weight DESC)",
    "CREATE INDEX IF NOT EXISTS graph_nodes_label ON graph_nodes(label)",

    "CREATE TABLE IF NOT EXISTS graph_edges ("
    "  src TEXT NOT NULL,"
    "  dst TEXT NOT NULL,"
    "  rel TEXT NOT NULL,"
    "  weight REAL DEFAULT 1,"
    "  ref TEXT,"                  # table|column|value — why this edge exists
    "  PRIMARY KEY(src, dst, rel))",

    "CREATE INDEX IF NOT EXISTS graph_edges_src ON graph_edges(src)",
    "CREATE INDEX IF NOT EXISTS graph_edges_dst ON graph_edges(dst)",
)

_LOCK = threading.RLock()
_STATE = {"phase": "idle", "detail": "", "nodes": 0, "edges": 0, "at": 0.0}


def _set(**kw):
    with _LOCK:
        _STATE.update(kw)
        _STATE["at"] = time.time()


def status() -> dict:
    with _LOCK:
        return dict(_STATE)


def ensure_schema(conn: sqlite3.Connection) -> None:
    for ddl in _DDL:
        conn.execute(ddl)
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════
# IDS
# ══════════════════════════════════════════════════════════════════════════
# Ids are readable on purpose: a URL containing `t:frame_notes.objects:coffee`
# says what it points at, survives a rebuild, and can be shared. They are split
# from the left with a bounded count, so a token containing a colon is safe.
def video_id(key) -> str:
    return f"v:{key}"


def dim_id(table: str, row_id) -> str:
    return f"d:{table}:{row_id}"


def tag_id(table: str, column: str, token: str) -> str:
    return f"t:{table}.{column}:{token}"


def hashtag_id(token: str) -> str:
    return f"h:{token}"


def parse_id(node_id: str) -> dict:
    """Break an id back into its parts. Never raises — an unknown id is a miss."""
    s = str(node_id or "")
    kind = s[:1]
    if kind == "v":
        return {"kind": "video", "key": s[2:]}
    if kind == "d":
        rest = s[2:]
        table, _, row = rest.rpartition(":")
        return {"kind": "dim", "table": table, "row": row}
    if kind == "t":
        rest = s[2:]
        head, _, token = rest.partition(":")
        table, _, column = head.partition(".")
        return {"kind": "tag", "table": table, "column": column,
                "token": token}
    if kind == "h":
        return {"kind": "hashtag", "token": s[2:]}
    return {"kind": ""}


# ══════════════════════════════════════════════════════════════════════════
# TOKENS
# ══════════════════════════════════════════════════════════════════════════
_HASHTAG = re.compile(r"#([A-Za-z0-9_À-ɏ]{2,40})")
_LIST_SEP = re.compile(r"[,;|/]| - ")
_WS = re.compile(r"\s+")


def _clean_token(value) -> str:
    """Normalise one list item, or "" if it is not worth a node."""
    s = _WS.sub(" ", str(value or "")).strip().strip("\"'[]{}()").lower()
    if len(s) < 2 or len(s) > MAX_TOKEN_LEN:
        return ""
    # A bare number is an id or a count, not a thing anybody explores by.
    if s.replace(".", "").isdigit():
        return ""
    return s


def split_list(value) -> list:
    """Items in a list-shaped value, or [] if this is prose.

    JSON first, because the harvester writes object lists as arrays and a
    naive comma split on `["a, b", "c"]` would invent items. Then separators.
    A value with no separator at all is one item only if it is short — a
    paragraph is not a one-item list.
    """
    s = str(value or "").strip()
    if not s:
        return []
    if s[0] in "[{":
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, list):
            out = []
            for item in obj:
                if isinstance(item, (str, int, float)):
                    out.append(str(item))
                elif isinstance(item, dict):
                    # A list of records: take the first stringy field, which is
                    # the label in every shape the harvester emits.
                    for v in item.values():
                        if isinstance(v, str) and v.strip():
                            out.append(v)
                            break
            return out
        if isinstance(obj, dict):
            return [str(k) for k in obj]
    parts = [p for p in _LIST_SEP.split(s) if p.strip()]
    if len(parts) > 1:
        return parts
    return [s] if len(s) <= MAX_TOKEN_LEN else []


def _is_list_column(values: list) -> bool:
    """Does this column hold sets of things rather than sentences?

    Judged on shape: most values split into items, the items are short, and
    they are not full sentences. `objects` passes; `description` does not, even
    though a description contains commas — its items are long and there are
    only a couple of them per row.
    """
    seen = 0
    listy = 0
    lengths = []
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        seen += 1
        items = split_list(s)
        good = [t for t in (_clean_token(i) for i in items) if t]
        if not good:
            continue
        lengths.extend(len(t.split()) for t in good)
        if len(good) >= 2 or (s[0] in "[{" and good):
            listy += 1
    if seen < 4 or not lengths:
        return False
    mean_words = sum(lengths) / len(lengths)
    return (listy / seen) >= 0.55 and mean_words <= 4.0


# ══════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════
def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _label_column(cols: list) -> str:
    """The column a human would call this row's name."""
    texts = reflect.content_columns(cols)
    if not texts:
        return ""
    by_norm = {reflect._norm(c): c for c in texts}
    for want in ("name", "title", "username", "label", "handle", "slug",
                 "caption"):
        if want in by_norm:
            return by_norm[want]
    return texts[0]


def rebuild(conn: sqlite3.Connection) -> dict:
    """Derive the whole graph from the current schema. Idempotent.

    Built into memory and written in one transaction, because a half-written
    graph is worse than an old one — the interface would show edges to nodes
    that do not exist yet.
    """
    t0 = time.time()
    _set(phase="reading", detail="deriving nodes from the schema",
         nodes=0, edges=0)
    ensure_schema(conn)

    nodes = {}      # id → [kind, label, sub, meta dict]
    edges = {}      # (src, dst, rel) → [weight, ref]

    def add_node(nid, kind, label, sub, meta=None):
        slot = nodes.get(nid)
        if slot is None:
            nodes[nid] = [kind, label, sub, meta or {}]
        elif meta:
            slot[3].update(meta)

    def add_edge(src, dst, rel, ref="", weight=1.0):
        if not src or not dst or src == dst:
            return
        cell = edges.get((src, dst, rel))
        if cell is None:
            edges[(src, dst, rel)] = [weight, ref]
        else:
            cell[0] += weight

    # ── videos ────────────────────────────────────────────────────────────
    videos = set()
    try:
        cur = conn.execute(
            "SELECT video_key, title, caption, creator, category, duration, "
            "msg_id, moment_count, likes, created_at FROM video_index")
        names = [d[0] for d in cur.description]
        for row in cur.fetchall():
            r = dict(zip(names, row))
            key = str(r["video_key"])
            videos.add(key)
            add_node(video_id(key), "video",
                     r.get("title") or r.get("caption") or f"video {key}",
                     "videos",
                     {"video_key": key, "msg_id": r.get("msg_id"),
                      "duration": r.get("duration"),
                      "moments": r.get("moment_count"),
                      "likes": r.get("likes"),
                      "created_at": r.get("created_at")})
    except sqlite3.Error as e:
        log(f"graph: no video index yet — {e}", "WARN")
        return {"ok": False, "note": "index not built"}

    if not videos:
        _write(conn, {}, {})
        _set(phase="done", detail="nothing indexed yet", nodes=0, edges=0)
        return {"ok": True, "nodes": 0, "edges": 0}

    tables = reflect.tables(conn)

    # ── dimensions ────────────────────────────────────────────────────────
    _set(phase="reading", detail="following foreign keys")
    for table in tables:
        cols = reflect.columns(conn, table)
        key = reflect.key_column(cols)
        if not key:
            continue
        for link in reflect.dimension_links(conn, table, cols):
            target = link["table"]
            tcols = reflect.columns(conn, target)
            label_col = _label_column(tcols)
            if not label_col:
                continue
            # The edge's name is the column minus its `_id` tail: `creator_id`
            # reads as "creator", which is what the relationship is called.
            rel = re.sub(r"[_ ]?id$", "", link["local"], flags=re.I) or target
            sql = (f"SELECT t.{_q(key)}, d.{_q(link['remote'])}, "
                   f"d.{_q(label_col)} "
                   f"FROM {_q(table)} t JOIN {_q(target)} d "
                   f"  ON t.{_q(link['local'])} = d.{_q(link['remote'])} "
                   f"WHERE d.{_q(label_col)} IS NOT NULL")
            try:
                rows = conn.execute(sql).fetchall()
            except sqlite3.Error:
                continue
            ref = f"{table}|{link['local']}"
            for raw_key, row_id, label in rows:
                vk = reflect.normalize_key(raw_key)
                if vk not in videos:
                    continue
                nid = dim_id(target, row_id)
                add_node(nid, "dim", str(label), target,
                         {"table": target, "row_id": row_id,
                          "id_column": link["remote"]})
                add_edge(video_id(vk), nid, rel, f"{ref}|{row_id}")

    # ── tags and hashtags ─────────────────────────────────────────────────
    _set(phase="reading", detail="mining list columns and hashtags")
    for table in tables:
        cols = reflect.columns(conn, table)
        key = reflect.key_column(cols)
        if not key:
            continue
        for column in reflect.content_columns(cols):
            try:
                sample = [r[0] for r in conn.execute(
                    f"SELECT {_q(column)} FROM {_q(table)} "
                    f"WHERE {_q(column)} IS NOT NULL AND TRIM({_q(column)}) <> '' "
                    f"LIMIT {SAMPLE_ROWS}")]
            except sqlite3.Error:
                continue
            if not sample:
                continue

            listy = _is_list_column(sample)
            # Hashtags are worth mining from any text; a caption is prose and
            # still carries them.
            wants_hash = any("#" in str(v or "") for v in sample)
            if not listy and not wants_hash:
                continue

            try:
                rows = conn.execute(
                    f"SELECT {_q(key)}, {_q(column)} FROM {_q(table)} "
                    f"WHERE {_q(column)} IS NOT NULL "
                    f"AND TRIM({_q(column)}) <> ''")
            except sqlite3.Error:
                continue

            counts = {}          # token → {video_key, …}
            hashes = {}
            for raw_key, value in rows:
                vk = reflect.normalize_key(raw_key)
                if vk not in videos:
                    continue
                text = str(value or "")
                if listy:
                    for item in split_list(text):
                        tok = _clean_token(item)
                        if tok:
                            counts.setdefault(tok, set()).add(vk)
                if wants_hash and "#" in text:
                    for m in _HASHTAG.finditer(text):
                        hashes.setdefault(m.group(1).lower(), set()).add(vk)

            source = reflect.source_label(table, column)
            keep = sorted(counts.items(), key=lambda kv: -len(kv[1]))
            kept = 0
            for tok, keys in keep:
                if len(keys) < MIN_TAG_COUNT or kept >= MAX_TAGS_PER_COLUMN:
                    break
                kept += 1
                nid = tag_id(table, column, tok)
                add_node(nid, "tag", tok, column,
                         {"table": table, "column": column, "token": tok,
                          "source": source, "videos": len(keys)})
                for vk in keys:
                    add_edge(video_id(vk), nid, column,
                             f"{table}|{column}|{tok}")

            for tok, keys in hashes.items():
                if len(keys) < 1:
                    continue
                nid = hashtag_id(tok)
                add_node(nid, "hashtag", "#" + tok, "hashtags",
                         {"token": tok})
                for vk in keys:
                    add_edge(video_id(vk), nid, "hashtag",
                             f"{table}|{column}|#{tok}")

    # Degree is the only ranking signal that needs no configuration, and it is
    # the right one: the nodes worth showing first are the ones that connect
    # the most of the archive.
    degree = {}
    for (src, dst, _rel), (w, _ref) in edges.items():
        degree[src] = degree.get(src, 0) + w
        degree[dst] = degree.get(dst, 0) + w

    _set(phase="writing", detail="storing the graph")
    _write(conn, nodes, edges, degree)

    took = time.time() - t0
    _set(phase="done", nodes=len(nodes), edges=len(edges),
         detail=f"{len(nodes)} nodes, {len(edges)} edges in {took:.1f}s")
    log(f"graph built — {len(nodes)} node(s), {len(edges)} edge(s) "
        f"in {took:.1f}s")
    return {"ok": True, "nodes": len(nodes), "edges": len(edges),
            "seconds": round(took, 2)}


def _write(conn: sqlite3.Connection, nodes: dict, edges: dict,
           degree: dict = None) -> None:
    degree = degree or {}
    ensure_schema(conn)
    with conn:
        conn.execute("DELETE FROM graph_edges")
        conn.execute("DELETE FROM graph_nodes")
        conn.executemany(
            "INSERT OR REPLACE INTO graph_nodes(id, kind, label, sub, weight, "
            "meta) VALUES (?,?,?,?,?,?)",
            [(nid, kind, label, sub, float(degree.get(nid, 0)),
              json.dumps(meta, default=str))
             for nid, (kind, label, sub, meta) in nodes.items()])
        conn.executemany(
            "INSERT OR REPLACE INTO graph_edges(src, dst, rel, weight, ref) "
            "VALUES (?,?,?,?,?)",
            [(src, dst, rel, float(w), ref)
             for (src, dst, rel), (w, ref) in edges.items()])


# ══════════════════════════════════════════════════════════════════════════
# READ
# ══════════════════════════════════════════════════════════════════════════
def _node_row(conn: sqlite3.Connection, node_id: str) -> dict:
    try:
        row = conn.execute(
            "SELECT id, kind, label, sub, weight, meta FROM graph_nodes "
            "WHERE id = ?", (node_id,)).fetchone()
    except sqlite3.Error:
        return {}
    if not row:
        return {}
    try:
        meta = json.loads(row[5] or "{}")
    except ValueError:
        meta = {}
    return {"id": row[0], "kind": row[1], "label": row[2], "sub": row[3],
            "weight": row[4], "meta": meta}


def nodes_by_id(conn: sqlite3.Connection, ids: list) -> list:
    """Hydrate a set of ids in one query, preserving nothing about order."""
    ids = [i for i in dict.fromkeys(ids) if i]
    if not ids:
        return []
    out = []
    CHUNK = 400
    for i in range(0, len(ids), CHUNK):
        part = ids[i:i + CHUNK]
        marks = ",".join("?" * len(part))
        try:
            cur = conn.execute(
                f"SELECT id, kind, label, sub, weight, meta FROM graph_nodes "
                f"WHERE id IN ({marks})", part)
        except sqlite3.Error:
            continue
        for r in cur.fetchall():
            try:
                meta = json.loads(r[5] or "{}")
            except ValueError:
                meta = {}
            out.append({"id": r[0], "kind": r[1], "label": r[2], "sub": r[3],
                        "weight": r[4], "meta": meta})
    return out


def edges_among(conn: sqlite3.Connection, ids: list) -> list:
    """Every stored edge whose both ends are in this set.

    Called after each expansion. Without it the layout is a star — you would
    only ever see the edges you asked for, and never that two of the creators
    on screen share a video. The triangles are the whole point of a graph view.
    """
    ids = [i for i in dict.fromkeys(ids) if i]
    if len(ids) < 2:
        return []
    marks = ",".join("?" * len(ids))
    try:
        cur = conn.execute(
            f"SELECT src, dst, rel, weight, ref FROM graph_edges "
            f"WHERE src IN ({marks}) AND dst IN ({marks})", ids + ids)
    except sqlite3.Error:
        return []
    return [{"src": r[0], "dst": r[1], "rel": r[2], "weight": r[3],
             "ref": r[4]} for r in cur.fetchall()]


def neighbors(conn: sqlite3.Connection, node_id: str, limit: int = FANOUT,
              kind: str = "") -> dict:
    """One node's neighbourhood: the nodes, and every edge among all of them.

    Edges are stored one way — always video → thing — but exploration goes both
    ways, so both directions are queried and the far end is whichever is not
    the node asked about.
    """
    limit = max(1, min(int(limit or FANOUT), 400))
    centre = _node_row(conn, node_id)
    if not centre:
        return {"ok": False, "note": "no such node"}

    found = []
    try:
        cur = conn.execute(
            "SELECT e.dst, e.rel, e.weight, e.ref, n.kind FROM graph_edges e "
            "JOIN graph_nodes n ON n.id = e.dst WHERE e.src = ? "
            "UNION ALL "
            "SELECT e.src, e.rel, e.weight, e.ref, n.kind FROM graph_edges e "
            "JOIN graph_nodes n ON n.id = e.src WHERE e.dst = ?",
            (node_id, node_id))
        found = cur.fetchall()
    except sqlite3.Error:
        found = []

    if kind:
        wanted = {k.strip() for k in kind.split(",") if k.strip()}
        found = [r for r in found if r[4] in wanted]

    # Heaviest first: the most-shared connections are the ones worth the
    # screen space, and a person expanding a 500-video creator wants the
    # richest videos, not the lowest row ids.
    found.sort(key=lambda r: -float(r[2] or 0))
    trimmed = found[:limit]
    ids = [r[0] for r in trimmed]

    ring = nodes_by_id(conn, ids)
    every = ids + [node_id]
    return {
        "ok": True,
        "centre": centre,
        "nodes": ring + [centre],
        "edges": edges_among(conn, every),
        "truncated": max(0, len(found) - len(trimmed)),
        "total": len(found),
    }


def overview(conn: sqlite3.Connection, limit: int = 16) -> dict:
    """The opening view: the archive's biggest hubs, with a few videos each.

    Not "the top N nodes" — that is a list of disconnected dots. Hubs plus
    their strongest videos is a graph with structure in it from the first
    frame, and it answers the question a person actually opens this with:
    what is in here, and what is it mostly about?
    """
    limit = max(3, min(int(limit or 16), 60))
    try:
        hubs = [r[0] for r in conn.execute(
            "SELECT id FROM graph_nodes WHERE kind <> 'video' "
            "ORDER BY weight DESC LIMIT ?", (limit,))]
    except sqlite3.Error:
        return {"ok": False, "nodes": [], "edges": [],
                "note": "graph not built"}
    if not hubs:
        return {"ok": True, "nodes": [], "edges": [],
                "note": "no relationships found in this database"}

    ids = list(hubs)
    for hub in hubs:
        try:
            ids += [r[0] for r in conn.execute(
                "SELECT e.src FROM graph_edges e WHERE e.dst = ? "
                "ORDER BY e.weight DESC LIMIT 4", (hub,))]
        except sqlite3.Error:
            continue

    ids = list(dict.fromkeys(ids))
    return {"ok": True, "nodes": nodes_by_id(conn, ids),
            "edges": edges_among(conn, ids), "seeded_from": hubs}


def find(conn: sqlite3.Connection, query: str, limit: int = 30) -> list:
    """Nodes whose label matches, best prefix first.

    Plain LIKE rather than FTS: labels are short, there are tens of thousands
    at most, and a substring match is what a person expects when they type
    half a creator's name.
    """
    q = (query or "").strip()
    if not q:
        return []
    limit = max(1, min(int(limit or 30), 200))
    like = f"%{q.lower()}%"
    starts = f"{q.lower()}%"
    try:
        cur = conn.execute(
            "SELECT id, kind, label, sub, weight, meta FROM graph_nodes "
            "WHERE LOWER(label) LIKE ? "
            "ORDER BY (CASE WHEN LOWER(label) LIKE ? THEN 0 ELSE 1 END), "
            "         weight DESC LIMIT ?", (like, starts, limit))
    except sqlite3.Error:
        return []
    out = []
    for r in cur.fetchall():
        try:
            meta = json.loads(r[5] or "{}")
        except ValueError:
            meta = {}
        out.append({"id": r[0], "kind": r[1], "label": r[2], "sub": r[3],
                    "weight": r[4], "meta": meta})
    return out


def detail(conn: sqlite3.Connection, node_id: str, rows: int = 40) -> dict:
    """Everything the database holds about one node.

    For a dimension node that is its own row plus the videos pointing at it;
    for a tag it is the rows that mentioned it. Both come back with the actual
    column values, because *"all information in database"* means the row, not
    a summary of the row.
    """
    node = _node_row(conn, node_id)
    if not node:
        return {"ok": False, "note": "no such node"}
    rows = max(1, min(int(rows or 40), 200))
    parsed = parse_id(node_id)
    out = {"ok": True, "node": node, "records": [], "videos": []}

    if parsed["kind"] == "dim":
        table = node["meta"].get("table") or parsed.get("table")
        id_col = node["meta"].get("id_column") or "id"
        row_id = node["meta"].get("row_id", parsed.get("row"))
        rec = _rows(conn, f'SELECT * FROM {_q(table)} WHERE {_q(id_col)} = ?',
                    (row_id,), 1)
        out["records"] = [{"table": table, "rows": rec}] if rec else []

    elif parsed["kind"] == "tag":
        table = node["meta"].get("table") or parsed.get("table")
        column = node["meta"].get("column") or parsed.get("column")
        token = node["meta"].get("token") or parsed.get("token")
        rec = _rows(
            conn,
            f'SELECT * FROM {_q(table)} WHERE LOWER({_q(column)}) LIKE ?',
            (f"%{str(token).lower()}%",), rows)
        out["records"] = [{"table": table, "rows": rec}] if rec else []

    elif parsed["kind"] == "video":
        out["video_key"] = parsed["key"]

    # Which videos this node reaches — the answer to "show me the footage".
    if parsed["kind"] != "video":
        try:
            keys = [r[0] for r in conn.execute(
                "SELECT src FROM graph_edges WHERE dst = ? "
                "ORDER BY weight DESC LIMIT ?", (node_id, rows))]
        except sqlite3.Error:
            keys = []
        out["videos"] = _video_cards(conn, [k[2:] for k in keys
                                            if k.startswith("v:")])
    return out


def edge_detail(conn: sqlite3.Connection, src: str, dst: str, rel: str,
                rows: int = 20) -> dict:
    """Why two nodes are connected — the rows that make the edge true.

    The stored `ref` is `table|column[|value]`, which is enough to rebuild the
    query that produced the edge. Clicking a line therefore lands on real data
    rather than on a tooltip repeating what the line already showed.
    """
    try:
        row = conn.execute(
            "SELECT weight, ref FROM graph_edges "
            "WHERE src=? AND dst=? AND rel=?", (src, dst, rel)).fetchone()
    except sqlite3.Error:
        row = None
    if not row:
        return {"ok": False, "note": "no such edge"}

    weight, ref = row[0], row[1] or ""
    parts = ref.split("|")
    a, b = _node_row(conn, src), _node_row(conn, dst)
    out = {"ok": True, "src": a, "dst": b, "rel": rel, "weight": weight,
           "ref": ref, "records": []}

    src_parsed = parse_id(src)
    if src_parsed["kind"] != "video" or len(parts) < 2:
        return out

    table, column = parts[0], parts[1]
    key_col = reflect.key_column(reflect.columns(conn, table))
    if not key_col:
        return out
    vkey = src_parsed["key"]
    rows = max(1, min(int(rows or 20), 100))

    if len(parts) >= 3 and not parts[2].startswith("#") and \
            b.get("kind") == "dim":
        found = _rows(conn,
                      f'SELECT * FROM {_q(table)} WHERE {_q(key_col)} = ? '
                      f'AND {_q(column)} = ?', (vkey, parts[2]), rows)
        if not found:                      # keys are stored as tg1234 as well
            found = _rows(conn,
                          f'SELECT * FROM {_q(table)} WHERE {_q(key_col)} '
                          f'LIKE ? AND {_q(column)} = ?',
                          (f"%{vkey}", parts[2]), rows)
    else:
        needle = parts[2] if len(parts) >= 3 else ""
        found = _rows(conn,
                      f'SELECT * FROM {_q(table)} WHERE {_q(key_col)} = ? '
                      f'AND LOWER({_q(column)}) LIKE ?',
                      (vkey, f"%{needle.lstrip('#').lower()}%"), rows)
        if not found:
            found = _rows(conn,
                          f'SELECT * FROM {_q(table)} WHERE {_q(key_col)} '
                          f'LIKE ? AND LOWER({_q(column)}) LIKE ?',
                          (f"%{vkey}", f"%{needle.lstrip('#').lower()}%"), rows)
    if found:
        out["records"] = [{"table": table, "rows": found}]
    return out


def path(conn: sqlite3.Connection, a: str, b: str, max_depth: int = 6) -> dict:
    """The shortest chain of relationships between two nodes.

    Breadth-first from both ends at once. Two-sided search matters here because
    a video hub has thousands of neighbours: one-sided BFS explores the whole
    archive before reaching depth three, while meeting in the middle keeps each
    frontier small enough to answer instantly.
    """
    if a == b:
        return {"ok": True, "path": nodes_by_id(conn, [a]), "edges": []}
    if not _node_row(conn, a) or not _node_row(conn, b):
        return {"ok": False, "note": "unknown node"}

    def step(node):
        try:
            return [r[0] for r in conn.execute(
                "SELECT dst FROM graph_edges WHERE src = ? "
                "UNION SELECT src FROM graph_edges WHERE dst = ?",
                (node, node))]
        except sqlite3.Error:
            return []

    front_a, front_b = {a: [a]}, {b: [b]}
    seen_a, seen_b = dict(front_a), dict(front_b)
    for _depth in range(max_depth):
        # Always grow the smaller frontier: that is what keeps the search from
        # degenerating into the one-sided version.
        if len(front_a) <= len(front_b):
            nxt = {}
            for node, chain in front_a.items():
                for peer in step(node):
                    if peer in seen_a:
                        continue
                    trail = chain + [peer]
                    if peer in seen_b:
                        full = trail[:-1] + list(reversed(seen_b[peer]))
                        return _path_result(conn, full)
                    seen_a[peer] = trail
                    nxt[peer] = trail
            front_a = nxt
            if not front_a:
                break
        else:
            nxt = {}
            for node, chain in front_b.items():
                for peer in step(node):
                    if peer in seen_b:
                        continue
                    trail = chain + [peer]
                    if peer in seen_a:
                        full = seen_a[peer] + list(reversed(trail))[1:]
                        return _path_result(conn, full)
                    seen_b[peer] = trail
                    nxt[peer] = trail
            front_b = nxt
            if not front_b:
                break
    return {"ok": False, "note": "no connection within "
                                 f"{max_depth} steps"}


def _path_result(conn: sqlite3.Connection, chain: list) -> dict:
    # Hydration comes back in whatever order SQLite feels like, and a path
    # whose nodes are shuffled is not a path. Re-sort into the chain's order so
    # a caller reading `nodes` gets the same story as one reading `path`.
    rank = {nid: i for i, nid in enumerate(chain)}
    found = nodes_by_id(conn, chain)
    found.sort(key=lambda n: rank.get(n["id"], 1e9))
    return {"ok": True, "path": chain, "nodes": found,
            "edges": edges_among(conn, chain)}


def from_keys(conn: sqlite3.Connection, keys: list, limit: int = 24,
              per_video: int = 5) -> dict:
    """A graph built from a set of videos — the bridge from search.

    Running a query and then seeing its results as a graph answers a different
    question than the result list does: not "which videos match" but "what do
    the matching videos have in common". The shared nodes are visible as the
    points several videos hang off.
    """
    keys = [str(k) for k in list(keys)[:max(1, min(int(limit or 24), 120))]]
    if not keys:
        return {"ok": True, "nodes": [], "edges": []}
    ids = [video_id(k) for k in keys]
    ring = []
    for vid in ids:
        try:
            ring += [r[0] for r in conn.execute(
                "SELECT dst FROM graph_edges WHERE src = ? "
                "ORDER BY weight DESC LIMIT ?", (vid, int(per_video)))]
        except sqlite3.Error:
            continue
    every = list(dict.fromkeys(ids + ring))
    return {"ok": True, "nodes": nodes_by_id(conn, every),
            "edges": edges_among(conn, every)}


def schema_graph(conn: sqlite3.Connection) -> dict:
    """The database's own shape as a graph: tables joined by their keys.

    A different object from the data graph and worth having next to it — this
    is the map of the schema Atlas inferred, so a person can see *why* creators
    ended up attached to videos, and spot a table that is not connected to
    anything (which is exactly what an unindexable table looks like).
    """
    nodes, edges = [], []
    for table in reflect.tables(conn):
        cols = reflect.columns(conn, table)
        key = reflect.key_column(cols)
        content = reflect.content_columns(cols)
        start, end = reflect.time_columns(cols)
        nodes.append({
            "id": f"table:{table}", "kind": "table", "label": table,
            "sub": "tables",
            "weight": float(reflect.row_count(conn, table)),
            "meta": {"rows": reflect.row_count(conn, table),
                     "key": key, "start": start, "end": end,
                     "content": content,
                     "columns": [c["name"] for c in cols],
                     "indexed": bool(key and content)},
        })
        for link in reflect.dimension_links(conn, table, cols):
            edges.append({"src": f"table:{table}",
                          "dst": f"table:{link['table']}",
                          "rel": link["local"], "weight": 1,
                          "ref": f"{table}|{link['local']}"})
        if key:
            edges.append({"src": f"table:{table}", "dst": "table:__videos__",
                          "rel": key, "weight": 2, "ref": f"{table}|{key}"})
    if any(e["dst"] == "table:__videos__" for e in edges):
        nodes.append({"id": "table:__videos__", "kind": "anchor",
                      "label": "video key", "sub": "atlas",
                      "weight": float(len(nodes) or 1),
                      "meta": {"note": "the join every table agrees on"}})
    return {"ok": True, "nodes": nodes, "edges": edges}


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _rows(conn: sqlite3.Connection, sql: str, args: tuple,
          limit: int) -> list:
    try:
        cur = conn.execute(sql + f" LIMIT {int(limit)}", args)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]
    except sqlite3.Error:
        return []


def _video_cards(conn: sqlite3.Connection, keys: list) -> list:
    """Enough about each video to render a card and open the player."""
    keys = [k for k in keys if k]
    if not keys:
        return []
    marks = ",".join("?" * len(keys))
    try:
        cur = conn.execute(
            f"SELECT video_key, title, caption, creator, category, duration, "
            f"msg_id, moment_count, likes, created_at, local_path "
            f"FROM video_index WHERE video_key IN ({marks})", keys)
    except sqlite3.Error:
        return []
    names = [d[0] for d in cur.description]
    order = {k: i for i, k in enumerate(keys)}
    out = [dict(zip(names, r)) for r in cur.fetchall()]
    out.sort(key=lambda r: order.get(r["video_key"], 1e9))
    return out


def counts(conn: sqlite3.Connection) -> dict:
    """Node and edge totals, by kind. What the Graph tab shows as its header."""
    out = {"nodes": 0, "edges": 0, "kinds": {}, "groups": []}
    try:
        for kind, n in conn.execute(
                "SELECT kind, COUNT(*) FROM graph_nodes GROUP BY kind"):
            out["kinds"][kind] = n
            out["nodes"] += n
        out["edges"] = conn.execute(
            "SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        out["groups"] = [
            {"sub": r[0], "kind": r[1], "count": r[2]}
            for r in conn.execute(
                "SELECT sub, kind, COUNT(*) c FROM graph_nodes "
                "WHERE sub IS NOT NULL GROUP BY sub, kind ORDER BY c DESC")]
    except sqlite3.Error:
        pass
    return out
