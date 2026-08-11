"""
The Atlas server.

A single FastAPI app with one job: answer questions about the imported database
fast enough that the interface feels local.

Three decisions shape the whole file.

**Connections are per-thread, not per-request.** SQLite objects cannot cross
threads, and FastAPI runs `def` handlers in a worker pool. A thread-local
connection is opened once per worker and reused for the life of the process, so
a search costs no connection setup and keeps its 64 MB page cache warm across
requests. That cache being warm is a large part of why the second search feels
instant.

**Boot is asynchronous and the site is up during it.** The app starts serving
before the channel has been scanned, before the index is built and before the
encoder has loaded. Everything reports its own progress through `/api/status`,
and the interface renders whatever is ready. The alternative — block until the
channel is fully imported — means a blank browser tab for several minutes.

**No handler names a table.** Everything about the shape of the data comes from
`reflect`, so a bundle with new columns is browsable, searchable and displayable
without touching this file.
"""

import json
import os
import sqlite3
import threading
import time

from fastapi import FastAPI, Query, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               Response, StreamingResponse)

from . import config, graph, index, ingest, maps, media, reflect, search
from .tgchannel import log, recent_log

BOOT_T0 = time.time()

_LOCAL = threading.local()
_BOOT = {
    "phase": "starting",     # starting | scanning | indexing | ready | error
    "detail": "",
    "started_at": BOOT_T0,
    "ready_at": 0.0,
    "error": "",
}
_BOOT_LOCK = threading.Lock()


def _boot_set(**kw):
    with _BOOT_LOCK:
        _BOOT.update(kw)


def db() -> sqlite3.Connection:
    """This thread's connection to atlas.db."""
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        conn = ingest.connect()
        ingest.ensure_meta(conn)
        _LOCAL.conn = conn
    return conn


# ══════════════════════════════════════════════════════════════════════════
# BOOT
# ══════════════════════════════════════════════════════════════════════════
def _index_if_stale(conn: sqlite3.Connection, force: bool = False) -> bool:
    """Rebuild the moment index when the data or the schema has moved.

    The fingerprint check is what makes a changed database work without a code
    change: a bundle carrying a new column produces a different schema hash, and
    that alone triggers a rebuild which picks the column up as searchable text.
    """
    try:
        have = conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0]
    except sqlite3.Error:
        have = 0
    stored = ingest.meta_get(conn, "index_fingerprint", "")
    current = reflect.fingerprint(conn)
    if not force and have and stored == current:
        return False
    _boot_set(phase="indexing", detail="building the moment index")
    index.rebuild(conn, embed=True)
    # The graph is derived from the same schema the index just read, so the
    # moment it can go stale is the moment the index does. Rebuilding it here
    # rather than on demand is what keeps opening the Graph tab instant.
    _rebuild_graph(conn)
    return True


def _rebuild_graph(conn: sqlite3.Connection) -> None:
    """Derive the graph. Never fatal — Atlas without a graph is still Atlas."""
    _boot_set(phase="indexing", detail="deriving the relationship graph")
    try:
        graph.rebuild(conn)
    except Exception as e:                                  # noqa: BLE001
        log(f"graph build failed — {type(e).__name__}: {e}", "WARN")


def _boot() -> None:
    """Bring Atlas up in the order that makes the site useful soonest."""
    conn = ingest.connect()
    ingest.ensure_meta(conn)
    index.ensure_schema(conn)
    graph.ensure_schema(conn)
    maps.ensure_schema(conn)

    # Sparse files are only usable while the process that built them remembers
    # which chunks landed, so last run's leftovers are dropped before anything
    # can read a hole as video data.
    stale = media.sweep_sparse()
    if stale:
        log(f"cleared {stale} partial video file(s) from the last run")

    # Anything already imported is searchable before the network is touched.
    try:
        held = conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0]
    except sqlite3.Error:
        held = 0
    if held:
        log(f"resuming with {held} passage(s) already indexed")
        search.reload_vectors()
        _boot_set(phase="ready", detail=f"{held} passage(s) from a previous run",
                  ready_at=time.time())

    missing = config.missing_secrets()
    if os.environ.get("ATLAS_NO_SCAN") == "1":
        note = "started with --no-scan; the channel was not contacted"
        log(note)
        _boot_set(phase="ready", detail=note, ready_at=time.time())
    elif missing:
        note = ("Telegram credentials missing: " + ", ".join(missing) +
                ". Atlas can serve what is already imported but cannot reach "
                "the channel.")
        log(note, "WARN")
        _boot_set(phase="ready" if held else "error", detail=note,
                  error="" if held else note, ready_at=time.time())
    else:
        _boot_set(phase="scanning", detail="scanning the channel for bundles")

        def after_bundle(bundle_conn, result):
            # Re-index after each bundle rather than only at the end, so the
            # first bundle is searchable while the rest are still downloading.
            try:
                _index_if_stale(bundle_conn, force=True)
            except Exception as e:
                log(f"index after bundle {result.get('seq')} failed — {e}")

        ingest.scan_and_import(full=True, on_bundle=after_bundle)

    try:
        _index_if_stale(conn)
    except Exception as e:
        log(f"index build failed — {type(e).__name__}: {e}")
        _boot_set(error=f"{type(e).__name__}: {e}")

    # An index carried over from an earlier run is not proof of a graph: this
    # database may predate the graph tables entirely. Deriving it costs a few
    # seconds and only happens when there is genuinely nothing stored.
    try:
        if not graph.counts(conn)["nodes"]:
            _rebuild_graph(conn)
    except Exception as e:                                  # noqa: BLE001
        log(f"graph check failed — {type(e).__name__}: {e}", "WARN")

    search.reload_vectors()
    try:
        moments = conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0]
        videos = conn.execute("SELECT COUNT(*) FROM video_index").fetchone()[0]
    except sqlite3.Error:
        moments = videos = 0
    _boot_set(phase="ready", ready_at=time.time(),
              detail=f"{moments} passage(s) across {videos} video(s)")
    log(f"Atlas ready in {time.time() - BOOT_T0:.1f}s — "
        f"{moments} passage(s), {videos} video(s)")
    conn.close()

    # Last, because it competes for the same CPU as the index build and nobody
    # is typing yet.
    try:
        from .encoder import warm
        warm()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Atlas", docs_url=None, redoc_url=None)


class _Timing:
    """Stamp X-Atlas-Ms without ever standing between the body and the socket.

    This was `@app.middleware("http")`, which is `BaseHTTPMiddleware`, which
    wraps `send` to count what the endpoint emits and compares the total against
    the `Content-Length` it saw in `http.response.start`. Every video request
    here is a byte range with an exact Content-Length from `media.range_plan`,
    and a browser seeking mid-playback closes the connection before the range
    finishes — a completely normal thing for a player to do. The accounting then
    raised `RuntimeError: Response content shorter than Content-Length` from
    inside an anyio ExceptionGroup, several frames deep, naming nothing.

    Pure ASGI has no such accounting. The only thing this needs is one header on
    the response-start message, so it rewrites that message and passes every
    other one straight through, untouched and uncounted.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter()

        async def _send(message):
            if message.get("type") == "http.response.start":
                ms = (time.perf_counter() - t0) * 1000
                headers = list(message.get("headers") or [])
                headers.append((b"x-atlas-ms", f"{ms:.1f}".encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, _send)


app.add_middleware(_Timing)


def start_boot() -> None:
    threading.Thread(target=_boot, name="atlas-boot", daemon=True).start()


# ── the page ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    path = os.path.join(config.WEB_DIR, "index.html")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except OSError as e:
        return HTMLResponse(f"<h1>Atlas</h1><p>Interface missing: {e}</p>",
                            status_code=500)


@app.get("/atlas.css")
def css():
    return FileResponse(os.path.join(config.WEB_DIR, "atlas.css"),
                        media_type="text/css")


@app.get("/atlas.js")
def js():
    return FileResponse(os.path.join(config.WEB_DIR, "atlas.js"),
                        media_type="application/javascript")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/sitemap.js")
def sitemap_js_standalone():
    """The shared footer, for when Atlas runs on its own port.

    Mounted under the main server, `/sitemap.js` is answered by the parent app
    and never reaches here. Standalone (`atlas_boot.py`) there is no parent, so
    Atlas serves the same file itself rather than 404-ing its own footer away.
    """
    try:
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from sitemap import sitemap_js          # noqa: PLC0415
        return Response(sitemap_js(), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})
    except Exception:
        # A missing footer is not worth a 500 on a page that works fine.
        return Response("", media_type="application/javascript")


# ── state ─────────────────────────────────────────────────────────────────
@app.get("/api/status")
def api_status():
    """One call the interface polls for everything that changes."""
    conn = db()
    with _BOOT_LOCK:
        boot = dict(_BOOT)
    boot["elapsed"] = round(time.time() - boot["started_at"], 1)

    try:
        bundles = conn.execute(
            "SELECT COUNT(*) FROM bundles WHERE status='ok'").fetchone()[0]
    except sqlite3.Error:
        bundles = 0

    return {
        "boot": boot,
        "ingest": ingest.status(),
        "index": index.status(),
        "search": search.stats(conn),
        "graph": graph.counts(conn),
        "map": maps.status(),
        "bundles": bundles,
        "cache": media.cache_stats(),
        "telegram": {"configured": config.telegram_ready(),
                     "missing": config.missing_secrets(),
                     "channel": config.CHANNEL_ID},
    }


@app.get("/api/channel")
def api_channel():
    from . import tgchannel
    return tgchannel.probe()


@app.get("/api/log")
def api_log(limit: int = 120):
    return {"lines": recent_log(limit)}


@app.post("/api/scan")
def api_scan(full: bool = True, max_messages: int = 0):
    """Re-scan the channel. Safe to call repeatedly — imports are merges."""
    def after(conn, result):
        try:
            _index_if_stale(conn, force=True)
        except Exception as e:
            log(f"post-bundle index failed — {e}")

    started = ingest.scan_in_background(full=full, max_messages=max_messages,
                                        on_bundle=after)
    return {"ok": started,
            "note": "" if started else "a scan is already running"}


@app.post("/api/reindex")
def api_reindex(embed: bool = True):
    result = index.rebuild(db(), embed=embed)
    if result.get("ok"):
        search.clear_cache()
    return result


# ── search ────────────────────────────────────────────────────────────────
@app.get("/api/search")
def api_search(q: str = Query("", description="natural language query"),
               limit: int = 24, offset: int = 0, source: str = "",
               video: str = "", prefetch: bool = True):
    """The moment search.

    Prefetch fires from here rather than from the browser: the server already
    knows which videos won, and starting the transfers now buys the few hundred
    milliseconds a person spends reading the first result.
    """
    conn = db()
    sources = [s for s in source.split(",") if s] if source else None
    out = search.search(conn, q, limit=limit, offset=offset, sources=sources,
                        video_key=video or None)
    if prefetch and out.get("results"):
        media.prefetch_async(config.DB_PATH,
                             [r["video_key"] for r in out["results"]])
    return out


@app.get("/api/suggest")
def api_suggest(q: str = "", limit: int = 8):
    return {"suggestions": search.suggestions(db(), q, limit)}


@app.get("/api/similar/{video_key}")
def api_similar(video_key: str, limit: int = 12):
    return {"results": search.similar(db(), video_key, limit)}


# ── library ───────────────────────────────────────────────────────────────
_SORTS = {
    "recent":   "COALESCE(created_at, 0) DESC, video_key DESC",
    "oldest":   "COALESCE(created_at, 0) ASC, video_key ASC",
    "richest":  "moment_count DESC",
    "longest":  "COALESCE(duration, 0) DESC",
    "shortest": "CASE WHEN COALESCE(duration,0) > 0 THEN duration END ASC",
    "liked":    "COALESCE(likes, 0) DESC",
}


def _resident_clause(conn: sqlite3.Connection) -> str:
    """A SQL clause matching videos that are playable without a download.

    Residency is a fact about the disk, not the database, so it cannot be
    expressed in SQL directly. Loading the keys into a per-connection temp
    table lets the filter, the count and the paging all agree — which an
    `IN (…)` list of twenty thousand terms would not survive.
    """
    keys = media.resident_keys(conn)
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS resident(k TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM resident")
    conn.executemany("INSERT OR IGNORE INTO resident(k) VALUES (?)",
                     [(k,) for k in keys])
    return "video_key IN (SELECT k FROM resident)"


def _keys_matching(conn: sqlite3.Connection, q: str, cap: int = 800) -> list:
    """Video keys whose indexed moments match `q`.

    The library filter is a browse, not a ranked search, so this asks the FTS
    index for the *set* of videos that contain the words and throws the scores
    away. Capped, because the filter is meant to narrow a grid — a query that
    matches most of the archive has not narrowed anything, and the metadata
    clauses beside it still apply.
    """
    if not q.strip():
        return []
    try:
        hits = search.search(conn, q, limit=cap, offset=0)
    except Exception:                                       # noqa: BLE001
        return []
    keys, seen = [], set()
    for h in hits.get("results", []):
        k = h.get("video_key")
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


@app.get("/api/library")
def api_library(limit: int = 40, offset: int = 0, sort: str = "recent",
                creator: str = "", category: str = "", has: str = "",
                q: str = ""):
    """Browse every video, with the filters the data actually supports."""
    conn = db()
    where, args = [], []
    if creator:
        where.append("creator = ?")
        args.append(creator)
    if category:
        where.append("category = ?")
        args.append(category)
    if has == "speech":
        where.append("has_speech = 1")
    elif has == "narrative":
        where.append("has_narrative = 1")
    elif has == "playable":
        where.append(_resident_clause(conn))
    inside: list = []
    if q:
        # Metadata match, plus every video whose *contents* match. Filtering the
        # library on title alone hides the video that spends thirty seconds on
        # the subject but never names it — which is the whole reason this
        # archive is indexed to the second in the first place.
        clauses = ["LOWER(COALESCE(title,'')) LIKE ?",
                   "LOWER(COALESCE(caption,'')) LIKE ?",
                   "LOWER(COALESCE(creator,'')) LIKE ?"]
        needle = f"%{q.lower()}%"
        args += [needle, needle, needle]
        inside = _keys_matching(conn, q)
        if inside:
            marks = ",".join("?" * len(inside))
            clauses.append(f"video_key IN ({marks})")
            args += inside
        where.append("(" + " OR ".join(clauses) + ")")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    order = _SORTS.get(sort, _SORTS["recent"])
    try:
        total = conn.execute(
            f"SELECT COUNT(*) FROM video_index{clause}", args).fetchone()[0]
        cur = conn.execute(
            f"SELECT * FROM video_index{clause} ORDER BY {order} "
            f"LIMIT ? OFFSET ?", args + [limit, offset])
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
    except sqlite3.Error as e:
        return {"ok": False, "note": str(e), "results": [], "total": 0}

    resident = media.resident_keys(conn)
    inside_set = set(inside)
    needle_l = q.lower().strip()
    for r in rows:
        try:
            r["sources"] = json.loads(r.get("sources") or "{}")
        except (ValueError, TypeError):
            r["sources"] = {}
        r["has_file"] = r.get("video_key") in resident
        if needle_l:
            # Which half of the OR above put this row here. The library says so
            # out loud, because "why is this video in my results?" is a fair
            # question when the word appears nowhere on the card.
            meta_hit = any(needle_l in (r.get(f) or "").lower()
                           for f in ("title", "caption", "creator"))
            r["matched"] = ("both" if meta_hit and r.get("video_key") in inside_set
                            else "meta" if meta_hit else "inside")
        r.pop("local_path", None)
    return {"ok": True, "results": rows, "total": total, "offset": offset,
            "limit": limit, "inside": len(inside)}


@app.get("/api/facets")
def api_facets():
    """The filter values that exist, so the UI never offers an empty filter."""
    conn = db()

    def top(column, n=40):
        try:
            return [{"value": v, "count": c} for v, c in conn.execute(
                f"SELECT {column}, COUNT(*) FROM video_index "
                f"WHERE {column} IS NOT NULL AND {column} <> '' "
                f"GROUP BY {column} ORDER BY COUNT(*) DESC LIMIT ?", (n,))]
        except sqlite3.Error:
            return []

    stats = search.stats(conn)
    return {"creators": top("creator"), "categories": top("category"),
            "sources": stats.get("by_source", {}),
            "totals": {"videos": stats.get("videos", 0),
                       "moments": stats.get("moments", 0)}}


# ── one video, everything known about it ──────────────────────────────────
@app.get("/api/video/{video_key}")
def api_video(video_key: str, full: bool = True):
    """Every fact in the database about one video.

    *"Display all and all information available in database."* The related-rows
    walk below does that literally: every table with a video key is asked for
    this video's rows, whatever those tables happen to be. A bundle carrying a
    table Atlas has never seen shows up here as a section with no code change.
    """
    conn = db()
    key = reflect.normalize_key(video_key)

    cur = conn.execute("SELECT * FROM video_index WHERE video_key = ?", (key,))
    row = cur.fetchone()
    meta = dict(zip([d[0] for d in cur.description], row)) if row else {}
    if meta:
        try:
            meta["sources"] = json.loads(meta.get("sources") or "{}")
        except (ValueError, TypeError):
            meta["sources"] = {}
        meta["has_file"] = media.resident(meta.get("local_path"), key)
        meta.pop("local_path", None)

    moments = []
    try:
        cur = conn.execute(
            "SELECT id, t_start, t_end, source, src_table, text FROM moments "
            "WHERE video_key = ? ORDER BY COALESCE(t_start, -1), id", (key,))
        moments = [dict(zip([d[0] for d in cur.description], r))
                   for r in cur.fetchall()]
    except sqlite3.Error:
        pass

    related = []
    if full:
        for table in reflect.tables(conn):
            cols = reflect.columns(conn, table)
            kcol = reflect.key_column(cols)
            if not kcol:
                continue
            try:
                # The stored key can be `tg1234` or `1234` depending on which
                # side of the pipeline wrote it, and both mean this video.
                cur = conn.execute(
                    f'SELECT * FROM "{table}" WHERE "{kcol}" = ? '
                    f'OR "{kcol}" = ? LIMIT 400', (key, f"tg{key}"))
                names = [d[0] for d in cur.description]
                rows = [dict(zip(names, r)) for r in cur.fetchall()]
            except sqlite3.Error:
                continue
            if rows:
                related.append({"table": table, "key": kcol,
                                "columns": names, "rows": rows})

    playback = media.resolve(conn, key)
    return {"ok": bool(meta or moments), "video_key": key, "meta": meta,
            "moments": moments, "related": related,
            "playback": {"where": playback["where"],
                         "size": playback.get("size", 0),
                         "msg_id": playback.get("msg_id")}}


# ── the raw database ──────────────────────────────────────────────────────
@app.get("/api/schema")
def api_schema(samples: int = 0):
    return reflect.describe(db(), samples=samples)


@app.get("/api/table/{name}")
def api_table(name: str, limit: int = 50, offset: int = 0, q: str = "",
              order: str = "", desc: bool = False):
    """A generic row browser for any table in the bundle.

    The table name is checked against `reflect.tables()` rather than escaped,
    and the sort column against that table's real columns. An allow-list built
    from the live schema is the one form of SQL-injection defence that cannot
    be got subtly wrong, and it costs one lookup.
    """
    conn = db()
    if name not in reflect.tables(conn):
        return JSONResponse({"ok": False, "note": f"no table named {name}"},
                            status_code=404)

    cols = reflect.columns(conn, name)
    col_names = [c["name"] for c in cols]
    where, args = "", []
    if q:
        texty = [c["name"] for c in cols
                 if not c["type"] or "CHAR" in c["type"] or "TEXT" in c["type"]]
        if texty:
            where = " WHERE " + " OR ".join(
                f'LOWER(CAST("{c}" AS TEXT)) LIKE ?' for c in texty)
            args = [f"%{q.lower()}%"] * len(texty)

    order_sql = ""
    if order and order in col_names:
        order_sql = f' ORDER BY "{order}" {"DESC" if desc else "ASC"}'

    try:
        total = conn.execute(
            f'SELECT COUNT(*) FROM "{name}"{where}', args).fetchone()[0]
        cur = conn.execute(
            f'SELECT * FROM "{name}"{where}{order_sql} LIMIT ? OFFSET ?',
            args + [min(int(limit), 500), int(offset)])
        rows = [list(r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        return JSONResponse({"ok": False, "note": str(e)}, status_code=400)

    return {"ok": True, "table": name, "columns": col_names,
            "types": [c["type"] for c in cols], "rows": rows, "total": total,
            "offset": offset, "limit": limit}


@app.get("/api/bundles")
def api_bundles():
    return {"bundles": ingest.bundle_rows(db()),
            "sources": [
                {"table": s["table"], "text": s["text"], "source": s["source"],
                 "key": s["key"], "start": s["start"], "via": s["via"]}
                for s in reflect.text_sources(db())]}


# ── the graph ─────────────────────────────────────────────────────────────
# Every route here answers in one indexed lookup against the two derived
# tables, because the interface calls them on every click and a graph that
# thinks before it expands is a graph nobody explores.
@app.get("/api/graph")
def api_graph(limit: int = 16):
    """The opening view, plus what the whole graph contains."""
    conn = db()
    view = graph.overview(conn, limit=limit)
    view["counts"] = graph.counts(conn)
    view["status"] = graph.status()
    return view


@app.get("/api/graph/expand/{node_id:path}")
def api_graph_expand(node_id: str, limit: int = 0, kind: str = ""):
    """One node's neighbours, and every edge among the result.

    `:path` on the parameter is deliberate: node ids contain colons and may
    contain a slash inside a token, and a starlette path converter takes the
    rest of the URL verbatim rather than stopping at the next segment.
    """
    return graph.neighbors(db(), node_id, limit=limit or graph.FANOUT,
                           kind=kind)


@app.get("/api/graph/node/{node_id:path}")
def api_graph_node(node_id: str, rows: int = 40):
    """Everything the database holds about one node, and the videos it reaches."""
    found = graph.detail(db(), node_id, rows=rows)
    if not found.get("ok"):
        return JSONResponse(found, status_code=404)
    return found


@app.get("/api/graph/edge")
def api_graph_edge(src: str, dst: str, rel: str, rows: int = 20):
    """Why two nodes are connected — the rows that make the edge true."""
    found = graph.edge_detail(db(), src, dst, rel, rows=rows)
    if not found.get("ok"):
        return JSONResponse(found, status_code=404)
    return found


@app.get("/api/graph/find")
def api_graph_find(q: str = "", limit: int = 30):
    return {"ok": True, "results": graph.find(db(), q, limit=limit)}


@app.get("/api/graph/path")
def api_graph_path(a: str, b: str, depth: int = 6):
    """The shortest chain of relationships between two nodes."""
    return graph.path(db(), a, b, max_depth=max(1, min(int(depth), 8)))


@app.get("/api/graph/schema")
def api_graph_schema():
    """The database's own shape: tables joined by the keys Atlas inferred."""
    return graph.schema_graph(db())


@app.get("/api/graph/from")
def api_graph_from(keys: str = "", limit: int = 24, per_video: int = 5):
    """A graph built from a set of videos — what a result page has in common."""
    wanted = [reflect.normalize_key(k) for k in keys.split(",") if k.strip()]
    return graph.from_keys(db(), wanted, limit=limit, per_video=per_video)


@app.post("/api/graph/rebuild")
def api_graph_rebuild():
    conn = db()
    try:
        return graph.rebuild(conn)
    except Exception as e:                                  # noqa: BLE001
        return JSONResponse({"ok": False, "note": f"{type(e).__name__}: {e}"},
                            status_code=500)


# ── media ─────────────────────────────────────────────────────────────────
def _serve_file(path: str, range_header: str) -> StreamingResponse:
    """A 206 out of a complete file on disk. The fastest path there is."""
    media.touch(path)
    plan = media.range_plan(path, range_header)
    return StreamingResponse(
        media.stream(path, plan["start"], plan["end"]),
        status_code=plan["status"], headers=plan["headers"],
        media_type=plan["headers"]["Content-Type"])


@app.get("/api/play/{video_key}")
def api_play(video_key: str, request: Request):
    """Serve the video, whether or not it has been downloaded yet.

    The old design waited for the whole file. A 30 MB reel takes several seconds
    to pull out of Telegram, so the first click either stalled or timed out into
    a 503 — the browser's `<video>` element treats that as a hard failure and
    stops, which is exactly what "the videos are not playing" looked like.

    So nothing waits for a whole file any more. Three tiers, cheapest first:

    1. **On disk.** Harvested locally or cached from an earlier watch. Ordinary
       range serving, no network.
    2. **In the sparse file.** A previous watch already pulled the chunks this
       range needs, even though the rest of the video is still missing. Scrubbing
       backwards and re-opening a video therefore cost nothing.
    3. **Straight from the channel.** `stream_remote` starts at the 1 MiB chunk
       holding the first requested byte and yields it onward as it arrives, so
       the first frame appears after one chunk instead of after the file. Chunks
       are written into the sparse file on the way past, which is how tier 2
       fills in, and a video watched to the end promotes itself into the cache.

    Only when MTProto is unavailable — bot-only credentials, a dead session —
    does this fall back to the old blocking download, because the HTTP Bot API
    cannot stream and 20 MB is all it will hand over.
    """
    conn = db()
    key = reflect.normalize_key(video_key)
    rng = request.headers.get("range", "")

    found = media.resolve(conn, key)
    if found["where"] in ("local", "cache"):
        return _serve_file(found["path"], rng)
    if found["where"] == "missing":
        return JSONResponse(
            {"ok": False, "note": "no Telegram message id for this video"},
            status_code=404)

    plan = {}
    note = ""
    try:
        plan = media.remote_plan(key, found["msg_id"], rng)
    except Exception as e:                      # MTProto down, message gone
        note = f"{type(e).__name__}: {e}"

    if plan:
        part = media.sparse_hit(key, plan["start"], plan["end"])
        if part:
            media.touch(part)
            body = media.stream(part, plan["start"], plan["end"])
        else:
            # Serving this range on demand covers the next few seconds of
            # playback; the background fill covers everything after it, so the
            # seeks that follow are answered from disk instead of costing a new
            # Telegram media session each.
            media.fill(key, found["msg_id"], plan["size"])
            body = media.stream_remote(key, plan["message"], plan["start"],
                                       plan["end"], plan["size"])
        return StreamingResponse(
            body, status_code=plan["status"], headers=plan["headers"],
            media_type=plan["headers"]["Content-Type"])

    media.ensure(conn, key, wait=20.0)
    found = media.resolve(conn, key)
    if found["where"] in ("local", "cache"):
        return _serve_file(found["path"], rng)

    st = media.state(key)
    return JSONResponse(
        {"ok": False, "state": st,
         "note": note or st.get("note") or "still downloading from Telegram"},
        status_code=503, headers={"Retry-After": "3"})


@app.get("/api/media/{video_key}/state")
def api_media_state(video_key: str):
    key = reflect.normalize_key(video_key)
    st = media.state(key)
    st["where"] = media.resolve(db(), key)["where"]
    # A video being streamed through is playing right now even though no file
    # exists yet, so the interface must be able to tell that apart from a
    # download that has not started.
    part = media.stream_progress(key)
    if part:
        st["streamed_bytes"] = part["bytes"]
        if st.get("status") in ("absent", "unknown", ""):
            st["status"] = "streaming"
    return st


@app.post("/api/prefetch")
def api_prefetch(keys: str = "", limit: int = 0):
    """Warm the cache for videos the interface thinks are about to be played."""
    wanted = [reflect.normalize_key(k) for k in keys.split(",") if k.strip()]
    if not wanted:
        return {"ok": True, "started": 0}
    started = media.prefetch(db(), wanted, limit or len(wanted))
    return {"ok": True, "started": started}


@app.get("/api/poster/{video_key}")
def api_poster(video_key: str, t: float = None):
    key = reflect.normalize_key(video_key)
    path = media.poster(db(), key, at=t)
    if not path:
        return Response(status_code=204)
    return FileResponse(path, media_type="image/jpeg", headers={
        "Cache-Control": "public, max-age=604800, immutable"})


@app.post("/api/cache/clear")
def api_cache_clear():
    return media.clear_cache()


# ══════════════════════════════════════════════════════════════════════════
# CLIPS — the preview path
# ══════════════════════════════════════════════════════════════════════════
@app.get("/api/clip/{video_key}")
def api_clip(video_key: str, t: float = 0.0, request: Request = None):
    """The two-second clip covering `t`, as a playable mp4.

    This is what a hovered search result plays. It is a different thing from
    `/api/play`, on purpose: `play` streams the *whole reel* and has to solve
    seeking, buffering and a media session; this hands over one small complete
    file that a `<video>` can start rendering the instant it lands. No range
    logic, no sparse index, no MTProto — the clip is small enough for the Bot
    API's own download endpoint.

    204 when the video has no clip index (captured before assets existed, or
    an asset upload that failed). The interface treats that as "fall back to
    the player" rather than as an error, which is why this is not a 404.
    """
    key = reflect.normalize_key(video_key)
    got = media.clip_fetch(db(), key, max(0.0, float(t or 0.0)))
    if not got:
        return Response(status_code=204)
    rng = (request.headers.get("range", "") if request is not None else "")
    resp = _serve_file(got["path"], rng)
    # The player needs to know where in the reel this clip sits, so a preview
    # can show a real timestamp and a click can seek the full player to it.
    resp.headers["X-Clip-Start"] = f"{got['t0']:.3f}"
    resp.headers["X-Clip-End"] = f"{got['t1']:.3f}"
    resp.headers["X-Clip-Seq"] = str(got["seq"])
    return resp


@app.get("/api/clips/{video_key}")
def api_clips(video_key: str, t0: float = None, t1: float = None):
    """The clip index for one video — what exists, and for which seconds."""
    key = reflect.normalize_key(video_key)
    rows = index.clips_for(db(), key, t0, t1)
    return {"ok": True, "key": key, "count": len(rows),
            "chunk_seconds": (rows[0]["t_end"] - rows[0]["t_start"]
                              if rows else None),
            "clips": [{"seq": r["seq"], "t0": r["t_start"], "t1": r["t_end"],
                       "bytes": r["bytes"]} for r in rows]}


# ══════════════════════════════════════════════════════════════════════════
# THE MAPS — the archive as one picture
# ══════════════════════════════════════════════════════════════════════════
# Three views over one projection: a semantic map, the same points coloured by
# cluster, and a scatter plot of any two numeric columns. The first two need
# the dense index; the third never does, so at least one map always works.
@app.get("/api/map")
def api_map(level: str = "video"):
    """Legend, cluster names, method and readiness — everything but the points."""
    conn = db()
    out = maps.meta(conn, level)
    if not out["count"]:
        # A missing map is a normal state on a fresh archive, not an error: the
        # encoder may still be running. Say which, so the interface can show a
        # progress line instead of an empty canvas with no explanation.
        st = index.status()
        out["note"] = ("the dense index is still building — the map appears "
                       "when it finishes" if st.get("phase") == "embedding"
                       else maps.status().get("detail", ""))
    return out


@app.get("/api/map/points")
def api_map_points(level: str = "video"):
    """The point cloud as a packed binary buffer.

    Binary rather than JSON because this is the one response whose size scales
    with the whole archive. Twelve bytes a point against roughly forty, and the
    browser gets a typed array it can hand straight to the canvas instead of a
    parse pass over a megabyte of text.
    """
    conn = db()
    buf = maps.points_binary(conn, level)
    return Response(buf, media_type="application/octet-stream", headers={
        "X-Map-Count": str(len(buf) // 12),
        "X-Map-Stride": "12",
        "Cache-Control": "no-cache"})


@app.get("/api/map/refs")
def api_map_refs(level: str = "video"):
    """What each point *is*, in the same order as the binary buffer."""
    return maps.refs(db(), level)


@app.get("/api/map/point")
def api_map_point(level: str = "video", ref: str = ""):
    """One dot, fully unpacked — the drill-down that makes a map clickable."""
    found = maps.point(db(), level, ref)
    if not found.get("ok"):
        return JSONResponse(found, status_code=404)
    return found


@app.get("/api/map/region")
def api_map_region(level: str = "video", x0: float = 0.0, y0: float = 0.0,
                   x1: float = 1.0, y1: float = 1.0, limit: int = 500):
    """Everything inside a dragged box — the selection other tabs receive."""
    return maps.region(db(), level, x0, y0, x1, y1, limit)


@app.get("/api/map/cluster/{cluster}")
def api_map_cluster(cluster: int, level: str = "video", limit: int = 30):
    """One cluster: its name, the words behind the name, its most typical members."""
    found = maps.cluster_detail(db(), level, cluster, limit)
    if not found.get("ok"):
        return JSONResponse(found, status_code=404)
    return found


@app.get("/api/map/axes")
def api_map_axes():
    """Every numeric column worth plotting, read from the live schema."""
    return maps.axes(db())


@app.get("/api/map/scatter")
def api_map_scatter(x: str = "duration", y: str = "moment_count",
                    colour: str = "cluster", limit: int = 6000,
                    log_x: bool = False, log_y: bool = False):
    out = maps.scatter(db(), x, y, colour, limit, log_x, log_y)
    if not out.get("ok"):
        return JSONResponse(out, status_code=400)
    return out


@app.post("/api/map/rebuild")
def api_map_rebuild(method: str = "auto"):
    """Refit the projection. `method` forces umap | tsne | pca for comparison."""
    started = maps.start_build(config.DB_PATH, method)
    return {"ok": True, "started": started, "status": maps.status(),
            "note": "" if started else "a map build is already running"}


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def serve(port: int = None, host: str = "0.0.0.0") -> None:
    import uvicorn
    start_boot()
    uvicorn.run(app, host=host, port=port or config.PORT, log_level="warning",
                access_log=False)
