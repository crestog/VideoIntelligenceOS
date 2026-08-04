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

from . import config, index, ingest, media, reflect, search
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
    return True


def _boot() -> None:
    """Bring Atlas up in the order that makes the site useful soonest."""
    conn = ingest.connect()
    ingest.ensure_meta(conn)
    index.ensure_schema(conn)

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


@app.middleware("http")
async def _timing(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Atlas-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return resp


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
    if q:
        where.append("(LOWER(COALESCE(title,'')) LIKE ? OR "
                     "LOWER(COALESCE(caption,'')) LIKE ? OR "
                     "LOWER(COALESCE(creator,'')) LIKE ?)")
        needle = f"%{q.lower()}%"
        args += [needle, needle, needle]

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
    for r in rows:
        try:
            r["sources"] = json.loads(r.get("sources") or "{}")
        except (ValueError, TypeError):
            r["sources"] = {}
        r["has_file"] = r.get("video_key") in resident
        r.pop("local_path", None)
    return {"ok": True, "results": rows, "total": total, "offset": offset,
            "limit": limit}


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


# ── media ─────────────────────────────────────────────────────────────────
@app.get("/api/play/{video_key}")
def api_play(video_key: str, request: Request):
    """Serve the video with byte ranges, fetching it first if it is not here.

    A short blocking wait on the first request is deliberate. A person who just
    clicked would rather the request take a moment than get an error for a file
    that is arriving; the browser's own `<video>` retry would otherwise turn a
    working download into a failed play.
    """
    conn = db()
    key = reflect.normalize_key(video_key)
    found = media.resolve(conn, key)
    if found["where"] not in ("local", "cache"):
        media.ensure(conn, key, wait=25.0)
        found = media.resolve(conn, key)
    if found["where"] not in ("local", "cache"):
        st = media.state(key)
        return JSONResponse(
            {"ok": False, "state": st,
             "note": st.get("note") or "still downloading from Telegram"},
            status_code=503, headers={"Retry-After": "3"})

    plan = media.range_plan(found["path"], request.headers.get("range", ""))
    return StreamingResponse(
        media.stream(found["path"], plan["start"], plan["end"]),
        status_code=plan["status"], headers=plan["headers"],
        media_type=plan["headers"]["Content-Type"])


@app.get("/api/media/{video_key}/state")
def api_media_state(video_key: str):
    key = reflect.normalize_key(video_key)
    st = media.state(key)
    st["where"] = media.resolve(db(), key)["where"]
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
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def serve(port: int = None, host: str = "0.0.0.0") -> None:
    import uvicorn
    start_boot()
    uvicorn.run(app, host=host, port=port or config.PORT, log_level="warning",
                access_log=False)
