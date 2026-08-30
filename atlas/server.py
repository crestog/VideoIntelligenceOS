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

import contextlib
import json
import os
import sqlite3
import threading
import time

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               Response, StreamingResponse)

from . import (config, graph, identity, index, ingest, maps, media, reflect,
               roadmap, search, vsearch)
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
    roadmap.ensure_schema(conn)

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
        search.reload_vectors(expect=ingest.meta_get(conn, "index_build_id", ""))
        _boot_set(phase="ready", detail=f"{held} passage(s) from a previous run",
                  ready_at=time.time())

    # The frame index is loaded from file rather than rebuilt, and it needs no
    # `expect` guard: it is keyed by `(video_key, frame_idx)`, which no reindex
    # reassigns, so last session's file is still valid against this session's
    # database. Frame-to-frame image search therefore works before the channel
    # is contacted, and it needs no model at all.
    try:
        vsearch.reload()
    except Exception as e:                                  # noqa: BLE001
        log(f"image index reload failed — {type(e).__name__}: {e}", "WARN")

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
            #
            # Only when the import actually carried rows, though. A forced
            # rebuild is a DELETE of every moment, a full re-INSERT, an FTS5
            # rebuild, a graph derivation and a dense re-embed of every passage;
            # doing that for a shard whose tables were all already held is
            # minutes of work that cannot change a single row. `force=False`
            # still rebuilds when the schema fingerprint moved, so a bundle
            # carrying a new column is picked up either way.
            added = sum(int(v or 0) for v in (result.get("rows") or {}).values())
            try:
                _index_if_stale(bundle_conn, force=bool(added))
            except Exception as e:
                log(f"index after bundle {result.get('seq')} failed — {e}")
            # And the image index, on its own debounce. A shard that carried no
            # vector bytes moves nothing here, so this costs nothing on the
            # common case; `build_if_due` owns how often a rebuild is worth it.
            try:
                vsearch.build_if_due(bundle_conn, result.get("vec_bytes") or 0)
            except Exception as e:                          # noqa: BLE001
                log(f"image index after bundle {result.get('seq')} failed — "
                    f"{type(e).__name__}: {e}")

        ingest.scan_and_import(full=True, on_bundle=after_bundle)

    try:
        _index_if_stale(conn)
    except Exception as e:
        log(f"index build failed — {type(e).__name__}: {e}")
        _boot_set(error=f"{type(e).__name__}: {e}")

    # The end-of-scan build. Forced only when nothing is resident — the first
    # build of a database that already held payloads, or a session whose index
    # files were lost with the scratch disk. With an index already loaded this
    # just flushes whatever bytes the debounce was still holding.
    try:
        vsearch.build_if_due(conn, force=not vsearch.ready())
    except Exception as e:                                  # noqa: BLE001
        log(f"image index build failed — {type(e).__name__}: {e}", "WARN")

    # An index carried over from an earlier run is not proof of a graph: this
    # database may predate the graph tables entirely. Deriving it costs a few
    # seconds and only happens when there is genuinely nothing stored.
    try:
        if not graph.counts(conn)["nodes"]:
            _rebuild_graph(conn)
    except Exception as e:                                  # noqa: BLE001
        log(f"graph check failed — {type(e).__name__}: {e}", "WARN")

    search.reload_vectors(expect=ingest.meta_get(conn, "index_build_id", ""))
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
@contextlib.asynccontextmanager
async def _lifespan(_app):
    """Raise anyio's worker-thread ceiling for the standalone entry point.

    Fifty routes in this file are sync `def`, and FastAPI runs those in anyio's
    default thread pool, which holds 40 tokens for the whole process. A route
    waiting on a SQLite lock (`busy_timeout=60000`) holds its token for the
    whole wait, so a saturated pool queues silently in arrival order and the
    page cannot even fetch its own script. Threads parked on a lock cost a
    stack, so raise it.

    It has to happen inside the loop: anyio stores the limiter in a RunVar
    bound to the running event loop. When this app is *mounted* (ui_server),
    Starlette does not run a sub-app's lifespan and the parent already does
    the same thing — so this is the laptop path, and it never doubles up.
    """
    try:
        import anyio.to_thread
        want = max(40, int(os.environ.get("VIOS_THREADPOOL", "192")))
        limiter = anyio.to_thread.current_default_thread_limiter()
        if want > limiter.total_tokens:
            log(f"worker threads {limiter.total_tokens} → {want}")
            limiter.total_tokens = want
    except Exception as exc:                                # pragma: no cover
        log(f"could not raise the thread ceiling — "
            f"{type(exc).__name__}: {exc}", "WARN")
    yield


app = FastAPI(title="Atlas", docs_url=None, redoc_url=None, lifespan=_lifespan)


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


def boot_phase() -> str:
    """`starting` until something opens Atlas, then scanning → indexing → ready.

    Public because the other planes in this process have to know whether Atlas is
    mid-boot. Its boot is the slowest thing here and the only one the operator is
    watching, so work that competes with it for the channel — the asset backfill
    — waits for `ready` rather than racing it.
    """
    with _BOOT_LOCK:
        return str(_BOOT.get("phase") or "")


# ── the page ──────────────────────────────────────────────────────────────
# The interface is three files, and they are the three files that must never
# fail to arrive: a page that cannot fetch its own script is indistinguishable
# from a page whose script crashed, and both look like "Atlas is broken".
#
# Every other route here is a sync `def`, which FastAPI runs in anyio's thread
# pool — 40 threads for the whole process, shared with the main VIOS server this
# is mounted inside, its polling UIs, and any sqlite call that decides to wait
# 30 seconds on a lock. When that pool is drained, a `def` route does not fail;
# it queues, silently, for as long as it takes. That is what "the Atlas tab just
# says starting and nothing clicks" was: the HTML came from the browser cache
# and atlas.js never arrived.
#
# So these four are `async def` and answer from memory. They need no thread and
# no lock, and they are correct even while the rest of the process is wedged —
# which is exactly when somebody needs the page to load and tell them so.
_WEB_CACHE: dict = {}          # filename → (mtime, size, text/bytes, etag)


def _web_asset(name: str):
    """Read a web file once; re-read only when it changes on disk."""
    path = os.path.join(config.WEB_DIR, name)
    st = os.stat(path)
    sig = (st.st_mtime_ns, st.st_size)
    hit = _WEB_CACHE.get(name)
    if hit and hit[0] == sig:
        return hit[1], hit[2]
    with open(path, "rb") as f:
        raw = f.read()
    etag = '"%s-%x-%x"' % (name, st.st_size, st.st_mtime_ns & 0xFFFFFFFF)
    _WEB_CACHE[name] = (sig, raw, etag)
    return raw, etag


def _asset_response(request: Request, name: str, media_type: str):
    try:
        raw, etag = _web_asset(name)
    except OSError as e:
        return Response(f"/* {name} is missing: {e} */", status_code=500,
                        media_type=media_type)
    # no-store would re-send 250 KB on every tab switch; no-cache with an ETag
    # asks once and gets a 304, so an edited file is picked up on reload without
    # the browser ever serving a stale copy.
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(raw, media_type=media_type, headers=headers)


def page_html(root: str = "") -> str:
    """The interface, told where it is mounted. Raises OSError if it is missing.

    A function rather than only a route body because the parent server serves
    `/atlas` itself — see the note on `home` — and both paths must hand the
    browser byte-identical HTML, or the tab that answers without a redirect and
    the tab that answers after one behave differently for no reason a reader
    could find.
    """
    raw, _ = _web_asset("index.html")
    html = raw.decode("utf-8")
    root = str(root or "").rstrip("/")
    html = html.replace('href="atlas.css"', f'href="{root}/atlas.css"')
    html = html.replace('src="atlas.js"', f'src="{root}/atlas.js"')
    return html.replace(
        "<head>", f'<head>\n<meta name="atlas-base" content="{root}">', 1)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """The page, told where it is mounted.

    `root_path` is '' standalone and '/atlas' mounted. The interface used to
    infer that from its own script URL, which is right until the page is opened
    at `/atlas` with no trailing slash: relative refs then resolve against the
    root, the script's URL says `/atlas.js`, and every API call goes to the
    parent server — which answers `/api/status` with a completely different
    shape and 404s everything else. The page rendered, polled, and did nothing.

    The server is the only participant that knows the answer, so it says it, and
    the asset refs are made absolute for the same reason.
    """
    try:
        html = page_html(request.scope.get("root_path") or "")
    except OSError as e:
        return HTMLResponse(f"<h1>Atlas</h1><p>Interface missing: {e}</p>",
                            status_code=500)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/atlas.css")
async def css(request: Request):
    return _asset_response(request, "atlas.css", "text/css")


@app.get("/atlas.js")
async def js(request: Request):
    return _asset_response(request, "atlas.js", "application/javascript")


@app.get("/favicon.ico")
async def favicon():
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


def rescan(full: bool = True, max_messages: int = 0) -> bool:
    """Re-scan the channel in the background. False if one is already running.

    A function rather than only a route because the other planes in this process
    have reason to call it. The capture plane's asset backfill, in particular,
    writes a `-manifest.json` per video into the channel, and until Atlas reads
    those manifests into `parts` it keeps resolving media the slow way — so the
    thing that just created the fast path is the thing that should say so, rather
    than the clips waiting for the next session's boot scan to be noticed.

    Imports are merges and bundles already held are skipped without being
    downloaded, so calling this repeatedly is cheap and safe.

    Refuses before Atlas has booted. `ingest` allows one scan at a time, so a
    scan started here while Atlas is still unopened would make its own boot scan
    return "already running" and skip — leaving the site to index whatever this
    pass happened to import, with none of the per-bundle re-indexing boot does.
    Nothing is lost by waiting: a boot that has not run yet is a boot that will
    read these manifests anyway.
    """
    if boot_phase() == "starting":
        log("rescan asked for before boot — the boot scan will read it instead")
        return False

    def after(conn, result):
        # Forced only when rows actually landed — see `after_bundle` in _boot().
        added = sum(int(v or 0) for v in (result.get("rows") or {}).values())
        try:
            _index_if_stale(conn, force=bool(added))
        except Exception as e:
            log(f"post-bundle index failed — {e}")
        try:
            vsearch.build_if_due(conn, result.get("vec_bytes") or 0)
        except Exception as e:                              # noqa: BLE001
            log(f"post-bundle image index failed — {type(e).__name__}: {e}")

    return ingest.scan_in_background(full=full, max_messages=max_messages,
                                     on_bundle=after)


@app.post("/api/scan")
def api_scan(full: bool = True, max_messages: int = 0):
    """Re-scan the channel. Safe to call repeatedly — imports are merges."""
    started = rescan(full=full, max_messages=max_messages)
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
               video: str = "", prefetch: bool = True,
               sort: str = "relevance", creator: str = "", category: str = "",
               collection: str = "",
               min_dur: float = None, max_dur: float = None,
               min_hits: int = None):
    """The moment search.

    Prefetch fires from here rather than from the browser: the server already
    knows which videos won, and starting the transfers now buys the few hundred
    milliseconds a person spends reading the first result.

    `sort` is validated against the known set rather than passed through, so a
    typo in a URL falls back to relevance instead of silently ordering by
    nothing.
    """
    conn = db()
    sources = [s for s in source.split(",") if s] if source else None
    if sort not in search.SORTS:
        sort = "relevance"
    out = search.search(conn, q, limit=limit, offset=offset, sources=sources,
                        video_key=video or None, sort=sort,
                        creator=creator or None, category=category or None,
                        collection=collection or None,
                        min_dur=min_dur, max_dur=max_dur, min_hits=min_hits)
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


# ── image search ──────────────────────────────────────────────────────────
# A lane of its own, deliberately not fused into `/api/search`'s rank fusion:
# that fusion mixes BM25 with Atlas's *text* embedding space, and folding a third
# geometry into the same scoring is a recall trade-off worth measuring against a
# working image lane rather than guessing at now.
def _vsearch_reply(got: dict, took: float) -> dict:
    hits = got.get("hits") or []
    return {"ok": True, "count": len(hits), "took_ms": round(took * 1000),
            "hits": hits, "space": got.get("space", ""),
            "query": got.get("query") or {},
            "searched_videos": got.get("searched_videos", 0),
            # Present only when there is nothing to show. Four different causes
            # produce an empty list — no payloads imported, no numpy, no
            # encoder, or a genuinely unlike archive — and a caller cannot tell
            # them apart from the list alone.
            "reason": got.get("reason", "")}


@app.get("/api/vsearch")
def api_vsearch(q: str = Query("", description="text, into the image space"),
                frame: str = Query("", description="<video_key>:<frame_idx>"),
                t: float = Query(None, description="seconds, instead of an idx"),
                space: str = "", limit: int = 40,
                same_video: bool = False):
    """Search frames. `frame=` needs no model; `q=` loads CLIP on first use.

    `frame=<key>:<idx>` is the mode to reach for — the query vector is already
    in the database, so it answers in either space with nothing downloaded. `q=`
    goes through CLIP's text tower into the `clip` space, which is the only
    space a CLIP vector may be compared in.

    `frame=<key>&t=<seconds>` is the same query from a player's point of view:
    the interface has a playhead, not a frame number, and `fps` is a column
    Atlas holds and the browser does not. Converting here keeps the one piece of
    arithmetic that can be wrong in the place that has the data for it.
    """
    t0 = time.time()
    limit = max(1, min(int(limit or 40), 200))
    if frame:
        key, _sep, raw = frame.rpartition(":")
        if not key:
            key, raw = frame, ""
        key = reflect.normalize_key(key)
        if t is not None and not raw:
            idx = vsearch.frame_for_time(db(), key, t)
            if idx is None:
                return {"ok": False, "count": 0, "hits": [],
                        "reason": f"{key} has no frame rate on record, so a "
                                  f"timestamp cannot name a frame — pass "
                                  f"frame={key}:<frame_idx>"}
        else:
            try:
                idx = int(raw or 0)
            except ValueError:
                return {"ok": False, "count": 0, "hits": [],
                        "reason": f"'{frame}' is not <video_key>:<frame_idx>"}
        sp = space or config.VSEARCH_SPACES[0]
        got = vsearch.similar_to_frame(db(), key, idx, space=sp, limit=limit,
                                       same_video=bool(same_video))
    elif q.strip():
        if space and space != "clip":
            return {"ok": False, "count": 0, "hits": [],
                    "reason": "text and image queries are encoded by CLIP, so "
                              "they can only be searched in the clip space"}
        got = vsearch.search_text(db(), q.strip(), limit=limit)
    else:
        return {"ok": False, "count": 0, "hits": [],
                "reason": "pass ?q=<text> or ?frame=<video_key>:<frame_idx>"}
    return _vsearch_reply(got, time.time() - t0)


@app.post("/api/vsearch/image")
async def api_vsearch_image(file: UploadFile = File(...), limit: int = 40):
    """Drop a screenshot, get the frames that look like it.

    Read with a ceiling rather than `await file.read()` unbounded: this is a
    public-facing multipart endpoint and the file only has to survive being
    decoded to 224x224. Eight megabytes is a generous screenshot.
    """
    t0 = time.time()
    data = await file.read(8 * 1024 * 1024)
    if not data:
        return {"ok": False, "count": 0, "hits": [],
                "reason": "the upload was empty"}
    got = vsearch.search_image(db(), data,
                               limit=max(1, min(int(limit or 40), 200)))
    return _vsearch_reply(got, time.time() - t0)


@app.get("/api/vsearch/state")
def api_vsearch_state():
    """Spaces, frame counts, the chosen stride, resident MB and encoder state."""
    return {"ok": True, **vsearch.state()}


@app.post("/api/vsearch/build")
def api_vsearch_build():
    """Rebuild the frame index now. Normally the import loop does this."""
    return {"ok": True, "spaces": vsearch.build(db())}


@app.post("/api/vsearch/warm")
def api_vsearch_warm():
    """Load the query encoder now, so the first text query is not the slow one."""
    ok = vsearch.warm()
    return {"ok": ok, "encoder": vsearch.state()["encoder"]}


@app.get("/api/frame/{video_key}")
def api_frame(video_key: str, i: int = 0, t: float = None):
    """Render one frame of one video, for an image-search result thumbnail.

    Frames are not stored — the processing plane's frame tiers live on its own
    scratch disk and never enter a shard — so this renders from the poster path
    Atlas already has, at the timestamp the frame index implies. `t` may be
    given directly when the caller already has one.
    """
    key = reflect.normalize_key(video_key)
    at = t if t is not None else vsearch.frame_time(db(), key, i)
    path = media.poster(db(), key, at=at)
    if not path:
        return Response(status_code=204)
    return FileResponse(path, media_type="image/jpeg", headers={
        "Cache-Control": "public, max-age=604800, immutable"})


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
                collection: str = "", q: str = ""):
    """Browse every video, with the filters the data actually supports."""
    conn = db()
    where, args = [], []
    if creator:
        where.append("creator = ?")
        args.append(creator)
    if category:
        where.append("category = ?")
        args.append(category)
    if collection:
        # Asked of the membership table, not of the card's `collections` column.
        # The column is JSON and a `LIKE '%steak%'` over it would also match a
        # collection called "steak dinners", which is a different shelf. The
        # subquery is an index seek on `video_collection(collection)` and it is
        # exact, so a video in five collections is found by any of the five and
        # counted once.
        where.append("video_key IN (SELECT video_key FROM video_collection "
                     "WHERE collection = ?)")
        args.append(collection)
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
        # Same decode as the detail route, for the same reason: a grid card shows
        # its collection chips and its "also at message 10" line from these, and
        # a list is the shape to show them in.
        for name in ("aliases", "messages", "collections", "twin_of"):
            try:
                r[name] = json.loads(r.get(name) or "null") or []
            except (ValueError, TypeError):
                r[name] = []
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
    # Collections come from the membership table, so the count is filings and
    # not videos — a video in two collections is counted by both, which is what
    # a shelf label means. Kept beside creators and categories because the
    # interface treats all three the same way: a chip that narrows, never a
    # chip that hides a video from itself.
    collections = [{"value": c["name"], "count": c["videos"]}
                   for c in identity.collection_counts(conn)]
    return {"creators": top("creator"), "categories": top("category"),
            "collections": collections,
            "sources": stats.get("by_source", {}),
            "totals": {"videos": stats.get("videos", 0),
                       "moments": stats.get("moments", 0),
                       "collections": stats.get("collections", 0),
                       "memberships": stats.get("memberships", 0)}}


@app.post("/api/collections/add")
def api_collections_add(collection: str = "", keys: str = ""):
    """File one or more reels under a saved collection.

    Additive by contract, which is `set_collections`'s rule and not this route's
    convenience: a label is a statement that this reel belongs somewhere, and a
    later refresh from the capture ledger must not silently unsay it. There is
    deliberately no remove here — taking a reel off a shelf is a different
    decision from adding it to one, and it needs its own explicit door.

    Only keys the archive already knows are accepted. A membership for a key that
    is not a video is exactly the kind of row that turned thirty reels into
    sixty-two names, so it is refused at the edge rather than resolved later.
    """
    name = (collection or "").strip()
    wanted = [k.strip() for k in (keys or "").split(",") if k.strip()]
    if not name or not wanted:
        return JSONResponse({"ok": False, "error": "collection and keys required"},
                            status_code=400)
    conn = db()
    identity.ensure(conn)
    known = {r[0] for r in conn.execute(
        "SELECT video_key FROM video_index")} if wanted else set()
    unknown = [k for k in wanted if k not in known]
    filed = [k for k in wanted if k in known]
    added = 0
    for key in filed:
        added += identity.set_collections(conn, key, [name], "desktop")
        # `video_index` is the card table and `collections` is derived into it, so
        # the rows just touched are re-derived now rather than left stale until
        # the next full rebuild. The chip has to appear on the card that was
        # selected, or the action reads as having failed.
        conn.execute("UPDATE video_index SET collections = ? WHERE video_key = ?",
                     (json.dumps(identity.collections_for(conn, key)), key))
    conn.commit()
    # The filter reads from this data and the results are cached by their filter
    # set, so a stale entry would answer "not in steak" for a reel just filed
    # there.
    search.clear_cache()
    return {"ok": True, "collection": name, "videos": len(filed),
            "added": added, "unknown": unknown}


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
        # The identity columns hold JSON text because SQLite has no list type.
        # Decoded here, once, rather than in every client: a card saying "also at
        # message 10" should receive `[10, 40]`, not a string to parse. A
        # malformed or empty value degrades to `[]` instead of throwing inside
        # whatever is rendering the page.
        for name in ("aliases", "messages", "collections", "twin_of"):
            try:
                meta[name] = json.loads(meta.get(name) or "null") or []
            except (ValueError, TypeError):
                meta[name] = []
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
                         "via": playback.get("via", "original"),
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

    # rowid comes back alongside the columns so a cell can be pointed at later
    # without guessing a primary key. A WITHOUT ROWID table has none, and then
    # the drill-down falls back to explaining the column and the value.
    has_rowid = True
    try:
        conn.execute(f'SELECT rowid FROM "{name}" LIMIT 1').fetchone()
    except sqlite3.Error:
        has_rowid = False

    select = f'SELECT {"rowid, " if has_rowid else ""}* FROM "{name}"'
    try:
        total = conn.execute(
            f'SELECT COUNT(*) FROM "{name}"{where}', args).fetchone()[0]
        cur = conn.execute(
            f'{select}{where}{order_sql} LIMIT ? OFFSET ?',
            args + [min(int(limit), 500), int(offset)])
        raw = [list(r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        return JSONResponse({"ok": False, "note": str(e)}, status_code=400)

    rowids = [r[0] for r in raw] if has_rowid else []
    rows = [r[1:] for r in raw] if has_rowid else raw

    return {"ok": True, "table": name, "columns": col_names,
            "types": [c["type"] for c in cols], "rows": rows,
            "rowids": rowids, "total": total,
            "offset": offset, "limit": limit}


@app.get("/api/cell")
def api_cell(table: str, column: str, rowid: int = None, value: str = None):
    """What one cell actually is: its meaning, its provenance, its neighbours.

    *"Clicking a cell should show the truth, backend, what does it mean or
    refer to."* So this answers four questions about a single value:

    * **what it is** — the column's inferred role, its declared type, and
      whether search reads it (and as which kind of evidence);
    * **what it refers to** — if the column is a foreign key by naming
      convention, the row it points at, resolved; if it is a video key, the
      reel;
    * **how common it is** — how many rows in this table carry the same value,
      because a value shared by 4,000 rows means something different from one
      that is unique;
    * **who else says it** — every other table carrying the same value in a
      comparably named column.

    Nothing here is configured. It is all read from the live schema, so a
    bundle carrying a table Atlas has never seen still explains itself.
    """
    conn = db()
    if table not in reflect.tables(conn):
        return JSONResponse({"ok": False, "note": f"no table named {table}"},
                            status_code=404)
    cols = reflect.columns(conn, table)
    by_name = {c["name"]: c for c in cols}
    if column not in by_name:
        return JSONResponse({"ok": False, "note": f"{table} has no column {column}"},
                            status_code=404)

    col = by_name[column]
    key = reflect.key_column(cols)
    start, end = reflect.time_columns(cols)
    content = set(reflect.content_columns(cols))
    role = ("key" if column == key else "start" if column == start else
            "end" if column == end else
            "content" if column in content else "field")

    row = {}
    if rowid is not None:
        try:
            cur = conn.execute(f'SELECT * FROM "{table}" WHERE rowid = ?', (rowid,))
            got = cur.fetchone()
            if got:
                row = dict(zip([d[0] for d in cur.description], got))
        except sqlite3.Error:
            row = {}
    if value is None:
        value = row.get(column)

    out = {
        "ok": True, "table": table, "column": column, "value": value,
        "role": role, "type": col["type"] or "TEXT", "pk": col["pk"],
        "indexed": column in content,
        "source": (reflect.source_label(table, column)
                   if column in content else None),
        "row": row, "refers_to": None, "same_value": None, "elsewhere": [],
        "video_key": None,
        # Named so the caller can open the reel at the moment this row is
        # about rather than at its start. A row with no time column has none.
        "time_column": start, "end_column": end,
    }

    # The reel this cell belongs to, if the row names one. This is what turns a
    # row of numbers into something you can watch.
    if key and row.get(key):
        vk = reflect.normalize_key(row[key])
        found = conn.execute(
            "SELECT video_key, title, duration FROM video_index "
            "WHERE video_key = ?", (vk,)).fetchone()
        if found:
            out["video_key"] = found[0]
            out["video"] = {"video_key": found[0], "title": found[1],
                            "duration": found[2]}

    if value in (None, ""):
        return out

    # What it points at, when the name says it points somewhere.
    for link in reflect.dimension_links(conn, table, cols):
        if link["local"] != column:
            continue
        try:
            cur = conn.execute(
                f'SELECT * FROM "{link["table"]}" WHERE "{link["remote"]}" = ? '
                f'LIMIT 1', (value,))
            got = cur.fetchone()
        except sqlite3.Error:
            got = None
        if got:
            out["refers_to"] = {
                "table": link["table"], "on": link["remote"],
                "row": dict(zip([d[0] for d in cur.description], got)),
            }
        break

    # How many rows here say the same thing.
    try:
        out["same_value"] = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" = ?',
            (value,)).fetchone()[0]
    except sqlite3.Error:
        out["same_value"] = None

    # And who else says it. Only same-named columns are checked: matching a
    # value across unrelated columns would pair a like count with a msg id.
    norm = reflect._norm(column)                                # noqa: SLF001
    for other in reflect.tables(conn):
        if other == table:
            continue
        for oc in reflect.columns(conn, other):
            if reflect._norm(oc["name"]) != norm:              # noqa: SLF001
                continue
            try:
                n = conn.execute(
                    f'SELECT COUNT(*) FROM "{other}" WHERE "{oc["name"]}" = ?',
                    (value,)).fetchone()[0]
            except sqlite3.Error:
                continue
            if n:
                out["elsewhere"].append(
                    {"table": other, "column": oc["name"], "rows": n})
            break
    out["elsewhere"].sort(key=lambda x: -x["rows"])
    return out


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
        out = graph.rebuild(conn)
        # Every plan was derived from the graph that just went away.
        roadmap.invalidate()
        return out
    except Exception as e:                                  # noqa: BLE001
        return JSONResponse({"ok": False, "note": f"{type(e).__name__}: {e}"},
                            status_code=500)


# ── the roadmap ───────────────────────────────────────────────────────────
# The same graph, ordered. Building a plan costs a group-by over the edge table
# plus one FTS lookup per step, so it is cached against the graph's own size and
# a repeat request is a dictionary hit — which is what makes typing a goal feel
# like filtering rather than like a job.
@app.get("/api/roadmap")
def api_roadmap(goal: str = "", breadth: int = 0, min_support: int = 0):
    """A watch order for the archive, or for the part of it a goal names."""
    try:
        return roadmap.plan(db(), goal,
                            breadth=breadth or roadmap.BREADTH,
                            min_support=min_support or roadmap.MIN_SUPPORT)
    except Exception as e:                                  # noqa: BLE001
        return JSONResponse({"ok": False, "note": f"{type(e).__name__}: {e}"},
                            status_code=500)


@app.get("/api/roadmap/step/{step_id:path}")
def api_roadmap_step(step_id: str, goal: str = "", limit: int = 0):
    """One step in full. `:path` for the same reason the graph routes use it."""
    found = roadmap.step(db(), step_id, goal=goal,
                         limit=limit or roadmap.MOMENTS_IN_STEP)
    if not found.get("ok"):
        return JSONResponse(found, status_code=404)
    return found


@app.get("/api/roadmap/goals")
def api_roadmap_goals(limit: int = 14):
    """Goals worth offering before anything is typed."""
    return {"ok": True, "goals": roadmap.suggest(db(), limit=max(1, min(limit, 60)))}


@app.get("/api/roadmap/progress")
def api_roadmap_progress():
    conn = db()
    return {"ok": True, "progress": roadmap.progress(conn),
            "counts": roadmap.counts(conn)}


@app.post("/api/roadmap/progress")
def api_roadmap_mark(step_id: str = "", state: str = "", goal: str = "",
                     clear: bool = False):
    """Tick a step, skip it, clear one, or clear the lot."""
    conn = db()
    out = roadmap.clear(conn) if clear else roadmap.mark(conn, step_id, state,
                                                        goal=goal)
    if not out.get("ok"):
        return JSONResponse(out, status_code=400)
    return out


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

    Every tier below `resolve` is keyed by `found["cache_key"]`, not by `key`.
    When the plane has uploaded a 480p `+faststart` proxy that is what plays,
    and it is a different file with a different length — so the sparse file,
    the window arithmetic and the background fill all have to agree on which
    file they are talking about. Mixing the two would compute a range against
    the proxy's size and serve chunks of the original into it, which plays as
    corruption rather than as an error.
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

    ck = found.get("cache_key") or key
    plan = {}
    note = ""
    try:
        plan = media.remote_plan(ck, found["msg_id"], rng)
    except Exception as e:                      # MTProto down, message gone
        note = f"{type(e).__name__}: {e}"

    if plan:
        part = media.sparse_hit(ck, plan["start"], plan["end"])
        if part:
            media.touch(part)
            body = media.stream(part, plan["start"], plan["end"])
        else:
            # Serving this range on demand covers the next few seconds of
            # playback; the background fill covers everything after it, so the
            # seeks that follow are answered from disk instead of costing a new
            # Telegram media session each.
            media.fill(ck, found["msg_id"], plan["size"])
            body = media.stream_remote(ck, plan["message"], plan["start"],
                                       plan["end"], plan["size"])
        return StreamingResponse(
            body, status_code=plan["status"], headers=plan["headers"],
            media_type=plan["headers"]["Content-Type"])

    media.ensure(conn, key, wait=20.0)
    found = media.resolve(conn, key)
    if found["where"] in ("local", "cache"):
        return _serve_file(found["path"], rng)

    st = media.state(found.get("cache_key") or key)
    return JSONResponse(
        {"ok": False, "state": st,
         "note": note or st.get("note") or "still downloading from Telegram"},
        status_code=503, headers={"Retry-After": "3"})


@app.get("/api/media/{video_key}/state")
def api_media_state(video_key: str):
    key = reflect.normalize_key(video_key)
    found = media.resolve(db(), key)
    ck = found.get("cache_key") or key
    st = media.state(ck)
    st["where"] = found["where"]
    st["via"] = found.get("via", "original")
    # A video being streamed through is playing right now even though no file
    # exists yet, so the interface must be able to tell that apart from a
    # download that has not started.
    part = media.stream_progress(ck)
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
