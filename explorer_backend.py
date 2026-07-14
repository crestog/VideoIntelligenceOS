"""
VIOS Explorer Backend — Search, Database Explorer, Pipeline & Queue APIs

Everything the v17 UI's Search tabs, Pipeline dashboard, and Database
explorer (with per-chunk Data Inspector and interactive graph) talk to.

Design principles:
  • Zero-error: every endpoint returns structured JSON, never a traceback.
  • Graceful degradation: an offline store reports {"status": "offline"}.
  • Conflict-safe: mutating queue ops take Redis SET-NX locks → 409 if busy.
  • Fast: blocking store probes run in threads with hard timeout budgets.
"""

import os
import json
import time
import asyncio
import sqlite3

import aiosqlite
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import DB_PATH, VIDEO_DIR, SQLITE_TIMEOUT, PREVIEW_DIR_NAME
from config import env_check_masked
from db_schema import init_sqlite_schema, sanitize_fts_query
from queue_manager import (
    ALL_QUEUES, get_redis, get_queue_metrics,
    replay_dlq, peek_dlq, purge_dlq, peek_processing,
    pause_queue, resume_queue, reap_stale_jobs,
    acquire_op_lock, release_op_lock,
)

explorer_router = APIRouter()

ENGINE_TIMEOUT_SEC = 4.0     # per-engine budget inside unified search
STORE_PROBE_TIMEOUT = 5.0    # per-store budget inside /api/db/overview
RRF_K = 60                   # standard reciprocal-rank-fusion constant


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def _err(message, status=400):
    return JSONResponse({"error": str(message)[:300]}, status_code=status)


async def _with_timeout(coro, seconds, fallback):
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return fallback
    except Exception as e:
        fb = dict(fallback) if isinstance(fallback, dict) else {"status": "error"}
        if isinstance(fb, dict):
            fb["reason"] = str(e)[:200]
        return fb


def _table_exists(conn, name):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name = ?", (name,)).fetchone()
    return row is not None


def _video_title_map(msg_ids):
    """Batch-fetch titles/thumbs for a set of msg_ids (sync, cheap)."""
    if not msg_ids:
        return {}
    out = {}
    try:
        conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        qmarks = ",".join("?" * len(msg_ids))
        for r in conn.execute(
                f"SELECT msg_id, title, thumb, duration_str FROM videos WHERE msg_id IN ({qmarks})",
                list(msg_ids)):
            out[r[0]] = {"title": r[1] or f"Video #{r[0]}", "thumb": r[2],
                         "duration_str": r[3]}
        conn.close()
    except Exception:
        pass
    return out


# ═══════════════════════════════════════════════════════════
# EMBED RPC — cross-process text embedding via Redis
# (model_manager owns the GPU models; we must NOT import it here)
# ═══════════════════════════════════════════════════════════
RPC_QUEUE = "VIOS_RPC"
RPC_RESULT_PREFIX = "VIOS_RPC_RESULT:"


def rpc_call_sync(kind, payload, timeout=8.0):
    """Push an RPC job for model_manager and poll for the result.
    Never raises — a dead Redis returns a clean error dict."""
    import uuid as _uuid
    try:
        r = get_redis()
        req_id = _uuid.uuid4().hex
        r.lpush(RPC_QUEUE, json.dumps({"id": req_id, "kind": kind, "payload": payload}))
        deadline = time.time() + timeout
        key = RPC_RESULT_PREFIX + req_id
        while time.time() < deadline:
            raw = r.get(key)
            if raw:
                r.delete(key)
                try:
                    return json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    return {"ok": False, "error": "corrupt RPC result"}
            time.sleep(0.05)
        return {"ok": False, "error": "model_manager RPC timeout (is it running?)"}
    except Exception as e:
        return {"ok": False, "error": f"queue broker unreachable: {str(e)[:120]}"}


def _semantic_search_sync(q, k=10, mode="siglip"):
    """Embed the query via RPC, then search Qdrant. Fully guarded."""
    collection_map = {
        "siglip": "frames_siglip",
        "clip": "frames_clip",
        "bge": "chunks_bge",
    }
    if mode not in collection_map:
        return {"status": "error", "reason": f"unknown mode '{mode}'", "results": []}

    rpc = rpc_call_sync("embed_text", {"text": q, "mode": mode}, timeout=10.0)
    if not rpc.get("ok"):
        return {"status": "offline", "reason": rpc.get("error", "embed failed"), "results": []}
    query_vec = rpc.get("vector")
    if not query_vec:
        return {"status": "offline", "reason": "empty embedding", "results": []}

    try:
        from tripartite_db import get_qdrant
        qdrant = get_qdrant()
        if qdrant is None:
            return {"status": "offline", "reason": "Qdrant unavailable", "results": []}
        hits = qdrant.search(collection_name=collection_map[mode],
                             query_vector=query_vec, limit=max(1, min(int(k), 50)))
        results = [{
            "score": round(float(h.score), 4),
            "video_path": (h.payload or {}).get("video_path", ""),
            "msg_id": (h.payload or {}).get("msg_id"),
            "timestamp": (h.payload or {}).get("timestamp") or (h.payload or {}).get("start_t"),
            "frame_idx": (h.payload or {}).get("frame_idx"),
            "chunk_id": (h.payload or {}).get("chunk_id"),
        } for h in hits]
        return {"status": "ok", "results": results}
    except Exception as e:
        return {"status": "offline", "reason": str(e)[:200], "results": []}


# ═══════════════════════════════════════════════════════════
# SEARCH — MOMENTS (FTS5 keyword search, LIKE fallback)
# ═══════════════════════════════════════════════════════════
def _moment_search_sync(q, limit=30):
    safe_q = sanitize_fts_query(q)
    if not safe_q:
        return {"status": "ok", "results": []}
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    try:
        results = []
        if _table_exists(conn, "moments_search"):
            try:
                rows = conn.execute(
                    """SELECT msg_id, ts_sec, source,
                              snippet(moments_search, 3, '<mark>', '</mark>', '…', 18),
                              bm25(moments_search)
                       FROM moments_search WHERE moments_search MATCH ?
                       ORDER BY bm25(moments_search) LIMIT ?""",
                    (safe_q, max(1, min(int(limit), 100)))).fetchall()
                for msg_id, ts, source, snip, rank in rows:
                    results.append({"msg_id": msg_id, "timestamp": ts, "source": source,
                                    "snippet": snip, "rank": round(float(rank), 3)})
                return {"status": "ok", "results": results}
            except sqlite3.OperationalError:
                pass  # fall through to LIKE
        # Fallback: LIKE over transcripts + frame_notes (FTS missing/broken)
        like = f"%{q.strip()[:80]}%"
        if _table_exists(conn, "transcripts"):
            for msg_id, ts, text in conn.execute(
                    "SELECT msg_id, start_sec, text FROM transcripts WHERE text LIKE ? LIMIT ?",
                    (like, limit)).fetchall():
                results.append({"msg_id": msg_id, "timestamp": ts, "source": "speech",
                                "snippet": (text or "")[:200], "rank": 0})
        if _table_exists(conn, "frame_notes"):
            for msg_id, ts, desc in conn.execute(
                    "SELECT msg_id, ts_sec, description FROM frame_notes WHERE description LIKE ? LIMIT ?",
                    (like, limit)).fetchall():
                results.append({"msg_id": msg_id, "timestamp": ts, "source": "visual",
                                "snippet": (desc or "")[:200], "rank": 0})
        return {"status": "ok", "results": results[:limit], "fallback": "LIKE"}
    except Exception as e:
        return {"status": "error", "reason": str(e)[:200], "results": []}
    finally:
        conn.close()


def _enrich_moments(results):
    titles = _video_title_map({r["msg_id"] for r in results if r.get("msg_id")})
    for r in results:
        meta = titles.get(r.get("msg_id"), {})
        r["title"] = meta.get("title", f"Video #{r.get('msg_id')}")
        r["thumb"] = meta.get("thumb")
    return results


@explorer_router.get("/api/search/moments")
async def api_search_moments(q: str = "", limit: int = 30):
    if not q.strip():
        return _err("query 'q' is required")
    data = await asyncio.to_thread(_moment_search_sync, q, limit)
    data["results"] = await asyncio.to_thread(_enrich_moments, data.get("results", []))
    data["query"] = q
    return data


# ═══════════════════════════════════════════════════════════
# SEARCH — SEMANTIC (Qdrant via embed RPC)
# ═══════════════════════════════════════════════════════════
@explorer_router.get("/api/search/semantic")
async def api_search_semantic(q: str = "", k: int = 10, mode: str = "siglip"):
    if not q.strip():
        return _err("query 'q' is required")
    data = await asyncio.to_thread(_semantic_search_sync, q, k, mode)
    data["results"] = await asyncio.to_thread(_enrich_moments, data.get("results", []))
    data["query"] = q
    data["mode"] = mode
    return data


# ═══════════════════════════════════════════════════════════
# SEARCH — GRAPH (Neo4j entities)
# ═══════════════════════════════════════════════════════════
def _graph_search_sync(q, limit=25):
    try:
        from tripartite_db import graph_entity_search
        return graph_entity_search(q, limit)
    except Exception as e:
        return {"status": "offline", "reason": str(e)[:200], "results": []}


@explorer_router.get("/api/search/graph")
async def api_search_graph(q: str = "", limit: int = 25):
    if not q.strip():
        return _err("query 'q' is required")
    data = await asyncio.to_thread(_graph_search_sync, q, limit)
    data["query"] = q
    return data


# ═══════════════════════════════════════════════════════════
# SEARCH — UNIFIED (parallel fan-out + Reciprocal Rank Fusion)
# ═══════════════════════════════════════════════════════════
@explorer_router.get("/api/search/unified")
async def api_search_unified(q: str = "", limit: int = 20):
    if not q.strip():
        return _err("query 'q' is required")

    off = {"status": "timeout", "results": []}
    moments_t = _with_timeout(asyncio.to_thread(_moment_search_sync, q, 30),
                              ENGINE_TIMEOUT_SEC, dict(off))
    semantic_t = _with_timeout(asyncio.to_thread(_semantic_search_sync, q, 30, "siglip"),
                               ENGINE_TIMEOUT_SEC * 3, dict(off))  # embed RPC needs longer
    graph_t = _with_timeout(asyncio.to_thread(_graph_search_sync, q, 20),
                            ENGINE_TIMEOUT_SEC, dict(off))
    moments, semantic, graph = await asyncio.gather(
        moments_t, semantic_t, graph_t, return_exceptions=False)

    # ── Reciprocal Rank Fusion, deduped by (msg_id, 5s time bucket) ──
    fused = {}

    def _feed(engine, items, get_key, get_base):
        for rank, item in enumerate(items):
            key = get_key(item)
            if key is None:
                continue
            entry = fused.setdefault(key, {"engines": [], "rrf": 0.0, **get_base(item)})
            entry["rrf"] += 1.0 / (RRF_K + rank + 1)
            if engine not in entry["engines"]:
                entry["engines"].append(engine)
            # Prefer a richer snippet if we don't have one yet
            if not entry.get("snippet") and item.get("snippet"):
                entry["snippet"] = item["snippet"]

    def _bucket(msg_id, ts):
        if msg_id is None:
            return None
        try:
            return (int(msg_id), int(float(ts or 0) // 5))
        except (TypeError, ValueError):
            return (msg_id, 0)

    _feed("moments", moments.get("results", []),
          lambda i: _bucket(i.get("msg_id"), i.get("timestamp")),
          lambda i: {"msg_id": i.get("msg_id"), "timestamp": i.get("timestamp"),
                     "snippet": i.get("snippet", ""), "source": i.get("source")})
    _feed("semantic", semantic.get("results", []),
          lambda i: _bucket(i.get("msg_id"), i.get("timestamp")),
          lambda i: {"msg_id": i.get("msg_id"), "timestamp": i.get("timestamp"),
                     "snippet": "", "chunk_id": i.get("chunk_id"),
                     "frame_idx": i.get("frame_idx")})

    graph_hits = []
    for gr in graph.get("results", []):
        ent = gr.get("entity", {})
        for ctx in gr.get("contexts", [])[:3]:
            graph_hits.append({"msg_id": ctx.get("msg_id"),
                               "timestamp": ctx.get("start_t"),
                               "snippet": f"Entity: {ent.get('label', '')}",
                               "entity": ent.get("label")})
    _feed("graph", graph_hits,
          lambda i: _bucket(i.get("msg_id"), i.get("timestamp")),
          lambda i: {"msg_id": i.get("msg_id"), "timestamp": i.get("timestamp"),
                     "snippet": i.get("snippet", ""), "entity": i.get("entity")})

    merged = sorted(fused.values(), key=lambda x: -x["rrf"])[:max(1, min(int(limit), 50))]
    merged = await asyncio.to_thread(_enrich_moments, merged)
    for m in merged:
        m["rrf"] = round(m["rrf"], 5)

    return {
        "query": q,
        "results": merged,
        "engines": {
            "moments": moments.get("status", "error"),
            "semantic": semantic.get("status", "error"),
            "graph": graph.get("status", "error"),
        },
        "engine_counts": {
            "moments": len(moments.get("results", [])),
            "semantic": len(semantic.get("results", [])),
            "graph": len(graph_hits),
        },
    }


# ═══════════════════════════════════════════════════════════
# DATABASE EXPLORER — overview / videos / per-video / per-chunk
# ═══════════════════════════════════════════════════════════
def _sqlite_overview_sync():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        counts = {}
        for table in ("videos", "posts", "transcripts", "frame_notes",
                      "oracle_chunks", "processing_events"):
            counts[table] = (conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                             if _table_exists(conn, table) else 0)
        counts["moments_search"] = (
            conn.execute("SELECT COUNT(*) FROM moments_search").fetchone()[0]
            if _table_exists(conn, "moments_search") else 0)
        size_mb = round(os.path.getsize(DB_PATH) / (1024 * 1024), 1) if os.path.exists(DB_PATH) else 0
        conn.close()
        return {"status": "ok", "counts": counts, "size_mb": size_mb}
    except Exception as e:
        return {"status": "error", "reason": str(e)[:200]}


# NOTE: path is /api/explorer/overview (NOT /api/db/overview) because
# v17_backend already registers /api/db/overview and FastAPI's
# first-registered route would shadow this one.
@explorer_router.get("/api/explorer/overview")
async def api_db_overview():
    from tripartite_db import qdrant_overview, postgres_overview, neo4j_overview
    off = {"status": "timeout"}
    sqlite_r, qdrant_r, pg_r, neo_r = await asyncio.gather(
        _with_timeout(asyncio.to_thread(_sqlite_overview_sync), STORE_PROBE_TIMEOUT, dict(off)),
        _with_timeout(asyncio.to_thread(qdrant_overview), STORE_PROBE_TIMEOUT, dict(off)),
        _with_timeout(asyncio.to_thread(postgres_overview), STORE_PROBE_TIMEOUT, dict(off)),
        _with_timeout(asyncio.to_thread(neo4j_overview), STORE_PROBE_TIMEOUT, dict(off)),
    )
    return {"sqlite": sqlite_r, "qdrant": qdrant_r, "postgres": pg_r, "neo4j": neo_r,
            "generated_at": time.time()}


_STAGE_SQL = """
SELECT v.msg_id, v.title, v.frames, v.duration_str, v.thumb, v.file_size_mb,
       v.created_at,
       EXISTS(SELECT 1 FROM transcripts t WHERE t.msg_id = v.msg_id) AS has_transcript,
       EXISTS(SELECT 1 FROM frame_notes f WHERE f.msg_id = v.msg_id) AS has_notes,
       EXISTS(SELECT 1 FROM oracle_chunks o WHERE o.msg_id = v.msg_id) AS has_narration,
       (SELECT COUNT(*) FROM oracle_chunks o WHERE o.msg_id = v.msg_id) AS chunk_count
FROM videos v
"""


@explorer_router.get("/api/db/videos")
async def api_db_videos(offset: int = 0, limit: int = 24, q: str = "",
                        sort: str = "newest"):
    """Paginated video registry with per-video pipeline stage flags.
    IMPORTANT: does NOT trigger a Telegram sync (that's POST /api/db/sync)."""
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 100))
    order = {"newest": "v.created_at DESC", "oldest": "v.created_at ASC",
             "largest": "v.file_size_mb DESC", "longest": "v.duration_sec DESC"
             }.get(sort, "v.created_at DESC")
    try:
        async with aiosqlite.connect(DB_PATH, timeout=20) as conn:
            where, params = "", []
            if q.strip():
                where = "WHERE v.title LIKE ? OR CAST(v.msg_id AS TEXT) LIKE ?"
                params = [f"%{q.strip()[:60]}%", f"%{q.strip()[:60]}%"]
            async with conn.execute(
                    f"SELECT COUNT(*) FROM videos v {where}", params) as cur:
                total = (await cur.fetchone())[0]
            async with conn.execute(
                    f"{_STAGE_SQL} {where} ORDER BY {order} LIMIT ? OFFSET ?",
                    params + [limit, offset]) as cur:
                rows = await cur.fetchall()
        videos = []
        for r in rows:
            stages = {
                "downloaded": True,
                "frames": bool(r[2]),
                "analyzed": bool(r[7] or r[8]),
                "narrated": bool(r[9]),
            }
            videos.append({
                "msg_id": r[0], "title": r[1] or f"Video #{r[0]}",
                "frames": r[2] or 0, "duration_str": r[3], "thumb": r[4],
                "file_size_mb": r[5], "created_at": r[6],
                "chunk_count": r[10] or 0, "stages": stages,
            })
        return {"total": total, "offset": offset, "limit": limit, "videos": videos}
    except Exception as e:
        return _err(f"db query failed: {e}", 500)


@explorer_router.post("/api/db/sync")
async def api_db_sync():
    """Explicit, lock-guarded Telegram folder sync (was implicit before)."""
    token, err = _try_lock("db_sync", 300)
    if err:
        return err
    try:
        from v17_backend import sync_v17_database
        result = await asyncio.to_thread(sync_v17_database)
        return {"ok": True, "synced": result if isinstance(result, (int, dict)) else True}
    except Exception as e:
        return _err(f"sync failed: {str(e)[:160]}", 500)
    finally:
        _safe_release("db_sync", token)


def _video_full_sync(msg_id):
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    conn.row_factory = sqlite3.Row
    try:
        video = conn.execute("SELECT * FROM videos WHERE msg_id = ?", (msg_id,)).fetchone()
        if not video:
            return None
        out = {"video": dict(video)}
        out["chunks"] = ([dict(r) for r in conn.execute(
            "SELECT * FROM oracle_chunks WHERE msg_id = ? ORDER BY start_t", (msg_id,))]
            if _table_exists(conn, "oracle_chunks") else [])
        out["transcript_count"] = (conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE msg_id = ?", (msg_id,)).fetchone()[0]
            if _table_exists(conn, "transcripts") else 0)
        out["frame_note_count"] = (conn.execute(
            "SELECT COUNT(*) FROM frame_notes WHERE msg_id = ?", (msg_id,)).fetchone()[0]
            if _table_exists(conn, "frame_notes") else 0)
        out["events"] = ([dict(r) for r in conn.execute(
            "SELECT * FROM processing_events WHERE msg_id = ? ORDER BY created_at DESC LIMIT 100",
            (msg_id,))] if _table_exists(conn, "processing_events") else [])
        # Video file availability + playback URL (served by StaticFiles /videos)
        path = video["abs_path"]
        vfile = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
        out["video_file"] = {
            "exists": os.path.exists(vfile),
            "url": f"/videos/video_{msg_id}.mp4" if os.path.exists(vfile) else None,
        }
        out["frames_available"] = bool(path and os.path.isdir(path))
        return out
    finally:
        conn.close()


@explorer_router.get("/api/db/video/{msg_id}/full")
async def api_db_video_full(msg_id: int):
    try:
        data = await asyncio.to_thread(_video_full_sync, msg_id)
        if data is None:
            return _err(f"video {msg_id} not found", 404)
        return data
    except Exception as e:
        return _err(f"lookup failed: {e}", 500)


def _chunk_detail_sync(chunk_id):
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "oracle_chunks"):
            return None
        chunk = conn.execute(
            "SELECT * FROM oracle_chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if not chunk:
            return None
        chunk = dict(chunk)
        msg_id, start_t, end_t = chunk["msg_id"], chunk["start_t"] or 0, chunk["end_t"] or 0
        out = {"chunk": chunk}

        # Everything the AI wrote for this exact time window
        out["transcripts"] = ([dict(r) for r in conn.execute(
            "SELECT start_sec, end_sec, text FROM transcripts "
            "WHERE msg_id = ? AND end_sec >= ? AND start_sec <= ? ORDER BY start_sec",
            (msg_id, start_t, end_t))] if _table_exists(conn, "transcripts") else [])
        out["frame_notes"] = ([dict(r) for r in conn.execute(
            "SELECT frame_idx, ts_sec, objects, ocr_text, description FROM frame_notes "
            "WHERE msg_id = ? AND ts_sec >= ? AND ts_sec <= ? ORDER BY ts_sec",
            (msg_id, start_t, end_t))] if _table_exists(conn, "frame_notes") else [])
        out["moments"] = ([dict(zip(("msg_id", "ts_sec", "source", "content"), r))
                           for r in conn.execute(
            "SELECT msg_id, ts_sec, source, content FROM moments_search "
            "WHERE msg_id = ? AND CAST(ts_sec AS REAL) >= ? AND CAST(ts_sec AS REAL) <= ? LIMIT 50",
            (msg_id, start_t, end_t))] if _table_exists(conn, "moments_search") else [])
        out["events"] = ([dict(r) for r in conn.execute(
            "SELECT * FROM processing_events WHERE chunk_id = ? OR "
            "(msg_id = ? AND chunk_id IS NULL) ORDER BY created_at LIMIT 100",
            (chunk_id, msg_id))] if _table_exists(conn, "processing_events") else [])

        # Frame images on disk within this chunk's window (served via /data mount)
        frames = []
        video_row = conn.execute(
            "SELECT abs_path, folder_id FROM videos WHERE msg_id = ?", (msg_id,)).fetchone()
        if video_row and video_row["abs_path"] and os.path.isdir(video_row["abs_path"]):
            folder = os.path.basename(video_row["abs_path"].rstrip("/"))
            # Timestamps only exist in full-res names: frame_00042_ts_12.340s.jpg
            # (preview files are frame_00042.jpg — no ts — so parse full-res,
            # then serve the lightweight preview twin if it exists)
            preview_dir = os.path.join(video_row["abs_path"], PREVIEW_DIR_NAME)
            has_preview = os.path.isdir(preview_dir)
            try:
                preview_names = set(os.listdir(preview_dir)) if has_preview else set()
                for fname in sorted(os.listdir(video_row["abs_path"])):
                    if not fname.endswith(".jpg") or "_ts_" not in fname:
                        continue
                    try:
                        ts = float(fname.rsplit("_ts_", 1)[1].rsplit("s.jpg", 1)[0].rstrip("s"))
                    except (IndexError, ValueError):
                        continue
                    if start_t <= ts <= end_t:
                        idx_part = fname.split("_ts_")[0]           # frame_00042
                        preview_twin = f"{idx_part}.jpg"
                        if preview_twin in preview_names:
                            url = f"/data/{folder}/{PREVIEW_DIR_NAME}/{preview_twin}"
                        else:
                            url = f"/data/{folder}/{fname}"
                        frames.append({"ts": ts, "url": url,
                                       "full_url": f"/data/{folder}/{fname}"})
            except OSError:
                pass
        out["frames"] = frames[:60]

        # Playback info: the actual chunk of the actual video
        vfile = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
        out["playback"] = {
            "exists": os.path.exists(vfile),
            "url": f"/videos/video_{msg_id}.mp4" if os.path.exists(vfile) else None,
            "start_t": start_t, "end_t": end_t,
        }
        return out
    finally:
        conn.close()


@explorer_router.get("/api/db/chunk/{chunk_id}")
async def api_db_chunk(chunk_id: str):
    try:
        data = await asyncio.to_thread(_chunk_detail_sync, chunk_id)
        if data is None:
            return _err(f"chunk '{chunk_id}' not found", 404)
        # Entity context from Neo4j (best-effort, never blocks the response long)
        async def _ents():
            def _sync():
                try:
                    from tripartite_db import graph_entity_search  # noqa
                    from tripartite_db import get_neo4j_driver
                    driver = get_neo4j_driver()
                    if driver is None:
                        return {"status": "offline", "entities": []}
                    with driver.session() as s:
                        recs = s.run(
                            "MATCH (c:Chunk {id: $cid})-[*1..2]->(e:Entity) "
                            "RETURN DISTINCT e.name AS name LIMIT 25", cid=chunk_id)
                        ents = [r["name"] for r in recs]
                    driver.close()
                    return {"status": "ok", "entities": ents}
                except Exception as e:
                    return {"status": "offline", "reason": str(e)[:120], "entities": []}
            return await asyncio.to_thread(_sync)
        data["graph"] = await _with_timeout(_ents(), 3.0,
                                            {"status": "timeout", "entities": []})
        return data
    except Exception as e:
        return _err(f"chunk lookup failed: {e}", 500)


@explorer_router.get("/api/db/chunk/{chunk_id}/events")
async def api_db_chunk_events(chunk_id: str):
    def _sync():
        conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        conn.row_factory = sqlite3.Row
        try:
            if not _table_exists(conn, "processing_events"):
                return []
            return [dict(r) for r in conn.execute(
                "SELECT * FROM processing_events WHERE chunk_id = ? ORDER BY created_at",
                (chunk_id,))]
        finally:
            conn.close()
    try:
        return {"chunk_id": chunk_id, "events": await asyncio.to_thread(_sync)}
    except Exception as e:
        return _err(str(e), 500)


# ═══════════════════════════════════════════════════════════
# GRAPH EXPLORER (Neo4j → interactive force-graph)
# ═══════════════════════════════════════════════════════════
@explorer_router.get("/api/db/graph")
async def api_db_graph(limit: int = 200, q: str = "", label: str = ""):
    from tripartite_db import fetch_graph
    return await _with_timeout(
        asyncio.to_thread(fetch_graph, limit, q or None, label or None),
        8.0, {"status": "timeout", "nodes": [], "edges": []})


@explorer_router.get("/api/db/graph/expand/{node_id}")
async def api_db_graph_expand(node_id: str):
    from tripartite_db import expand_node
    return await _with_timeout(
        asyncio.to_thread(expand_node, node_id),
        6.0, {"status": "timeout", "nodes": [], "edges": []})


@explorer_router.get("/api/db/graph/node/{node_id}")
async def api_db_graph_node(node_id: str):
    from tripartite_db import get_node_detail
    return await _with_timeout(
        asyncio.to_thread(get_node_detail, node_id),
        6.0, {"status": "timeout"})


# ═══════════════════════════════════════════════════════════
# QUEUE / PIPELINE MANAGEMENT
# ═══════════════════════════════════════════════════════════
def _valid_queue(name):
    return name in ALL_QUEUES


def _try_lock(name, ttl_sec):
    """
    Acquire an op-lock, distinguishing 'held by someone else' from
    'Redis is down'. Returns (token, error_response_or_None).
    """
    try:
        token = acquire_op_lock(name, ttl_sec=ttl_sec)
    except Exception as e:
        return None, _err(f"queue broker unreachable: {str(e)[:120]}", 503)
    if not token:
        return None, _err("operation already in progress", 409)
    return token, None


def _safe_release(name, token):
    try:
        release_op_lock(name, token)
    except Exception:
        pass  # lock has a TTL; it will expire on its own


@explorer_router.get("/api/queues")
async def api_queues():
    try:
        return await asyncio.to_thread(get_queue_metrics)
    except Exception as e:
        return _err(f"queue broker unreachable: {str(e)[:120]}", 503)


@explorer_router.post("/api/queues/{queue_name}/pause")
async def api_queue_pause(queue_name: str):
    if not _valid_queue(queue_name):
        return _err(f"unknown queue '{queue_name}'", 404)
    try:
        await asyncio.to_thread(pause_queue, queue_name)
        return {"ok": True, "queue": queue_name, "paused": True}
    except Exception as e:
        return _err(f"queue broker unreachable: {str(e)[:120]}", 503)


@explorer_router.post("/api/queues/{queue_name}/resume")
async def api_queue_resume(queue_name: str):
    if not _valid_queue(queue_name):
        return _err(f"unknown queue '{queue_name}'", 404)
    try:
        await asyncio.to_thread(resume_queue, queue_name)
        return {"ok": True, "queue": queue_name, "paused": False}
    except Exception as e:
        return _err(f"queue broker unreachable: {str(e)[:120]}", 503)


@explorer_router.get("/api/queues/{queue_name}/dlq")
async def api_queue_dlq_peek(queue_name: str, count: int = 20):
    if not _valid_queue(queue_name):
        return _err(f"unknown queue '{queue_name}'", 404)
    try:
        jobs = await asyncio.to_thread(peek_dlq, queue_name, max(1, min(int(count), 100)))
        return {"queue": queue_name, "jobs": jobs}
    except Exception as e:
        return _err(f"queue broker unreachable: {str(e)[:120]}", 503)


@explorer_router.post("/api/queues/{queue_name}/dlq/replay")
async def api_queue_dlq_replay(queue_name: str, count: int = 0):
    if not _valid_queue(queue_name):
        return _err(f"unknown queue '{queue_name}'", 404)
    token, err = _try_lock(f"dlq_{queue_name}", 120)
    if err:
        return err
    try:
        n = await asyncio.to_thread(replay_dlq, queue_name, int(count) or None)
        return {"ok": True, "queue": queue_name, "replayed": n}
    except Exception as e:
        return _err(f"replay failed: {str(e)[:160]}", 500)
    finally:
        _safe_release(f"dlq_{queue_name}", token)


@explorer_router.post("/api/queues/{queue_name}/dlq/purge")
async def api_queue_dlq_purge(queue_name: str):
    if not _valid_queue(queue_name):
        return _err(f"unknown queue '{queue_name}'", 404)
    token, err = _try_lock(f"dlq_{queue_name}", 120)
    if err:
        return err
    try:
        n = await asyncio.to_thread(purge_dlq, queue_name)
        return {"ok": True, "queue": queue_name, "purged": n}
    except Exception as e:
        return _err(f"purge failed: {str(e)[:160]}", 500)
    finally:
        _safe_release(f"dlq_{queue_name}", token)


@explorer_router.get("/api/queues/{queue_name}/processing")
async def api_queue_processing(queue_name: str, count: int = 25):
    if not _valid_queue(queue_name):
        return _err(f"unknown queue '{queue_name}'", 404)
    try:
        jobs = await asyncio.to_thread(peek_processing, queue_name, max(1, min(int(count), 100)))
        return {"queue": queue_name, "jobs": jobs}
    except Exception as e:
        return _err(f"queue broker unreachable: {str(e)[:120]}", 503)


@explorer_router.post("/api/queues/reap")
async def api_queues_reap():
    token, err = _try_lock("reap", 60)
    if err:
        return err
    try:
        result = await asyncio.to_thread(reap_stale_jobs)
        return {"ok": True, "reaped": result}
    except Exception as e:
        return _err(f"reap failed: {str(e)[:160]}", 500)
    finally:
        _safe_release("reap", token)


@explorer_router.get("/api/pipeline/videos")
async def api_pipeline_videos(offset: int = 0, limit: int = 30):
    """Per-video stage tracker (same flags as /api/db/videos, pipeline-shaped)."""
    return await api_db_videos(offset=offset, limit=limit, q="", sort="newest")


# ═══════════════════════════════════════════════════════════
# SYSTEM — environment / service visibility
# ═══════════════════════════════════════════════════════════
@explorer_router.get("/api/system/env-check")
async def api_env_check():
    def _services():
        out = {}
        try:
            get_redis().ping()
            out["redis"] = "ok"
        except Exception as e:
            out["redis"] = f"offline: {str(e)[:80]}"
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5)
            conn.execute("SELECT 1")
            conn.close()
            out["sqlite"] = "ok"
        except Exception as e:
            out["sqlite"] = f"offline: {str(e)[:80]}"
        return out
    from tripartite_db import qdrant_overview, postgres_overview, neo4j_overview
    off = {"status": "timeout"}
    services, qd, pg, neo = await asyncio.gather(
        asyncio.to_thread(_services),
        _with_timeout(asyncio.to_thread(qdrant_overview), 4.0, dict(off)),
        _with_timeout(asyncio.to_thread(postgres_overview), 4.0, dict(off)),
        _with_timeout(asyncio.to_thread(neo4j_overview), 4.0, dict(off)),
    )
    services["qdrant"] = qd.get("status", "unknown")
    services["postgres"] = pg.get("status", "unknown")
    services["neo4j"] = neo.get("status", "unknown")
    return {"secrets": env_check_masked(), "services": services}


# Ensure the full schema exists as soon as this module is imported by the app
try:
    init_sqlite_schema()
except Exception:
    pass
