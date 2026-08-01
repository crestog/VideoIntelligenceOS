"""
VIOS Omniscient Engine — Layer 5 Orchestration (unified)

One watchdog-managed GPU process that hosts:

  • Vision Worker   — QUEUE_OMNI_VISION → frames, depth/motion, SigLIP+CLIP
                      vectors → PostgreSQL + Qdrant
  • Oracle Worker   — QUEUE_OMNI_ORACLE → ffmpeg chunks → Qwen2.5-VL narrative
                      → PostgreSQL + BGE vectors + Neo4j GraphRAG (NIM)
  • Telegram Bot    — private uploads (PRIORITY lane) + natural-language
                      hybrid search with spatial proof + NIM synthesis
  • God-Mode Flask  — database explorer on 127.0.0.1:{OMNI_DASHBOARD_PORT},
                      reverse-proxied by ui_server at /omni (Omniscient tab)

Queueing uses queue_manager v3 (atomic claims, retries, DLQ). Bot uploads are
pushed with is_priority=True; Ghost-Worker harvest jobs ride the DEFAULT lane,
so user uploads always process first.
"""

import os
import json
import math
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid as uuid_lib

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HOME", "/kaggle/working/huggingface_cache"
                      if os.path.isdir("/kaggle/working") else
                      os.path.expanduser("~/.cache/huggingface"))

import cv2
import numpy as np
import redis as redis_lib
import scipy.ndimage
import scipy.signal
import torch
from PIL import Image

from config import (ARCHIVE_DIR, LAKE_DIR, API_ID, API_HASH, BOT_TOKEN,
                    QUEUE_OMNI_VISION, QUEUE_OMNI_ORACLE, OMNI_DEDUP_SET,
                    NIM_API_KEY, NIM_BASE_URL, NIM_MODEL, OMNI_DASHBOARD_PORT,
                    OMNI_MODE_OMNI, OMNI_MODE_BLITZ, OMNI_BLITZ_SAMPLE_FPS)
from logger import vios_log
from queue_manager import (claim_job, ack_job, fail_job, push_job, get_queue_metrics,
                           wait_for_redis)
import omni_db
from omni_db import (stable_id64, get_pg_conn, get_pg_conn_optional,
                     get_qdrant, get_neo4j)
import omni_models
from omni_models import (MODELS, device_0, hybrid_spatial_proof,
                         extract_img_features, qwen_describe_video,
                         siglip_text_vec, clip_text_vec, bge_encode)
from omni_prompts import PROMPTS


def log(msg, level="INFO"):
    vios_log(msg, "OMNI", level)


REDIS = redis_lib.Redis(host="localhost", port=6379, decode_responses=True,
                        socket_timeout=10, socket_connect_timeout=5, retry_on_timeout=True)


# ═══════════════════════════════════════════════════════════
# NVIDIA NIM CLIENT (lazy, optional — everything has a fallback)
# ═══════════════════════════════════════════════════════════
_nim_client = None


def nim():
    global _nim_client
    if _nim_client is None and NIM_API_KEY:
        try:
            from openai import OpenAI
            _nim_client = OpenAI(base_url=NIM_BASE_URL, api_key=NIM_API_KEY)
        except Exception as e:
            log(f"NIM client init failed: {e}", "WARN")
    return _nim_client


def nim_chat(messages, temperature=0.2, max_tokens=1024):
    """One NIM completion; returns text or raises."""
    client = nim()
    if not client:
        raise RuntimeError("NIM API not configured")
    resp = client.chat.completions.create(model=NIM_MODEL, messages=messages,
                                          temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content


# ═══════════════════════════════════════════════════════════
# GRAPHRAG — narrative → entities/relationships → Neo4j
# ═══════════════════════════════════════════════════════════
def extract_and_store_graphrag(neo4j_driver, narrative_text, video_uuid, chunk_id):
    extraction_prompt = PROMPTS["entity_extraction"].format(
        entity_types=", ".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        input_text=narrative_text)
    try:
        graph_output = nim_chat([{"role": "user", "content": extraction_prompt}],
                                temperature=0.1, max_tokens=2048)
    except Exception as e:
        log(f"GraphRAG NIM extraction skipped for {chunk_id}: {e}", "WARN")
        return

    try:
        lines = graph_output.split(PROMPTS["DEFAULT_RECORD_DELIMITER"])
        with neo4j_driver.session() as session:
            for line in lines:
                line = line.strip().strip("()")
                if not line or PROMPTS["DEFAULT_COMPLETION_DELIMITER"] in line:
                    continue
                parts = line.split(PROMPTS["DEFAULT_TUPLE_DELIMITER"])
                if len(parts) < 4:
                    continue
                record_type = parts[0].strip('"').lower()

                if record_type == "entity":
                    name, e_type, desc = parts[1].strip('"'), parts[2].strip('"'), parts[3].strip('"')
                    session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.description = $desc
                        WITH e
                        MATCH (c:Chunk {id: $cid})
                        MERGE (c)-[:CONTAINS_ENTITY]->(e)
                    """, name=name, type=e_type, desc=desc, cid=chunk_id)

                elif record_type == "relationship" and len(parts) >= 5:
                    src, tgt, desc, weight = (parts[1].strip('"'), parts[2].strip('"'),
                                              parts[3].strip('"'), parts[4].strip('"'))
                    try:
                        weight_int = int(float(re.sub(r"[^\d.]", "", weight) or 0))
                    except ValueError:
                        weight_int = 0
                    session.run("""
                        MATCH (s:Entity {name: $src})
                        MATCH (t:Entity {name: $tgt})
                        MERGE (s)-[r:RELATED_TO]->(t)
                        SET r.description = $desc, r.weight = $weight
                    """, src=src, tgt=tgt, desc=desc, weight=weight_int)
    except Exception as e:
        log(f"GraphRAG store failed for {chunk_id}: {e}", "WARN")


# ═══════════════════════════════════════════════════════════
# WORKER 1 — VISION (frames → depth/motion → SigLIP/CLIP → PG + Qdrant)
# ═══════════════════════════════════════════════════════════
def _paused():
    try:
        return REDIS.get("OMNI_PAUSED") == "1"
    except Exception:
        return False


def process_vision_job(payload):
    v_uuid, path, mode = payload["uuid"], payload["path"], payload.get("mode", "blitz")
    REDIS.hset(f"status:{v_uuid}", "vision", "Processing 👁️")

    cap = cv2.VideoCapture(path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # omni: every frame | blitz: ~OMNI_BLITZ_SAMPLE_FPS frames/sec regardless
    # of source fps (the notebook's fixed step-2 exploded on long videos)
    step = 1 if mode == "omni" else max(1, round(native_fps / OMNI_BLITZ_SAMPLE_FPS))

    frames_dir = os.path.join(ARCHIVE_DIR, f"{v_uuid}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    pil_images, timestamps, frame_indices = [], [], []
    idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if idx % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(frames_dir, f"frame_{idx}.jpg"), frame)
            pil_images.append(Image.fromarray(cv2.resize(rgb_frame, (384, 384))))
            timestamps.append(round(idx / native_fps, 3))
            frame_indices.append(idx)
        idx += 1
    cap.release()

    total_frames = len(pil_images)
    if total_frames == 0:
        raise RuntimeError("Zero frames extracted.")

    # Depth + motion per sampled frame
    omni_data = []
    depth_model, depth_proc = MODELS.get("depth_model"), MODELS.get("depth_processor")
    raft_model, raft_tf = MODELS.get("raft_model"), MODELS.get("raft_transforms")
    for i in range(total_frames):
        img = pil_images[i]
        mean_depth = 0.0
        if depth_model:
            in_depth = depth_proc(images=img, return_tensors="pt").to(device_0)
            with torch.no_grad():
                mean_depth = depth_model(**in_depth).predicted_depth.mean().item()
        motion = 0.0
        if mode == "omni" and i > 0 and raft_model:
            i1, i2 = raft_tf(
                torch.tensor(np.array(pil_images[i - 1])).permute(2, 0, 1).unsqueeze(0).to(device_0),
                torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(device_0))
            with torch.no_grad():
                motion = torch.norm(raft_model(i1, i2)[-1], dim=1).mean().item()
        omni_data.append({"depth": round(mean_depth, 2), "motion": round(motion, 2)})
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # Batched embeddings → PG rows + Qdrant points
    from qdrant_client.models import PointStruct
    siglip_points, clip_points = [], []
    pg_conn = get_pg_conn_optional()
    try:
        with pg_conn.cursor() as pg_cursor:
            for i in range(0, total_frames, 32):
                batch = pil_images[i:i + 32]
                s_vecs = extract_img_features(MODELS["siglip_model"], MODELS["siglip_processor"],
                                              device_0, batch).cpu().numpy().tolist()
                c_vecs = extract_img_features(MODELS["clip_model"], MODELS["clip_processor"],
                                              device_0, batch).cpu().numpy().tolist()
                for ts, f_idx, depth_mot, sv, cv_vec in zip(
                        timestamps[i:i + 32], frame_indices[i:i + 32],
                        omni_data[i:i + 32], s_vecs, c_vecs):
                    f_id_str = f"{v_uuid}_{f_idx}"
                    f_id_int = stable_id64(f_id_str)
                    pg_cursor.execute(
                        """INSERT INTO frames (frame_id, video_uuid, video_path, timestamp,
                           frame_idx, depth, motion) VALUES (%s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT DO NOTHING""",
                        (f_id_str, v_uuid, path, ts, f_idx,
                         depth_mot["depth"], depth_mot["motion"]))
                    payload_q = {"frame_id": f_id_str, "video_uuid": v_uuid,
                                 "video_path": path, "timestamp": ts, "frame_idx": f_idx}
                    siglip_points.append(PointStruct(id=f_id_int, vector=sv, payload=payload_q))
                    clip_points.append(PointStruct(id=f_id_int, vector=cv_vec, payload=payload_q))
        pg_conn.commit()
    finally:
        pg_conn.close()

    qdrant = get_qdrant()
    if qdrant:
        if siglip_points:
            qdrant.upsert(collection_name="frames_siglip", points=siglip_points)
        if clip_points:
            qdrant.upsert(collection_name="frames_clip", points=clip_points)

    torch.cuda.empty_cache()
    return total_frames


def vision_worker_loop():
    log("⚙️ Worker 1 (Vision) online — consuming QUEUE_OMNI_VISION")
    while True:
        try:
            if _paused():
                time.sleep(2)
                continue
            job, job_raw = claim_job(QUEUE_OMNI_VISION, timeout=3)
            if not job:
                continue
            payload = job.get("payload", job)
            v_uuid = payload.get("uuid", "?")
            try:
                n = process_vision_job(payload)
                REDIS.hset(f"status:{v_uuid}", "vision", "DONE ✅")
                ack_job(QUEUE_OMNI_VISION, job, job_raw)
                log(f"👁️ Vision {v_uuid}: {n} frames indexed", "SUCCESS")
            except Exception as e:
                REDIS.hset(f"status:{v_uuid}", "vision", f"ERROR: {str(e)[:80]}")
                result = fail_job(QUEUE_OMNI_VISION, job, job_raw, str(e))
                log(f"❌ Vision {v_uuid} failed → {result}: {str(e)[:200]}", "ERROR")
                torch.cuda.empty_cache()
                time.sleep(2)
        except Exception as outer:
            log(f"Vision loop critical error: {outer}", "ERROR")
            time.sleep(5)


# ═══════════════════════════════════════════════════════════
# WORKER 2 — ORACLE (chunks → Qwen narrative → PG + BGE + Neo4j GraphRAG)
# ═══════════════════════════════════════════════════════════
def process_oracle_job(payload):
    v_uuid, path, mode = payload["uuid"], payload["path"], payload.get("mode", "blitz")
    REDIS.hset(f"status:{v_uuid}", "oracle", "Thinking 🧠")

    cap = cv2.VideoCapture(path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    cfg = OMNI_MODE_OMNI if mode == "omni" else OMNI_MODE_BLITZ
    chunk_size, qwen_fps, qwen_tokens = cfg["chunk"], cfg["fps"], cfg["tokens"]
    chunks_dir = os.path.join(ARCHIVE_DIR, f"{v_uuid}_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    previous_context = "This is the start of the video."

    from qdrant_client.models import PointStruct
    bge_points = []
    pg_conn = get_pg_conn_optional()
    try:
        with pg_conn.cursor() as pg_cursor:
            for start_t in range(0, int(math.ceil(duration)), int(chunk_size)):
                if duration - start_t < 1.0:
                    break
                end_t = min(start_t + chunk_size, duration)
                chunk_file = os.path.join(chunks_dir, f"chunk_{start_t}.mp4")
                subprocess.run(
                    ["ffmpeg", "-ss", str(start_t), "-i", path, "-t", str(end_t - start_t),
                     "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                     chunk_file, "-y"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if not os.path.exists(chunk_file) or os.path.getsize(chunk_file) == 0:
                    log(f"⚠️ Oracle {v_uuid}: chunk @{start_t}s failed to render — skipping", "WARN")
                    continue

                narrative = qwen_describe_video(
                    chunk_file,
                    f"Context: '{previous_context}'. Write a highly intelligent, "
                    f"continuous narrative explaining what is occurring now.",
                    fps=qwen_fps, max_new_tokens=qwen_tokens)
                previous_context = narrative

                c_id_str = f"chunk_{v_uuid}_{start_t}"
                c_id_int = stable_id64(c_id_str)

                pg_cursor.execute(
                    """INSERT INTO chunks (chunk_id, video_uuid, video_path, start_t, end_t,
                       description) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                    (c_id_str, v_uuid, path, start_t, end_t, narrative))
                bge_points.append(PointStruct(
                    id=c_id_int, vector=bge_encode(narrative),
                    payload={"chunk_id": c_id_str, "video_uuid": v_uuid, "video_path": path,
                             "start_t": start_t, "end_t": end_t}))

                driver = get_neo4j()
                if driver:
                    try:
                        with driver.session() as session:
                            session.run("""
                                MERGE (v:Video {uuid: $vid})
                                MERGE (c:Chunk {id: $cid, start: $start, end: $end})
                                MERGE (v)-[:CONTAINS]->(c)
                                MERGE (n:Narrative {text: $desc})
                                MERGE (c)-[:DESCRIBED_BY]->(n)
                            """, vid=v_uuid, cid=c_id_str, start=start_t, end=end_t,
                                desc=narrative)
                        extract_and_store_graphrag(driver, narrative, v_uuid, c_id_str)
                    except Exception as e:
                        log(f"Neo4j insert failed for {c_id_str}: {e}", "WARN")
        pg_conn.commit()
    finally:
        pg_conn.close()

    qdrant = get_qdrant()
    if qdrant and bge_points:
        qdrant.upsert(collection_name="chunks_bge", points=bge_points)
    torch.cuda.empty_cache()
    return len(bge_points)


def oracle_worker_loop():
    log("🧠 Worker 2 (Oracle) online — consuming QUEUE_OMNI_ORACLE")
    while True:
        try:
            if _paused():
                time.sleep(2)
                continue
            job, job_raw = claim_job(QUEUE_OMNI_ORACLE, timeout=3)
            if not job:
                continue
            payload = job.get("payload", job)
            v_uuid = payload.get("uuid", "?")
            try:
                n = process_oracle_job(payload)
                REDIS.hset(f"status:{v_uuid}", "oracle", "DONE ✅")
                ack_job(QUEUE_OMNI_ORACLE, job, job_raw)
                log(f"🧠 Oracle {v_uuid}: {n} narrative chunks indexed", "SUCCESS")
            except Exception as e:
                REDIS.hset(f"status:{v_uuid}", "oracle", f"ERROR: {str(e)[:80]}")
                result = fail_job(QUEUE_OMNI_ORACLE, job, job_raw, str(e))
                log(f"❌ Oracle {v_uuid} failed → {result}: {str(e)[:200]}", "ERROR")
                torch.cuda.empty_cache()
                time.sleep(2)
        except Exception as outer:
            log(f"Oracle loop critical error: {outer}", "ERROR")
            time.sleep(5)


# ═══════════════════════════════════════════════════════════
# GOD-MODE FLASK DASHBOARD (proxied by ui_server at /omni)
# ═══════════════════════════════════════════════════════════
def build_dashboard():
    from flask import Flask, jsonify, send_file

    app_dashboard = Flask("OmniGodMode")
    dash_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "omni_dashboard.html")

    @app_dashboard.route("/")
    def index():
        with open(dash_html_path, "r", encoding="utf-8") as f:
            return f.read()

    @app_dashboard.route("/api/videos")
    def get_videos():
        if not omni_db.AVAILABLE["postgres"]:
            return jsonify({"error": "PostgreSQL offline — no video index available"})
        try:
            conn = get_pg_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT video_uuid FROM frames "
                            "UNION SELECT DISTINCT video_uuid FROM chunks")
                data = [{"video_uuid": r[0]} for r in cur.fetchall()]
            conn.close()
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    @app_dashboard.route("/api/video/<video_uuid>")
    def get_video_details(video_uuid):
        if not omni_db.AVAILABLE["postgres"]:
            return jsonify({"error": "PostgreSQL offline — no video details available"})
        try:
            conn = get_pg_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT chunk_id, start_t, end_t, description FROM chunks "
                            "WHERE video_uuid = %s ORDER BY start_t ASC", (video_uuid,))
                chunks = [{"chunk_id": r[0], "start_t": r[1], "end_t": r[2],
                           "description": r[3]} for r in cur.fetchall()]
                cur.execute("SELECT frame_id, frame_idx, depth, motion, timestamp FROM frames "
                            "WHERE video_uuid = %s ORDER BY frame_idx ASC", (video_uuid,))
                frames = [{"frame_id": r[0], "frame_idx": r[1], "depth": r[2],
                           "motion": r[3], "timestamp": r[4]} for r in cur.fetchall()]
            conn.close()
            return jsonify({"chunks": chunks, "frames": frames})
        except Exception as e:
            return jsonify({"error": str(e)})

    @app_dashboard.route("/media/chunk/<video_uuid>/<start_t>")
    def serve_chunk(video_uuid, start_t):
        try:
            t = float(start_t)
            t_str = str(int(t)) if t.is_integer() else str(t)
            file_path = os.path.join(ARCHIVE_DIR, f"{video_uuid}_chunks", f"chunk_{t_str}.mp4")
            if not os.path.exists(file_path):
                file_path = os.path.join(ARCHIVE_DIR, f"{video_uuid}_chunks",
                                         f"chunk_{float(start_t)}.mp4")
            if os.path.exists(file_path):
                # conditional=True → Range support so <video> seeking works
                return send_file(file_path, mimetype="video/mp4", conditional=True)
            return "Chunk not found", 404
        except Exception as e:
            return str(e), 500

    @app_dashboard.route("/media/frame/<video_uuid>/<int:frame_idx>")
    def serve_frame(video_uuid, frame_idx):
        file_path = os.path.join(ARCHIVE_DIR, f"{video_uuid}_frames", f"frame_{frame_idx}.jpg")
        if os.path.exists(file_path):
            return send_file(file_path, mimetype="image/jpeg", conditional=True)
        return "Frame not found on disk", 404

    @app_dashboard.route("/api/vector/<collection_name>/<frame_id>")
    def get_qdrant_vector(collection_name, frame_id):
        if collection_name not in ("frames_siglip", "frames_clip", "chunks_bge"):
            return jsonify({"error": "Unknown collection"})
        try:
            qdrant = get_qdrant()
            if not qdrant:
                return jsonify({"error": "Qdrant offline"})
            res = qdrant.retrieve(collection_name=collection_name,
                                  ids=[stable_id64(frame_id)], with_vectors=True)
            if res and res[0].vector:
                return jsonify({"vector": res[0].vector})
            return jsonify({"error": "Vector not found in Qdrant memory"})
        except Exception as e:
            return jsonify({"error": str(e)})

    @app_dashboard.route("/api/neo4j/graph")
    def get_neo4j_graph():
        driver = get_neo4j()
        if not driver:
            return jsonify({"error": "Neo4j offline — knowledge graph unavailable"})
        try:
            with driver.session() as session:
                results = session.run("""
                    MATCH (n)-[r]->(m)
                    RETURN id(n) AS src_id, labels(n) AS src_lbl, properties(n) AS src_props,
                           id(m) AS tgt_id, labels(m) AS tgt_lbl, properties(m) AS tgt_props,
                           id(r) AS rel_id, type(r) AS rel_type, properties(r) AS rel_props
                    LIMIT 300
                """).data()

            nodes_dict, edges = {}, []
            for row in results:
                src_id = row["src_id"]
                if src_id not in nodes_dict:
                    label_name = row["src_props"].get("name", row["src_props"].get(
                        "id", row["src_lbl"][0] if row["src_lbl"] else "?"))
                    nodes_dict[src_id] = {
                        "id": src_id, "label": str(label_name),
                        "color": "#00ffcc" if "Chunk" in row["src_lbl"] else "#ff0066",
                        "raw_properties": {"labels": row["src_lbl"],
                                           "properties": row["src_props"]}}
                tgt_id = row["tgt_id"]
                if tgt_id not in nodes_dict:
                    label_name = row["tgt_props"].get("name", row["tgt_props"].get(
                        "id", row["tgt_lbl"][0] if row["tgt_lbl"] else "?"))
                    nodes_dict[tgt_id] = {
                        "id": tgt_id, "label": str(label_name), "color": "#ff9900",
                        "raw_properties": {"labels": row["tgt_lbl"],
                                           "properties": row["tgt_props"]}}
                edges.append({"id": row["rel_id"], "from": src_id, "to": tgt_id,
                              "label": row["rel_type"],
                              "raw_properties": {"type": row["rel_type"],
                                                 "properties": row["rel_props"]}})
            return jsonify({"nodes": list(nodes_dict.values()), "edges": edges})
        except Exception as e:
            return jsonify({"error": str(e)})

    return app_dashboard


def run_dashboard():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app_dashboard = build_dashboard()
    # localhost only — the outside world reaches it through ui_server's /omni proxy
    app_dashboard.run(host="127.0.0.1", port=OMNI_DASHBOARD_PORT,
                      debug=False, use_reloader=False)


# ═══════════════════════════════════════════════════════════
# TELEGRAM BOT — upload ingestion (PRIORITY) + hybrid search
# ═══════════════════════════════════════════════════════════
from pyrogram import Client, filters, idle
from pyrogram.enums import ParseMode
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

app = Client(os.path.join(LAKE_DIR, "omni_bot"),
             api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

pending_videos = {}


async def send_long_message(client, chat_id, text):
    """Telegram-safe long message sender (4000-char pages, markdown fallback)."""
    safe_text = re.sub(r'^#+\s+(.*)', r'**\1**', text, flags=re.MULTILINE)
    max_len = 4000
    for i in range(0, len(safe_text), max_len):
        chunk = safe_text[i:i + max_len]
        try:
            await client.send_message(chat_id, chunk)
        except Exception:
            try:
                await client.send_message(chat_id, chunk, parse_mode=ParseMode.DISABLED)
            except Exception as e:
                log(f"send_long_message failed: {e}", "WARN")


async def send_diagnostic_frame(client, chat_id, video_path, frame_idx, caption):
    """Grab one frame from the video and send it as photo proof.
    (Referenced-but-undefined in the original notebook — implemented here.)"""
    tmp_path = None
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return
        tmp_path = os.path.join(ARCHIVE_DIR, f"diag_{uuid_lib.uuid4().hex}.jpg")
        cv2.imwrite(tmp_path, frame)
        await client.send_photo(chat_id, photo=tmp_path, caption=caption)
    except Exception as e:
        log(f"Diagnostic frame failed: {e}", "WARN")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def get_main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 System Status", callback_data="cmd_status"),
         InlineKeyboardButton("🧹 Purge Cache", callback_data="cmd_purge")],
        [InlineKeyboardButton("🧊 Freeze Database", callback_data="cmd_freeze")],
    ])


@app.on_message(filters.command("start") & filters.private)
async def handle_start(client, message):
    await message.reply_text(
        "👋 **Welcome to Omniscient AI**\n\nUpload a video to begin processing, "
        "or send any text to search the indexed library.",
        reply_markup=get_main_keyboard())


@app.on_message(filters.private & (filters.video | filters.document | filters.animation))
async def handle_incoming_video(client, message):
    msg = await message.reply_text("📥 Downloading to archive...")
    v_uuid = os.urandom(4).hex()
    master_path = os.path.join(ARCHIVE_DIR, f"{v_uuid}.mp4")
    try:
        actual_path = await message.download(file_name=master_path)
        pending_videos[v_uuid] = actual_path
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("⚡ Blitz Mode", callback_data=f"blitz|{v_uuid}")],
            [InlineKeyboardButton("👁️ Omniscient Mode", callback_data=f"omni|{v_uuid}")]])
        await msg.edit_text("🎯 **Video Downloaded.**\nSelect processing mode:",
                            reply_markup=keyboard)
    except Exception as e:
        await msg.edit_text(f"ERROR: Download failed: {str(e)[:200]}")


@app.on_callback_query()
async def handle_callback(client, callback_query: CallbackQuery):
    data = callback_query.data
    chat_id = callback_query.message.chat.id

    if data == "cmd_status":
        try:
            m = get_queue_metrics()
            v, o = m.get(QUEUE_OMNI_VISION, {}), m.get(QUEUE_OMNI_ORACLE, {})
            txt = (f"👁️ Vision: {v.get('pending_total', 0)} pending / "
                   f"{v.get('processing', 0)} active / {v.get('total_completed', 0)} done\n"
                   f"🧠 Oracle: {o.get('pending_total', 0)} pending / "
                   f"{o.get('processing', 0)} active / {o.get('total_completed', 0)} done")
            await client.send_message(chat_id, f"📊 **System Status**\n{txt}")
        except Exception as e:
            await client.send_message(chat_id, f"Status error: {e}")
        return await callback_query.answer()
    elif data == "cmd_purge":
        return await callback_query.answer("Purging cache...")
    elif data == "cmd_freeze":
        return await callback_query.answer("Freezing DB...")

    try:
        mode, v_uuid = data.split("|")[0], data.split("|")[1]
    except (IndexError, ValueError):
        return await callback_query.answer("Unknown action.", show_alert=True)

    if v_uuid not in pending_videos:
        return await callback_query.answer("Session expired.", show_alert=True)
    path = pending_videos.pop(v_uuid)

    REDIS.hset(f"status:{v_uuid}", mapping={
        "chat_id": chat_id, "msg_id": callback_query.message.id, "mode": mode,
        "vision": "Queued ⏳", "oracle": "Queued ⏳", "notified": "FALSE"})
    job = {"uuid": v_uuid, "path": path, "mode": mode, "source": "bot"}
    # Bot uploads ride the PRIORITY lane — they pre-empt Ghost-Worker harvest
    push_job(QUEUE_OMNI_VISION, job, is_priority=True)
    push_job(QUEUE_OMNI_ORACLE, job, is_priority=True)
    await callback_query.message.edit_text("🚀 Injecting into Layer 5 Queues (PRIORITY)...")


@app.on_message(filters.private & filters.text &
                ~filters.command(["start", "status", "purge_cache", "freeze", "awaken"]))
async def handle_search(client, message):
    raw_query = message.text
    status_msg = await message.reply_text(
        "🔍 **ENTERPRISE HYBRID SEARCH**\nInitializing NIM GraphRAG Protocol...")

    try:
        qdrant = get_qdrant()
        if not qdrant:
            return await status_msg.edit_text("🚫 **Vector store offline.** Try again shortly.")

        # ── 1. NIM query rewrite (graceful fallback to raw query) ──
        try:
            optimized_query = nim_chat(
                [{"role": "user", "content":
                  PROMPTS["query_rewrite_for_visual_retrieval"].format(input_text=raw_query)}],
                max_tokens=100).strip()
            await status_msg.edit_text(
                f"🧠 **NIM Query Optimization:**\n_{optimized_query}_\n\n"
                f"Querying Tri-Partite Matrix...")
        except Exception as api_err:
            log(f"NIM rewrite error: {api_err}", "WARN")
            optimized_query = raw_query
            await status_msg.edit_text(
                f"⚠️ **NIM Optimization Failed**\nFalling back to Raw Query: "
                f"_{optimized_query}_\n\nQuerying Matrix...")

        # ── 2. Embed the query in all three spaces ──
        siglip_q = siglip_text_vec(optimized_query)
        clip_q = clip_text_vec(optimized_query)
        bge_q = bge_encode(optimized_query)

        # ── 3. Semantic chunk hit → target video ──
        bge_hits = qdrant.query_points(collection_name="chunks_bge", query=bge_q, limit=1).points
        best_vid_path, best_chunk, c_desc = None, None, "Oracle indexing pending..."

        if bge_hits:
            best_chunk = bge_hits[0].payload
            best_vid_path = best_chunk["video_path"]
            mid_idx = 0
            # Optional: if PG is down, c_desc/mid_idx keep their defaults and the
            # cascade continues on vectors alone instead of aborting the search.
            pg_conn = get_pg_conn_optional()
            try:
                with pg_conn.cursor() as pg_cursor:
                    pg_cursor.execute("SELECT description FROM chunks WHERE chunk_id = %s",
                                      (best_chunk["chunk_id"],))
                    row = pg_cursor.fetchone()
                    if row:
                        c_desc = row[0]
                    mid_t = (best_chunk["start_t"] + best_chunk["end_t"]) / 2
                    pg_cursor.execute(
                        "SELECT frame_idx FROM frames WHERE video_path = %s "
                        "ORDER BY abs(timestamp - %s) LIMIT 1", (best_vid_path, mid_t))
                    idx_row = pg_cursor.fetchone()
                    mid_idx = idx_row[0] if idx_row else 0
            finally:
                pg_conn.close()
            if mid_idx:
                await send_diagnostic_frame(
                    client, message.chat.id, best_vid_path, mid_idx,
                    f"📊 **[MODEL 1: BGE Semantic]**\n- Confidence Score: {bge_hits[0].score:.4f}")
        else:
            await status_msg.edit_text("⏳ Oracle Queue Pending. Executing Global Vision Hunt...")
            global_s_hits = qdrant.query_points(collection_name="frames_siglip",
                                                query=siglip_q, limit=1).points
            if not global_s_hits:
                return await status_msg.edit_text("🚫 **Database is empty or no visual match.**")
            best_vid_path = global_s_hits[0].payload["video_path"]

        # ── 4. Per-frame visual scoring within the target video ──
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        vid_filter = Filter(must=[FieldCondition(key="video_path",
                                                 match=MatchValue(value=best_vid_path))])
        s_hits = qdrant.query_points(collection_name="frames_siglip", query=siglip_q,
                                     query_filter=vid_filter, limit=10000).points
        c_hits = qdrant.query_points(collection_name="frames_clip", query=clip_q,
                                     query_filter=vid_filter, limit=10000).points

        s_scores_map = {h.payload["frame_idx"]: h.score for h in s_hits}
        c_scores_map = {h.payload["frame_idx"]: h.score for h in c_hits}

        best_sig_idx = max(s_scores_map, key=s_scores_map.get) if s_scores_map else 0
        best_clip_idx = max(c_scores_map, key=c_scores_map.get) if c_scores_map else 0

        if best_sig_idx:
            await send_diagnostic_frame(
                client, message.chat.id, best_vid_path, best_sig_idx,
                f"👁️ **[MODEL 2: SigLIP Visual]**\n- Confidence Score: {s_scores_map[best_sig_idx]:.4f}")
        if best_clip_idx:
            await send_diagnostic_frame(
                client, message.chat.id, best_vid_path, best_clip_idx,
                f"👁️ **[MODEL 3: CLIP Visual]**\n- Confidence Score: {c_scores_map[best_clip_idx]:.4f}")

        pg_conn = get_pg_conn_optional()
        try:
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute("SELECT frame_idx, timestamp, depth, motion FROM frames "
                                  "WHERE video_path = %s ORDER BY frame_idx", (best_vid_path,))
                db_frames = pg_cursor.fetchall()
        finally:
            pg_conn.close()

        if not db_frames:
            hint = ("" if omni_db.AVAILABLE["postgres"]
                    else " (PostgreSQL is offline — the frame index is unavailable.)")
            return await status_msg.edit_text(
                f"🚫 **Vision Sync Error:** No temporal data found for this video.{hint}")

        # ── 5. Fused score curve → peak detection → best moment window ──
        scores, timestamps, indices = [], [], []
        best_frame_overall, max_tot_score = None, -1
        for idx, ts, depth, motion in db_frames:
            tot = s_scores_map.get(idx, 0.0) + c_scores_map.get(idx, 0.0)
            if best_chunk and (best_chunk["start_t"] <= ts <= best_chunk["end_t"]):
                tot += (bge_hits[0].score * 0.2)
            scores.append(tot)
            timestamps.append(ts)
            indices.append(idx)
            if tot > max_tot_score:
                max_tot_score = tot
                best_frame_overall = (idx, ts, depth, motion)

        smoothed = scipy.ndimage.gaussian_filter1d(np.array(scores), sigma=3)
        peaks, _ = scipy.signal.find_peaks(smoothed, prominence=0.01)

        best_peak_idx, best_ts, best_depth, best_motion = best_frame_overall
        best_start_t, best_end_t = max(0.0, best_ts - 1.5), best_ts + 1.5

        if len(peaks) > 0:
            _, _, left_ips, right_ips = scipy.signal.peak_widths(smoothed, peaks, rel_height=0.6)
            for i, peak_idx in enumerate(peaks):
                if (indices[int(left_ips[i])] <= best_peak_idx <= indices[int(right_ips[i])]
                        or indices[peak_idx] == best_peak_idx):
                    best_peak_idx = indices[peak_idx]
                    best_start_t = max(0.0, timestamps[int(left_ips[i])] - 1.0)
                    best_end_t = timestamps[int(right_ips[i])] + 1.0
                    break

        # ── 6. Spatial proof frame (OCR → DINO → SAM) ──
        cap = cv2.VideoCapture(best_vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_peak_idx)
        ret, frame_cv = cap.read()
        cap.release()
        if not ret or frame_cv is None:
            return await status_msg.edit_text("❌ **Frame Read Error:** could not decode proof frame.")

        rgb_frame = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        ocr_results = []
        if MODELS.get("ocr_reader"):
            try:
                ocr_results = MODELS["ocr_reader"].readtext(rgb_frame)
            except Exception as e:
                log(f"OCR failed on proof frame: {e}", "WARN")
        found_text = [t[1] for t in ocr_results]

        proof_arr, is_proven, proof_msg = hybrid_spatial_proof(
            Image.fromarray(rgb_frame), optimized_query, ocr_results)
        out_img_path = os.path.join(ARCHIVE_DIR, f"temp_proof_{uuid_lib.uuid4().hex}.jpg")
        cv2.imwrite(out_img_path, cv2.cvtColor(proof_arr, cv2.COLOR_RGB2BGR))
        await message.reply_photo(photo=out_img_path,
                                  caption=f"🎯 **[MODEL 6: Spatial Engine]**\n- {proof_msg}")
        os.remove(out_img_path)

        # ── 7. Render the answer subclip ──
        out_vid_path = os.path.join(ARCHIVE_DIR, f"temp_vid_{uuid_lib.uuid4().hex}.mp4")
        clip_dur = max(2.0, best_end_t - best_start_t)
        subprocess.run(["ffmpeg", "-ss", str(best_start_t), "-i", best_vid_path,
                        "-t", str(clip_dur), "-c:v", "libx264", "-preset", "ultrafast",
                        "-c:a", "aac", out_vid_path, "-y"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(out_vid_path) or os.path.getsize(out_vid_path) == 0:
            return await status_msg.edit_text("❌ **FFmpeg Rendering Error:** Subclip extraction failed.")

        # ── 8. Qwen local visual analysis of the subclip ──
        try:
            visual_analysis = qwen_describe_video(
                out_vid_path,
                f"Analyze this video clip based on the search: '{optimized_query}'. "
                f"What is physically occurring?",
                fps=2.0, max_new_tokens=250)
        except Exception as e:
            log(f"Qwen clip analysis failed: {e}", "WARN")
            visual_analysis = "(local visual analysis unavailable)"

        # ── 9. NIM GraphRAG synthesis with raw fallback ──
        video_metadata = (f"Visual Analysis: {visual_analysis} | OCR Found: {found_text} | "
                          f"Depth: {best_depth} | Motion: {best_motion}")
        rag_synthesis_prompt = PROMPTS["videorag_response_wo_reference"].format(
            response_type="A detailed, highly authoritative, and deeply reasoned analysis.",
            video_data=video_metadata, chunk_data=c_desc)
        try:
            oracle_answer = nim_chat(
                [{"role": "system", "content": "You are Omniscient AI, an elite intelligence matrix."},
                 {"role": "user", "content": rag_synthesis_prompt}],
                temperature=0.4, max_tokens=1000)
        except Exception as api_err:
            log(f"NIM synthesis error: {api_err}", "WARN")
            oracle_answer = (f"(⚠️ NIM Synthesis Offline - Showing Raw Visual Output)\n\n"
                             f"{visual_analysis}\n\nContext Fragment: {c_desc}")

        short_caption = (f"🔬 **Omni-Metadata (PostgreSQL)**:\n"
                         f"- Target: {os.path.basename(best_vid_path)}\n"
                         f"- Timestamp: ~{best_ts:.2f}s")
        await message.reply_video(video=out_vid_path, caption=short_caption)
        await send_long_message(client, message.chat.id,
                                f"🧠 **[God-Tier GraphRAG Synthesis]**\n\n{oracle_answer}")

        os.remove(out_vid_path)
        await status_msg.delete()
    except Exception:
        log(f"Search failed:\n{traceback.format_exc()}", "ERROR")
        try:
            await status_msg.edit_text(
                "⚠️ **Engine Error:** Failed to execute logic cascade. Check console logs.")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# UI UPDATER — live Telegram progress for bot-submitted jobs
# ═══════════════════════════════════════════════════════════
async def ui_updater_daemon():
    import asyncio as aio
    last_states = {}
    while True:
        try:
            keys = REDIS.keys("status:*")
            for key in keys:
                status = REDIS.hgetall(key)
                if status.get("notified") == "TRUE":
                    continue
                v_stat = status.get("vision", "WAITING")
                o_stat = status.get("oracle", "WAITING")
                done = "DONE" in v_stat and "DONE" in o_stat

                # Harvest jobs have no chat — close them out silently
                if "chat_id" not in status:
                    if done:
                        REDIS.hset(key, "notified", "TRUE")
                    continue

                chat_id, msg_id = int(status["chat_id"]), int(status["msg_id"])
                text = (f"⚙️ **Processing {status.get('mode', '').upper()} Pipeline:**\n\n"
                        f"👁️ **Vision Engine:** {v_stat}\n🧠 **Oracle Engine:** {o_stat}")
                if text != last_states.get(key):
                    try:
                        await app.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text)
                        last_states[key] = text
                    except Exception:
                        pass
                if done:
                    try:
                        await app.send_message(
                            chat_id, "✅ **Omniscient DB Updated!**\nVideo fully secured.")
                    except Exception:
                        pass
                    REDIS.hset(key, "notified", "TRUE")
        except Exception as e:
            log(f"UI updater error: {e}", "WARN")
        await aio.sleep(2.0)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    log("🚀 Omniscient Engine igniting — tri-partite DB + Layer 5 orchestration")

    # 0. Broker first — both worker loops and the bot push jobs immediately.
    if not wait_for_redis(label="OMNI"):
        log("❌ Redis unreachable — Omniscient engine cannot run. Exiting.", "ERROR")
        sys.exit(1)

    # 1. Databases (idempotent service start + schema)
    omni_db.ensure_services()
    omni_db.init_pg_schema()
    get_qdrant()

    # 2. Perception models (per-model failure tolerance)
    omni_models.load_all()

    # 3. Workers + dashboard
    threading.Thread(target=vision_worker_loop, daemon=True).start()
    threading.Thread(target=oracle_worker_loop, daemon=True).start()
    threading.Thread(target=run_dashboard, daemon=True).start()
    log(f"👁️ God-Mode Explorer on 127.0.0.1:{OMNI_DASHBOARD_PORT} → /omni tab in the workstation")

    # 4. Telegram bot (main asyncio loop)
    async def _run():
        await app.start()
        asyncio.create_task(ui_updater_daemon())
        log("⚡ OMNISCIENT ENGINE RUNNING — bot online, queues hot.", "SUCCESS")
        await idle()

    asyncio.get_event_loop().run_until_complete(_run())


if __name__ == "__main__":
    main()
