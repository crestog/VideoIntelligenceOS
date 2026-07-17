"""
VIOS Oracle Worker — Qwen2.5-VL-7B Narrative Generation + GraphRAG

Consumes QUEUE_ORACLE jobs and produces:
  1. Rich video narratives (per time-chunk) via Qwen2.5-VL-7B-Instruct (4-bit)
  2. BGE-large text embeddings → Qdrant chunks_bge collection
  3. Neo4j knowledge graph: (Video)-[:CONTAINS]->(Chunk)-[:DESCRIBED_BY]->(Narrative)
  4. GraphRAG entity/relationship extraction via NVIDIA NIM (Nemotron-3-Ultra)
     stored as (Entity)-[:RELATED_TO]->(Entity) with chunk linkage

Processing modes (set per job, defaulting to config.DEFAULT_ORACLE_MODE):
  blitz: 15s chunks, 1 fps, 75 max_new_tokens  — fast, low GPU memory
  omni:   5s chunks, 2 fps, 150 max_new_tokens  — rich detail

GraphRAG only runs in 'omni' mode unless VIOS_GRAPHRAG_OMNI_ONLY=0.
"""

import gc
import json
import math
import os
import subprocess
import time
import traceback

os.environ.setdefault("HF_HOME", "/kaggle/working/huggingface_cache")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import torch

from config import (
    QUEUE_ORACLE,
    NEO4J_BOLT_URL,
    NVIDIA_API_KEY, NIM_MODEL, NIM_BASE_URL,
    DEFAULT_ORACLE_MODE, GRAPHRAG_IN_OMNI_ONLY,
    SQLITE_TIMEOUT, DB_PATH,
)
from graphrag_prompts import PROMPTS
from logger import vios_log
from queue_manager import claim_job, ack_job, fail_job, job_heartbeat


# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════
def log(msg, level="INFO"):
    vios_log(msg, "ORACLE", level)


def clear_ram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# GPU ASSIGNMENT
# ═══════════════════════════════════════════════════════════
device_1 = (
    "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)

# ═══════════════════════════════════════════════════════════
# MODEL WARM-UP (Qwen2.5-VL-7B-Instruct, 4-bit quantized)
# ═══════════════════════════════════════════════════════════
QWEN_MODEL = None
QWEN_PROCESSOR = None
BGE_MODEL = None


def load_models():
    global QWEN_MODEL, QWEN_PROCESSOR, BGE_MODEL

    import shutil
    _free_gb = shutil.disk_usage("/kaggle/working").free / (1024**3)
    if _free_gb < 8:
        log(f"⛔ Skipping Qwen2.5-VL-7B — only {_free_gb:.1f}GB free, need ~8GB. "
            "Oracle narration disabled this session until disk space is freed.", "ERROR")
        return

    log("🧠 Loading Qwen2.5-VL-7B-Instruct (pre-quantized 4-bit)...")
    QWEN_REPO = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
    try:
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
        )
        QWEN_PROCESSOR = AutoProcessor.from_pretrained(QWEN_REPO)
        QWEN_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_REPO,
            torch_dtype=torch.bfloat16,
            device_map={"": device_1},
        ).eval()
        log("✅ Qwen2.5-VL-7B Oracle loaded", "SUCCESS")
    except Exception as e:
        log(f"❌ Qwen load failed: {e}", "ERROR")
        traceback.print_exc()

    log("📝 Loading BGE-large-en-v1.5...")
    try:
        from sentence_transformers import SentenceTransformer
        BGE_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5")
        if torch.cuda.is_available():
            BGE_MODEL = BGE_MODEL.to("cuda:0")
        log("✅ BGE-large loaded", "SUCCESS")
    except Exception as e:
        log(f"⚠️ BGE load failed (chunk search disabled): {e}", "WARN")


# ═══════════════════════════════════════════════════════════
# NIM CLIENT (lazy, avoids import errors when openai missing)
# ═══════════════════════════════════════════════════════════
_nim_client = None   # None = not checked yet | False = unavailable (cached) | client


def get_nim_client():
    """
    Return the NVIDIA NIM client, or None when GraphRAG is unavailable.
    The unavailable verdict is cached — without it, every chunk fired a
    doomed API call that failed with "Header of type `authorization` was
    missing" (the SDK sends no auth header when api_key is empty).
    """
    global _nim_client
    if _nim_client is False:
        return None
    if _nim_client is not None:
        return _nim_client
    if not NVIDIA_API_KEY:
        log("⚠️ VIOS_NVIDIA_API_KEY not set — GraphRAG entity extraction disabled "
            "(narratives + graph structure still stored)", "WARN")
        _nim_client = False
        return None
    try:
        from openai import OpenAI
        _nim_client = OpenAI(base_url=NIM_BASE_URL, api_key=NVIDIA_API_KEY)
        return _nim_client
    except ImportError:
        log("⚠️ openai package missing — GraphRAG NIM extraction disabled", "WARN")
        _nim_client = False
        return None


# ═══════════════════════════════════════════════════════════
# GRAPHRAG EXTRACTION
# ═══════════════════════════════════════════════════════════
def extract_and_store_graphrag(neo4j_driver, narrative_text: str, video_uuid: str, chunk_id: str):
    """
    Send narrative_text to Nemotron-3 via NIM for entity/relationship extraction,
    then write the results into Neo4j.
    """
    client = get_nim_client()
    if client is None or neo4j_driver is None:
        return

    extraction_prompt = PROMPTS["entity_extraction"].format(
        entity_types=", ".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        input_text=narrative_text,
    )
    try:
        resp = client.chat.completions.create(
            model=NIM_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        graph_output = resp.choices[0].message.content
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
                    name = parts[1].strip('"')
                    e_type = parts[2].strip('"')
                    desc = parts[3].strip('"')
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.description = $desc
                        WITH e
                        MATCH (c:Chunk {id: $cid})
                        MERGE (c)-[:CONTAINS_ENTITY]->(e)
                        """,
                        name=name, type=e_type, desc=desc, cid=chunk_id,
                    )

                elif record_type == "relationship" and len(parts) >= 5:
                    src = parts[1].strip('"')
                    tgt = parts[2].strip('"')
                    desc = parts[3].strip('"')
                    weight = parts[4].strip('"')
                    session.run(
                        """
                        MATCH (s:Entity {name: $src})
                        MATCH (t:Entity {name: $tgt})
                        MERGE (s)-[r:RELATED_TO]->(t)
                        SET r.description = $desc, r.weight = toInteger($weight)
                        """,
                        src=src, tgt=tgt, desc=desc, weight=weight,
                    )

    except Exception as e:
        log(f"⚠️ GraphRAG extraction failed for chunk {chunk_id}: {e}", "WARN")


# ═══════════════════════════════════════════════════════════
# QWEN INFERENCE — generate narrative for one video chunk
# ═══════════════════════════════════════════════════════════
def narrate_chunk(chunk_file: str, previous_context: str, fps: float, max_tokens: int) -> str:
    """
    Run Qwen2.5-VL on a single video chunk file.
    Returns the generated narrative string.
    """
    if QWEN_MODEL is None or QWEN_PROCESSOR is None:
        return ""

    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": chunk_file,
                "max_pixels": 360_000,
                "fps": fps,
            },
            {
                "type": "text",
                "text": (
                    f"Context: '{previous_context}'. "
                    "Write a highly intelligent, continuous narrative explaining "
                    "what is occurring now in this video segment."
                ),
            },
        ],
    }]

    text = QWEN_PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = QWEN_PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device_1)

    # Cast floating-point tensors to model dtype
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(QWEN_MODEL.dtype)

    with torch.no_grad():
        generated_ids = QWEN_MODEL.generate(**inputs, max_new_tokens=max_tokens)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    narrative = QWEN_PROCESSOR.batch_decode(trimmed, skip_special_tokens=True)[0]
    return narrative.strip()


# ═══════════════════════════════════════════════════════════
# CORE JOB PROCESSOR
# ═══════════════════════════════════════════════════════════
def process_oracle_job(payload: dict):
    msg_id = payload.get("msg_id")
    video_path = payload.get("path", "")
    v_uuid = payload.get("uuid", str(msg_id))
    mode = payload.get("mode", DEFAULT_ORACLE_MODE)  # 'blitz' | 'omni'

    if not video_path or not os.path.exists(video_path):
        log(f"⚠️ Oracle #{msg_id}: video file missing — skipping", "WARN")
        return

    # Mode parameters
    if mode == "omni":
        chunk_size, qwen_fps, max_tokens = 5.0, 2.0, 150
    else:  # blitz
        chunk_size, qwen_fps, max_tokens = 15.0, 1.0, 75

    run_graphrag = (mode == "omni") or not GRAPHRAG_IN_OMNI_ONLY

    # Probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "json", video_path],
        capture_output=True, text=True, timeout=30,
    )
    info = json.loads(probe.stdout or "{}")
    duration = float((info.get("format") or {}).get("duration", 0))
    if duration < 1.0:
        log(f"⚠️ Oracle #{msg_id}: video too short ({duration:.1f}s)", "WARN")
        return

    # Lazy DB imports
    try:
        from tripartite_db import get_qdrant, get_neo4j_driver, get_pg_conn
        qdrant = get_qdrant()
    except Exception:
        qdrant = None

    try:
        from tripartite_db import get_pg_conn
        pg_conn = get_pg_conn()
    except Exception:
        pg_conn = None

    neo4j_driver = None
    if run_graphrag:
        try:
            from tripartite_db import get_neo4j_driver
            neo4j_driver = get_neo4j_driver()
        except Exception:
            pass

    chunks_dir = os.path.join(os.path.dirname(video_path), f"oracle_{v_uuid}_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    previous_context = "This is the start of the video."
    bge_pending = []   # (point_id, narrative, payload) — encoded in one batch after the loop
    import sqlite3
    sqlite_rows = []

    try:
        from qdrant_client.models import PointStruct
    except ImportError:
        PointStruct = None

    t0 = time.time()
    for start_t in range(0, int(math.ceil(duration)), int(chunk_size)):
        if duration - start_t < 1.0:
            break

        end_t = min(start_t + chunk_size, duration)
        chunk_file = os.path.join(chunks_dir, f"chunk_{start_t}.mp4")

        # Extract chunk via ffmpeg — no audio track (-an): Qwen2.5-VL only
        # consumes video frames, so encoding AAC per chunk was pure waste
        subprocess.run(
            ["ffmpeg", "-ss", str(start_t), "-i", video_path,
             "-t", str(end_t - start_t),
             "-c:v", "libx264", "-preset", "ultrafast", "-an",
             chunk_file, "-y"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if not os.path.exists(chunk_file):
            continue

        # Generate narrative
        narrative = narrate_chunk(chunk_file, previous_context, qwen_fps, max_tokens)
        if not narrative:
            continue
        previous_context = narrative

        c_id_str = f"chunk_{v_uuid}_{start_t}"
        c_id_int = hash(c_id_str) & ((1 << 63) - 1)

        # BGE embedding — deferred: narratives are collected here and encoded
        # in ONE batched forward pass after the chunk loop (per-chunk encode
        # paid the full tokenize/transfer overhead for every single sentence)
        bge_pending.append((c_id_int, narrative, {
            "chunk_id": c_id_str,
            "video_path": video_path,
            "msg_id": msg_id,
            "start_t": start_t,
            "end_t": end_t,
            "mode": mode,
        }))

        # PostgreSQL insert
        if pg_conn is not None:
            try:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chunks
                            (chunk_id, video_uuid, video_path, msg_id, start_t, end_t, mode, description)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET description = EXCLUDED.description
                        """,
                        (c_id_str, v_uuid, video_path, msg_id, start_t, end_t, mode, narrative),
                    )
                pg_conn.commit()
            except Exception as e:
                log(f"⚠️ PG insert failed for {c_id_str}: {e}", "WARN")

        # SQLite mirror (so snapshot cycle covers narratives too)
        sqlite_rows.append((c_id_str, v_uuid, video_path, msg_id, start_t, end_t, mode, narrative))

        # Neo4j graph + GraphRAG
        if neo4j_driver is not None:
            try:
                with neo4j_driver.session() as session:
                    session.run(
                        """
                        MERGE (v:Video {uuid: $vid})
                        MERGE (c:Chunk {id: $cid, start: $start, end: $end})
                        MERGE (v)-[:CONTAINS]->(c)
                        MERGE (n:Narrative {text: $desc})
                        MERGE (c)-[:DESCRIBED_BY]->(n)
                        """,
                        vid=v_uuid, cid=c_id_str, start=start_t, end=end_t, desc=narrative,
                    )
                if run_graphrag:
                    extract_and_store_graphrag(neo4j_driver, narrative, v_uuid, c_id_str)
            except Exception as e:
                log(f"⚠️ Neo4j insert failed for {c_id_str}: {e}", "WARN")

    # Batch-encode all narratives in one BGE forward pass, then batch-upsert
    if bge_pending and BGE_MODEL is not None and qdrant is not None and PointStruct is not None:
        try:
            vecs = BGE_MODEL.encode([n for _, n, _ in bge_pending],
                                    normalize_embeddings=True, batch_size=32)
            bge_points = [
                PointStruct(id=pid, vector=vec.tolist(), payload=meta)
                for (pid, _n, meta), vec in zip(bge_pending, vecs)
            ]
            qdrant.upsert(collection_name="chunks_bge", points=bge_points)
            log(f"✅ Upserted {len(bge_points)} BGE chunk vectors (batched encode)")
        except Exception as e:
            log(f"⚠️ BGE encode/upsert failed: {e}", "WARN")

    # SQLite mirror
    if sqlite_rows:
        try:
            import sqlite3
            conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS oracle_chunks (
                    chunk_id TEXT PRIMARY KEY, video_uuid TEXT, video_path TEXT,
                    msg_id INTEGER, start_t REAL, end_t REAL, mode TEXT, description TEXT
                )
            """)
            conn.executemany(
                """
                INSERT OR REPLACE INTO oracle_chunks
                    (chunk_id, video_uuid, video_path, msg_id, start_t, end_t, mode, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                sqlite_rows,
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log(f"⚠️ SQLite oracle_chunks write failed: {e}", "WARN")

    # Cleanup chunk files
    import shutil
    shutil.rmtree(chunks_dir, ignore_errors=True)

    elapsed = time.time() - t0
    log(f"🏁 Oracle #{msg_id} DONE | {len(sqlite_rows)} chunks | {mode} | {elapsed:.1f}s", "SUCCESS")

    if pg_conn:
        pg_conn.close()
    if neo4j_driver:
        neo4j_driver.close()
    clear_ram()


# ═══════════════════════════════════════════════════════════
# MAIN — warm-up then job loop
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("🚀 Oracle Worker starting — loading models...")
    load_models()
    log(f"✅ Oracle Worker ready. Consuming {QUEUE_ORACLE}...")

    while True:
        try:
            job, job_raw = claim_job(QUEUE_ORACLE, timeout=5)
        except Exception as e:
            log(f"⚠️ Queue error: {type(e).__name__} — retrying in 3s", "WARN")
            time.sleep(3)
            continue

        if not job:
            continue

        payload = job.get("payload", job)
        msg_id = payload.get("msg_id", "?")
        log(f"📥 ORACLE JOB: Video #{msg_id} mode={payload.get('mode', DEFAULT_ORACLE_MODE)}")

        try:
            _t0 = time.time()
            with job_heartbeat(QUEUE_ORACLE, job.get("id"), "oracle_worker"):
                process_oracle_job(payload)
            ack_job(QUEUE_ORACLE, job, job_raw)
            log(f"🏁 ORACLE #{msg_id} COMPLETE", "SUCCESS")
            try:
                from db_schema import record_event
                record_event(msg_id=msg_id, stage="narrated", status="completed",
                             worker="oracle_worker",
                             duration_sec=round(time.time() - _t0, 2))
            except Exception:
                pass
        except Exception as e:
            result = fail_job(QUEUE_ORACLE, job, job_raw, str(e))
            log(f"❌ ORACLE #{msg_id} FAILED → {result}: {str(e)[:200]}", "ERROR")
            try:
                from db_schema import record_event
                record_event(msg_id=msg_id, stage="narrated", status="failed",
                             worker="oracle_worker", detail=f"{result}: {str(e)[:200]}")
            except Exception:
                pass
            clear_ram()
            time.sleep(2)
