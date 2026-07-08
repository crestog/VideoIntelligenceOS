"""
VIOS Vision Embed Worker — SigLIP + CLIP + Depth-Anything + RAFT

Consumes QUEUE_VISION_EMBED jobs (pushed by frame_worker after dual-tier
frame extraction) and produces per-frame vector embeddings + depth/motion
scores stored across Qdrant and PostgreSQL.

Per sampled frame:
  SigLIP-SO400M   → 1152-dim vector → Qdrant: frames_siglip
  CLIP-ViT-L-14   →  768-dim vector → Qdrant: frames_clip
  Depth-Anything-V2-Small → mean depth → PostgreSQL: frames.depth
  RAFT-Large      → optical flow mag → PostgreSQL: frames.motion

All models run on GPU 0 (vision GPU), same as the existing YOLO/OCR pipeline.
Batching is used for embedding models (32 frames/batch) to maximise throughput.
"""

import gc
import glob
import json
import os
import time
import traceback

os.environ.setdefault("HF_HOME", "/kaggle/working/huggingface_cache")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import torch
from PIL import Image

from config import (
    QUEUE_VISION_EMBED,
    VIDEO_DIR,
    PREVIEW_DIR_NAME,
    SQLITE_TIMEOUT, DB_PATH,
)
from logger import vios_log
from queue_manager import claim_job, ack_job, fail_job


# ═══════════════════════════════════════════════════════════
# LOGGING + GPU
# ═══════════════════════════════════════════════════════════
def log(msg, level="INFO"):
    vios_log(msg, "EMBED", level)


def clear_ram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


device_0 = "cuda:0" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════
# MODEL REGISTRY (lazy-loaded once at startup)
# ═══════════════════════════════════════════════════════════
MODELS = {}


def load_models():
    """Load all 4 vision models. Failures are non-fatal (model stays None)."""

    # 1. SigLIP-SO400M
    log("👁️ Loading SigLIP-SO400M-patch14-384...")
    try:
        from transformers import AutoModel, AutoProcessor
        MODELS["siglip_model"] = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            torch_dtype=torch.float16,
        ).to(device_0).eval()
        MODELS["siglip_proc"] = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        log("✅ SigLIP loaded", "SUCCESS")
    except Exception as e:
        log(f"⚠️ SigLIP load failed: {e}", "WARN")

    # 2. CLIP-ViT-Large-patch14
    log("👁️ Loading CLIP-ViT-Large-patch14...")
    try:
        from transformers import CLIPModel, CLIPProcessor
        MODELS["clip_model"] = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16,
        ).to(device_0).eval()
        MODELS["clip_proc"] = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        log("✅ CLIP loaded", "SUCCESS")
    except Exception as e:
        log(f"⚠️ CLIP load failed: {e}", "WARN")

    # 3. Depth-Anything-V2-Small
    log("📏 Loading Depth-Anything-V2-Small-hf...")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        MODELS["depth_proc"] = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        MODELS["depth_model"] = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        ).to(device_0).eval()
        log("✅ Depth-Anything-V2 loaded", "SUCCESS")
    except Exception as e:
        log(f"⚠️ Depth-Anything load failed: {e}", "WARN")

    # 4. RAFT-Large (optical flow)
    log("🌊 Loading RAFT-Large...")
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        weights = Raft_Large_Weights.DEFAULT
        MODELS["raft_transforms"] = weights.transforms()
        MODELS["raft_model"] = raft_large(weights=weights, progress=False).to(device_0).eval()
        log("✅ RAFT-Large loaded", "SUCCESS")
    except Exception as e:
        log(f"⚠️ RAFT load failed: {e}", "WARN")


# ═══════════════════════════════════════════════════════════
# EMBEDDING UTILITIES
# ═══════════════════════════════════════════════════════════
def _embed_images(model_key, proc_key, pil_imgs):
    """Return L2-normalised embedding matrix (N, dim) as numpy, or None."""
    model = MODELS.get(model_key)
    proc = MODELS.get(proc_key)
    if model is None or proc is None:
        return None
    try:
        inputs = proc(images=pil_imgs, return_tensors="pt").to(device_0)
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if feats.ndim == 3:
                feats = feats.mean(dim=1)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.cpu().float().numpy()
    except Exception as e:
        log(f"⚠️ Embed error ({model_key}): {e}", "WARN")
        return None


def _depth_score(pil_img):
    """Return mean predicted depth (scalar) for one PIL image."""
    model = MODELS.get("depth_model")
    proc = MODELS.get("depth_proc")
    if model is None or proc is None:
        return 0.0
    try:
        inp = proc(images=pil_img, return_tensors="pt").to(device_0)
        with torch.no_grad():
            return float(model(**inp).predicted_depth.mean().item())
    except Exception:
        return 0.0


def _motion_score(pil_prev, pil_curr):
    """Return mean optical flow magnitude between two PIL images."""
    raft = MODELS.get("raft_model")
    transforms = MODELS.get("raft_transforms")
    if raft is None or transforms is None:
        return 0.0
    try:
        t1 = torch.tensor(np.array(pil_prev)).permute(2, 0, 1).unsqueeze(0).to(device_0)
        t2 = torch.tensor(np.array(pil_curr)).permute(2, 0, 1).unsqueeze(0).to(device_0)
        i1, i2 = transforms(t1, t2)
        with torch.no_grad():
            flow = raft(i1, i2)[-1]
        return float(torch.norm(flow, dim=1).mean().item())
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════
# FRAME COLLECTION
# ═══════════════════════════════════════════════════════════
def collect_frames(folder_id: str, fps: float, sample_fps: float = 2.0):
    """
    Yield (frame_idx, ts_sec, pil_image_384) for sampled frames in the full tier.
    Uses the same frame-naming convention as frame_worker.py.
    """
    frames_dir = os.path.join(VIDEO_DIR, folder_id)
    if not os.path.isdir(frames_dir):
        log(f"⚠️ Frames directory not found: {frames_dir}", "WARN")
        return

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        return

    step = max(1, round(fps / sample_fps)) if fps > 0 else 15

    for i, path in enumerate(frame_files):
        if i % step != 0:
            continue
        name = os.path.basename(path)
        try:
            ts = float(name.split("_ts_")[1].replace("s.jpg", ""))
        except (IndexError, ValueError):
            ts = i / fps if fps > 0 else float(i)

        try:
            pil = Image.open(path).convert("RGB").resize((384, 384))
            yield i, ts, pil
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# CORE JOB PROCESSOR
# ═══════════════════════════════════════════════════════════
EMBED_BATCH = 32   # frames per embedding batch


def process_embed_job(payload: dict):
    msg_id = payload.get("msg_id")
    v_uuid = payload.get("uuid", str(msg_id))
    video_path = payload.get("path", "")
    folder_id = payload.get("folder_id", f"frames_{msg_id}")

    import sqlite3
    conn_s = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    row = conn_s.execute("SELECT fps FROM videos WHERE msg_id = ?", (msg_id,)).fetchone()
    conn_s.close()
    fps = (row[0] if row and row[0] else 30.0)

    # Collect all sampled frames into memory (idx, ts, pil)
    all_frames = list(collect_frames(folder_id, fps))
    if not all_frames:
        log(f"⚠️ Embed #{msg_id}: no frames found in {folder_id}", "WARN")
        return

    total = len(all_frames)
    log(f"📐 Embed #{msg_id}: processing {total} sampled frames...")

    # Lazy DB imports
    try:
        from tripartite_db import get_qdrant, get_pg_conn
        qdrant = get_qdrant()
        pg_conn = get_pg_conn()
    except Exception:
        qdrant = None
        pg_conn = None

    try:
        from qdrant_client.models import PointStruct
    except ImportError:
        PointStruct = None

    siglip_points = []
    clip_points = []
    pg_rows = []   # (frame_id, video_uuid, video_path, msg_id, timestamp, frame_idx, depth, motion)

    # Compute depth + motion per frame (sequential, each frame is cheap)
    depths = []
    motions = []
    for i, (f_idx, ts, pil) in enumerate(all_frames):
        depth = _depth_score(pil)
        motion = 0.0
        if i > 0:
            _, _, prev_pil = all_frames[i - 1]
            motion = _motion_score(prev_pil, pil)
        depths.append(round(depth, 3))
        motions.append(round(motion, 3))
        if i % 20 == 0:
            clear_ram()

    # Batch through SigLIP + CLIP
    for batch_start in range(0, total, EMBED_BATCH):
        batch = all_frames[batch_start: batch_start + EMBED_BATCH]
        pil_batch = [b[2] for b in batch]

        s_vecs = _embed_images("siglip_model", "siglip_proc", pil_batch)
        c_vecs = _embed_images("clip_model", "clip_proc", pil_batch)

        for j, (f_idx, ts, _pil) in enumerate(batch):
            global_j = batch_start + j
            d = depths[global_j]
            m = motions[global_j]

            f_id_str = f"{v_uuid}_{f_idx}"
            f_id_int = hash(f_id_str) & ((1 << 63) - 1)

            payload_meta = {
                "frame_id": f_id_str,
                "video_path": video_path,
                "msg_id": msg_id,
                "timestamp": ts,
                "frame_idx": f_idx,
            }

            if s_vecs is not None and qdrant is not None and PointStruct is not None:
                siglip_points.append(PointStruct(
                    id=f_id_int, vector=s_vecs[j].tolist(), payload=payload_meta
                ))
            if c_vecs is not None and qdrant is not None and PointStruct is not None:
                clip_points.append(PointStruct(
                    id=f_id_int, vector=c_vecs[j].tolist(), payload=payload_meta
                ))

            pg_rows.append((f_id_str, v_uuid, video_path, msg_id, ts, f_idx, d, m))

        if batch_start % (EMBED_BATCH * 4) == 0:
            clear_ram()

    # Qdrant batch upsert
    if qdrant is not None:
        if siglip_points:
            try:
                qdrant.upsert(collection_name="frames_siglip", points=siglip_points)
                log(f"✅ Upserted {len(siglip_points)} SigLIP frame vectors")
            except Exception as e:
                log(f"⚠️ Qdrant SigLIP upsert failed: {e}", "WARN")
        if clip_points:
            try:
                qdrant.upsert(collection_name="frames_clip", points=clip_points)
                log(f"✅ Upserted {len(clip_points)} CLIP frame vectors")
            except Exception as e:
                log(f"⚠️ Qdrant CLIP upsert failed: {e}", "WARN")

    # PostgreSQL batch insert
    if pg_conn is not None and pg_rows:
        try:
            with pg_conn.cursor() as cur:
                for row in pg_rows:
                    cur.execute(
                        """
                        INSERT INTO frames
                            (frame_id, video_uuid, video_path, msg_id, timestamp, frame_idx, depth, motion)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (frame_id) DO UPDATE
                            SET depth = EXCLUDED.depth, motion = EXCLUDED.motion
                        """,
                        row,
                    )
            pg_conn.commit()
            log(f"✅ PostgreSQL: {len(pg_rows)} frame rows written")
        except Exception as e:
            log(f"⚠️ PG frame insert failed: {e}", "WARN")
        finally:
            pg_conn.close()

    clear_ram()
    log(f"🏁 Embed #{msg_id} COMPLETE — {total} frames embedded", "SUCCESS")


# ═══════════════════════════════════════════════════════════
# MAIN — warm-up then job loop
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("🚀 Vision Embed Worker starting — loading models...")
    load_models()
    log(f"✅ Embed Worker ready. Consuming {QUEUE_VISION_EMBED}...")

    while True:
        try:
            job, job_raw = claim_job(QUEUE_VISION_EMBED, timeout=5)
        except Exception as e:
            log(f"⚠️ Queue error: {type(e).__name__} — retrying in 3s", "WARN")
            time.sleep(3)
            continue

        if not job:
            continue

        payload = job.get("payload", job)
        msg_id = payload.get("msg_id", "?")
        log(f"📥 EMBED JOB: Video #{msg_id}")

        try:
            process_embed_job(payload)
            ack_job(QUEUE_VISION_EMBED, job, job_raw)
            log(f"🏁 EMBED #{msg_id} COMPLETE", "SUCCESS")
        except Exception as e:
            result = fail_job(QUEUE_VISION_EMBED, job, job_raw, str(e))
            log(f"❌ EMBED #{msg_id} FAILED → {result}: {str(e)[:200]}", "ERROR")
            clear_ram()
            time.sleep(2)
