"""
VIOS Model Manager v2 — GPU Model Warm-up + Analysis Worker

Phase 1 (boot): load the 7 foundational models into VRAM (heaviest first).
Phase 2 (loop): consume QUEUE_ANALYZE jobs pushed by frame_worker after
extraction. For each video:

  GPU 1 (this process):
    • Whisper-Large-v3  → full transcript with word-level timestamps
                          → `transcripts` table (one row per segment)
    • YOLO11x           → objects per sampled frame (2/sec)
    • EasyOCR           → on-screen text per sampled frame
                          → `frame_notes` table (one row per sampled frame)

frame_notes is what powers the Workstation's per-frame description panel:
every sampled frame gets a timestamp + "what is happening" summary built
from detected objects + OCR text. Embedding/semantic models (SigLIP, CLIP,
DINOv2) stay warm for the moment-search indexer stage.

All writes are idempotent (DELETE + INSERT per msg_id), so re-running a job
after adding new models simply refreshes that video's rows.
"""

import os
import gc
import glob
import json
import sqlite3
import time

# Professional Logging Filter: Mute raw library noise
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("HF_HOME", "/kaggle/working/huggingface_cache")

import torch

from queue_manager import claim_job, ack_job, fail_job, pop_job, push_job
from config import DB_PATH, VIDEO_DIR, QUEUE_ANALYZE, SQLITE_TIMEOUT, PREVIEW_DIR_NAME
from logger import vios_log

device_0 = "cuda:0" if torch.cuda.is_available() else "cpu"
device_1 = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device_0
WARM_MODELS = {}

ANALYZE_SAMPLE_FPS = 2.0   # sampled frames per second for object/OCR notes


def log(msg, level="INFO"):
    vios_log(msg, "AI", level)


def clear_ram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# THE 7 SOTA VISION/AUDIO ENGINES
#   GPU 0: vision (YOLO, DINOv2, SigLIP, CLIP, RAFT, EasyOCR)
#   GPU 1: reasoning/audio (Whisper) — per the 2×T4 split
# ═══════════════════════════════════════════════════════════
def load_siglip():
    from transformers import AutoModel
    log("👁️ Loading SigLIP2...")
    WARM_MODELS['siglip_model'] = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384", torch_dtype=torch.float16).to(device_0).eval()

def load_clip():
    from transformers import CLIPModel
    log("👁️ Loading CLIP...")
    WARM_MODELS['clip_model'] = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device_0).eval()

def load_dinov2():
    from transformers import AutoModel
    log("🧩 Loading DINOv2-Large...")
    WARM_MODELS['dino_model'] = AutoModel.from_pretrained(
        "facebook/dinov2-large", torch_dtype=torch.float16).to(device_0).eval()

def load_whisper():
    from faster_whisper import WhisperModel
    log("🎙️ Loading Whisper-Large-v3...")
    dev_idx = 1 if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else 0
    WARM_MODELS['whisper_model'] = WhisperModel(
        "large-v3", device="cuda" if torch.cuda.is_available() else "cpu",
        device_index=dev_idx, compute_type="float16" if torch.cuda.is_available() else "int8")

def load_raft():
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    log("🌊 Loading RAFT-Large...")
    weights = Raft_Large_Weights.DEFAULT
    WARM_MODELS['raft_transforms'] = weights.transforms()
    WARM_MODELS['raft_model'] = raft_large(weights=weights, progress=False).to(device_0).eval()

def load_yolo():
    from ultralytics import YOLO
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    log("🎯 Loading YOLOv11x...")
    WARM_MODELS['yolo_model'] = YOLO("yolo11x.pt")
    WARM_MODELS['yolo_model'].to(device_0)

def load_easyocr():
    import easyocr
    import logging
    logging.getLogger("easyocr").setLevel(logging.ERROR)
    log("🔤 Loading EasyOCR...")
    WARM_MODELS['ocr_reader'] = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)


def load_grounding_dino():
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
    log("📍 Loading Grounding DINO-Base...")
    WARM_MODELS['dino_processor'] = GroundingDinoProcessor.from_pretrained(
        "IDEA-Research/grounding-dino-base")
    WARM_MODELS['dino_model'] = GroundingDinoForObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to(device_0).eval()


def load_sam():
    from transformers import SamModel, SamProcessor
    log("✂️  Loading SAM-ViT-Base...")
    WARM_MODELS['sam_processor'] = SamProcessor.from_pretrained("facebook/sam-vit-base")
    WARM_MODELS['sam_model'] = SamModel.from_pretrained(
        "facebook/sam-vit-base").to(device_0).eval()


def load_depth_anything():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    log("📏 Loading Depth-Anything-V2-Small...")
    WARM_MODELS['depth_proc'] = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf")
    WARM_MODELS['depth_model'] = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf").to(device_0).eval()

MODEL_REGISTRY = {
    "siglip":           load_siglip,
    "clip":             load_clip,
    "dinov2":           load_dinov2,
    "whisper":          load_whisper,
    "raft":             load_raft,
    "yolo":             load_yolo,
    "easyocr":          load_easyocr,
    "grounding_dino":   load_grounding_dino,
    "sam":              load_sam,
    "depth_anything":   load_depth_anything,
}


# ═══════════════════════════════════════════════════════════
# SPATIAL PROOF UTILITY (Grounding DINO → SAM overlay)
# ═══════════════════════════════════════════════════════════
def hybrid_spatial_proof(pil_image, text_query: str, ocr_results: list):
    """
    Localise `text_query` in `pil_image` using:
      1. EasyOCR result list  (text match → blue mask)
      2. Grounding DINO       (object match → green mask)
    Then refine the bounding box with SAM to produce a pixel-level mask.

    Returns (np.ndarray uint8 RGB, success: bool, message: str).
    """
    import numpy as np
    import cv2

    img_arr = np.array(pil_image)
    query_lower = text_query.lower()
    best_box = None
    box_type = None

    # 1. OCR hit
    for (bbox, text, conf) in ocr_results:
        if query_lower in text.lower() or any(
            word in text.lower() for word in query_lower.split() if len(word) > 3
        ):
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            best_box = [min(xs), min(ys), max(xs), max(ys)]
            box_type = "OCR"
            break

    # 2. Grounding DINO fallback
    if not best_box:
        dino_proc  = WARM_MODELS.get('dino_processor')
        dino_model = WARM_MODELS.get('dino_model')
        if dino_proc and dino_model:
            try:
                inputs = dino_proc(
                    images=pil_image, text=query_lower + ".", return_tensors="pt"
                ).to(device_0)
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                results = dino_proc.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    threshold=0.15, text_threshold=0.15,
                    target_sizes=[pil_image.size[::-1]],
                )[0]
                if len(results["boxes"]) > 0:
                    best_box = results["boxes"][0].tolist()
                    box_type = "DINO"
            except Exception as e:
                log(f"⚠️ Grounding DINO inference failed: {e}", "WARN")

    if not best_box:
        return img_arr.astype(np.uint8), False, "❌ No text or object boundary detected."

    # 3. SAM refinement
    sam_proc  = WARM_MODELS.get('sam_processor')
    sam_model = WARM_MODELS.get('sam_model')
    if sam_proc and sam_model:
        try:
            sam_inputs = sam_proc(
                images=pil_image, input_boxes=[[best_box]], return_tensors="pt"
            ).to(device_0)
            with torch.no_grad():
                sam_out = sam_model(**sam_inputs)
            mask = sam_proc.image_processor.post_process_masks(
                sam_out.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )[0][0][0].numpy()
            color = np.array([255, 50, 50]) if box_type == "OCR" else np.array([50, 255, 50])
            img_arr = img_arr.copy()
            img_arr[mask] = (img_arr[mask] * 0.6 + color * 0.4).clip(0, 255)
        except Exception as e:
            log(f"⚠️ SAM mask failed: {e}", "WARN")

    # Draw bounding box
    color_bgr = [50, 50, 255] if box_type == "OCR" else [50, 255, 50]
    cv2.rectangle(
        img_arr,
        (int(best_box[0]), int(best_box[1])),
        (int(best_box[2]), int(best_box[3])),
        color_bgr, 3,
    )
    msg = (
        f"🔤 EasyOCR matched → blue spatial mask."
        if box_type == "OCR"
        else f"🧩 Grounding DINO matched → green spatial mask."
    )
    return img_arr.astype(np.uint8), True, msg


# ═══════════════════════════════════════════════════════════
# ANALYSIS TABLES
# ═══════════════════════════════════════════════════════════
def ensure_analysis_tables():
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    try:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS transcripts
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      msg_id INTEGER, start_sec REAL, end_sec REAL,
                      text TEXT, created_at REAL)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_transcripts_msg ON transcripts(msg_id)')
        c.execute('''CREATE TABLE IF NOT EXISTS frame_notes
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      msg_id INTEGER, frame_idx INTEGER, ts_sec REAL,
                      objects TEXT,      -- JSON: [{label, conf}, ...]
                      ocr_text TEXT,     -- concatenated on-screen text
                      description TEXT,  -- human-readable "what is happening"
                      created_at REAL)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_frame_notes_msg ON frame_notes(msg_id)')
        c.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS moments_search USING fts5
                     (msg_id UNINDEXED, ts_sec UNINDEXED, source UNINDEXED, content)''')
        conn.commit()
    finally:
        conn.close()


def _rewrite_moments(c, msg_id):
    """Rebuild the FTS moment index for one video from transcripts + notes."""
    c.execute("DELETE FROM moments_search WHERE msg_id = ?", (msg_id,))
    c.execute("SELECT start_sec, text FROM transcripts WHERE msg_id = ?", (msg_id,))
    for start, text in c.fetchall():
        if text and text.strip():
            c.execute("INSERT INTO moments_search (msg_id, ts_sec, source, content) VALUES (?, ?, 'speech', ?)",
                      (msg_id, start, text.strip()))
    c.execute("SELECT ts_sec, description, ocr_text FROM frame_notes WHERE msg_id = ?", (msg_id,))
    for ts, desc, ocr in c.fetchall():
        blob = " ".join(x for x in (desc, ocr) if x)
        if blob.strip():
            c.execute("INSERT INTO moments_search (msg_id, ts_sec, source, content) VALUES (?, ?, 'visual', ?)",
                      (msg_id, ts, blob.strip()))


# ═══════════════════════════════════════════════════════════
# ANALYSIS STAGES
# ═══════════════════════════════════════════════════════════
def analyze_audio(msg_id, video_path):
    """Whisper transcript → transcripts table. Returns segment count."""
    whisper = WARM_MODELS.get('whisper_model')
    if not whisper:
        return 0
    segments, _info = whisper.transcribe(video_path, word_timestamps=False, vad_filter=True)
    rows = [(msg_id, s.start, s.end, s.text.strip(), time.time())
            for s in segments if s.text and s.text.strip()]

    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    try:
        c = conn.cursor()
        c.execute("DELETE FROM transcripts WHERE msg_id = ?", (msg_id,))
        c.executemany("INSERT INTO transcripts (msg_id, start_sec, end_sec, text, created_at) VALUES (?, ?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def _sample_frames(folder_id, fps):
    """Yield (frame_idx, ts_sec, full_res_path) at ~ANALYZE_SAMPLE_FPS."""
    frames_dir = os.path.join(VIDEO_DIR, folder_id)
    frames = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.jpg')))
    if not frames:
        return
    step = max(1, round(fps / ANALYZE_SAMPLE_FPS)) if fps > 0 else 15
    for i in range(0, len(frames), step):
        name = os.path.basename(frames[i])
        try:
            ts = float(name.split('_ts_')[1].replace('s.jpg', ''))
        except (IndexError, ValueError):
            ts = i / fps if fps > 0 else 0.0
        yield i, ts, frames[i]


def analyze_frames(msg_id, folder_id, fps):
    """YOLO + OCR per sampled frame → frame_notes table. Returns note count."""
    yolo = WARM_MODELS.get('yolo_model')
    ocr = WARM_MODELS.get('ocr_reader')
    notes = []

    for idx, ts, path in _sample_frames(folder_id, fps):
        objects = []
        ocr_text = ""
        if yolo:
            try:
                res = yolo.predict(path, verbose=False, conf=0.35)[0]
                names = res.names
                for b in res.boxes:
                    objects.append({"label": names[int(b.cls)], "conf": round(float(b.conf), 2)})
            except Exception:
                pass
        if ocr:
            try:
                ocr_text = " ".join(t[1] for t in ocr.readtext(path) if t[2] > 0.4)
            except Exception:
                pass

        # Human-readable description: object counts + screen text
        counts = {}
        for o in objects:
            counts[o["label"]] = counts.get(o["label"], 0) + 1
        parts = [f"{v}× {k}" if v > 1 else k for k, v in sorted(counts.items(), key=lambda x: -x[1])]
        desc = ", ".join(parts) if parts else ""
        if ocr_text:
            desc = (desc + " — " if desc else "") + f'on-screen text: "{ocr_text[:160]}"'
        if not desc:
            desc = "no salient objects or text detected"

        notes.append((msg_id, idx, ts, json.dumps(objects), ocr_text, desc, time.time()))

    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    try:
        c = conn.cursor()
        c.execute("DELETE FROM frame_notes WHERE msg_id = ?", (msg_id,))
        c.executemany('''INSERT INTO frame_notes
                         (msg_id, frame_idx, ts_sec, objects, ocr_text, description, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''', notes)
        _rewrite_moments(c, msg_id)
        conn.commit()
    finally:
        conn.close()
    return len(notes)


def process_analyze_job(payload):
    msg_id = payload.get("msg_id")
    video_path = payload.get("path")
    folder_id = payload.get("folder_id", f"frames_{msg_id}")

    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    try:
        row = conn.execute("SELECT fps FROM videos WHERE msg_id = ?", (msg_id,)).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        conn.close()
    fps = (row[0] if row and row[0] else 30.0)

    t0 = time.time()
    seg_count = 0
    if video_path and os.path.exists(video_path):
        seg_count = analyze_audio(msg_id, video_path)
        log(f"🎙️ #{msg_id}: {seg_count} transcript segments")
    else:
        log(f"⚠️ #{msg_id}: video file missing — skipping transcript (frames only)", "WARN")

    note_count = analyze_frames(msg_id, folder_id, fps)
    log(f"🧠 #{msg_id}: {note_count} frame notes | analysis done in {time.time() - t0:.1f}s")
    clear_ram()


# ═══════════════════════════════════════════════════════════
# MAIN — warm-up then analysis loop
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("🚀 Model Manager v2: warming the 7 foundational models...")
    ensure_analysis_tables()

    for model in ["yolo", "whisper", "dinov2", "siglip", "clip", "raft", "easyocr",
                  "grounding_dino", "sam", "depth_anything"]:
        push_job("QUEUE_MODELS", {"model_name": model})

    # Phase 1: drain QUEUE_MODELS (blocking until empty)
    while True:
        job = pop_job("QUEUE_MODELS", timeout=2)
        if not job:
            break
        m_name = job.get("model_name")
        if m_name in MODEL_REGISTRY:
            try:
                MODEL_REGISTRY[m_name]()
                clear_ram()
                log(f"✅ {m_name.upper()} loaded into VRAM.", "SUCCESS")
            except Exception as e:
                log(f"❌ Failed to load {m_name}: {e}", "ERROR")
                clear_ram()

    log(f"🧠 Warm-up complete ({len(WARM_MODELS)} engines). Consuming {QUEUE_ANALYZE}...")

    # Phase 2: reliable analysis loop
    while True:
        try:
            job, job_raw = claim_job(QUEUE_ANALYZE, timeout=5)
        except Exception as e:
            log(f"⚠️ Queue error: {type(e).__name__} — retrying in 3s...", "WARN")
            time.sleep(3)
            continue
        if not job:
            continue

        payload = job.get("payload", job)
        msg_id = payload.get("msg_id", "?")
        log(f"📥 ANALYZE JOB: Video #{msg_id}")
        try:
            process_analyze_job(payload)
            ack_job(QUEUE_ANALYZE, job, job_raw)
            log(f"🏁 ANALYZE #{msg_id} COMPLETE", "SUCCESS")
        except Exception as e:
            result = fail_job(QUEUE_ANALYZE, job, job_raw, str(e))
            log(f"❌ ANALYZE #{msg_id} FAILED → {result} | {str(e)[:200]}", "ERROR")
            clear_ram()
            time.sleep(2)
