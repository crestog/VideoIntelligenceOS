"""
VIOS Omniscient — Perception Model Layer

GPU 0 (vision stack):
  GroundingDINO-base   — open-vocabulary object detection (spatial proof)
  SAM ViT-base         — segmentation masks over detected boxes
  Depth-Anything-V2-S  — per-frame mean depth
  RAFT-Large           — optical flow / motion magnitude
  SigLIP so400m        — image/text embeddings (1152-d)
  CLIP ViT-L/14        — image/text embeddings (768-d)
  BGE-large-en-v1.5    — narrative text embeddings (1024-d)
  EasyOCR              — on-screen text

GPU 1 (oracle):
  Qwen2.5-VL-7B-Instruct (4-bit NF4) — video narration + clip analysis

Qwen generation is serialized behind QWEN_LOCK: the oracle worker thread and
the bot's search handler share the model, and concurrent .generate() calls on
one 4-bit model are a corruption/OOM hazard.
"""

import os
import re
import threading

# config before torch: importing it runs configure_environment(), which sets
# HF_HOME/HF_HUB_CACHE/SENTENCE_TRANSFORMERS_HOME on the scratch disk. This
# module pulls the heaviest weights in the system (Qwen2.5-VL-7B alone is
# 16.6 GB), so if this import lands after torch the whole download goes to the
# 20 GB output quota and the session runs out of disk mid-load.
from config import HF_TOKEN, MODEL_CACHE_DIR

import numpy as np
import torch
import cv2

from atlas.hfcompat import projected
from logger import vios_log

MODELS = {}
QWEN_LOCK = threading.Lock()

device_0 = "cuda:0" if torch.cuda.is_available() else "cpu"
device_1 = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device_0


def log(msg, level="INFO"):
    vios_log(msg, "OMNI", level)


# ═══════════════════════════════════════════════════════════
# LOADING — heaviest first per VRAM-stacking discipline
# ═══════════════════════════════════════════════════════════
def load_all():
    """Warm every Omniscient model. Failures are per-model, not fatal."""
    # Only log in when the token is not already the active one. huggingface_hub
    # reads HF_TOKEN from the environment by itself, and calling login() on top
    # of that prints a confusing "Environment variable HF_TOKEN is set and is
    # the current active token independently..." note on every boot.
    try:
        if HF_TOKEN and not os.environ.get("HF_TOKEN"):
            from huggingface_hub import login
            login(token=HF_TOKEN)
    except Exception as e:
        log(f"HF login skipped: {e}", "WARN")

    loaders = [
        ("qwen_oracle", _load_qwen),
        ("bge", _load_bge),
        ("siglip", _load_siglip),
        ("grounding_dino", _load_dino),
        ("clip", _load_clip),
        ("sam", _load_sam),
        ("depth", _load_depth),
        ("raft", _load_raft),
        ("easyocr", _load_easyocr),
    ]
    for name, fn in loaders:
        try:
            fn()
            log(f"✅ {name} warm", "SUCCESS")
        except Exception as e:
            log(f"❌ {name} failed to load: {e}", "ERROR")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log(f"Perception engine online — {len(MODELS)} model groups loaded", "SUCCESS")
    return MODELS


def _load_dino():
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
    MODELS["dino_processor"] = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    MODELS["dino_model"] = GroundingDinoForObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to(device_0).eval()


def _load_sam():
    from transformers import SamModel, SamProcessor
    MODELS["sam_processor"] = SamProcessor.from_pretrained("facebook/sam-vit-base")
    MODELS["sam_model"] = SamModel.from_pretrained("facebook/sam-vit-base").to(device_0).eval()


def _load_depth():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    MODELS["depth_processor"] = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    MODELS["depth_model"] = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf").to(device_0).eval()


def _load_raft():
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    weights = Raft_Large_Weights.DEFAULT
    MODELS["raft_transforms"] = weights.transforms()
    MODELS["raft_model"] = raft_large(weights=weights, progress=False).to(device_0).eval()


def _load_siglip():
    from transformers import AutoModel, AutoProcessor
    MODELS["siglip_model"] = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384", torch_dtype=torch.float16).to(device_0).eval()
    MODELS["siglip_processor"] = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")


def _load_clip():
    from transformers import CLIPModel, CLIPProcessor
    MODELS["clip_model"] = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device_0).eval()
    MODELS["clip_processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def _load_bge():
    from sentence_transformers import SentenceTransformer
    MODELS["bge_model"] = SentenceTransformer("BAAI/bge-large-en-v1.5").to(device_0)


def _load_easyocr():
    import easyocr
    MODELS["ocr_reader"] = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)


def _gpu_dtype(device):
    """
    Pick bfloat16 only where the hardware actually implements it.

    Kaggle's T4s are Turing (sm_75) and have no bf16 tensor cores. Torch will
    happily accept torch.bfloat16 there and emulate it, but bitsandbytes picks
    its 4-bit kernel path off the compute dtype and the bf16 path assumes
    sm_80+. On Turing that pairing is a documented source of illegal memory
    accesses under load. float16 is native on every GPU this stack runs on and
    is already what SigLIP and CLIP use here, so nothing else has to change.
    """
    if not torch.cuda.is_available():
        return torch.float32
    try:
        index = torch.device(device).index or 0
        major, _minor = torch.cuda.get_device_capability(index)
    except Exception:
        return torch.float16          # unknown GPU → take the universally safe path
    return torch.bfloat16 if major >= 8 else torch.float16


def _load_qwen():
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              BitsAndBytesConfig)
    qwen_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype = _gpu_dtype(device_1)
    log(f"Oracle compute dtype on {device_1}: {str(dtype).split('.')[-1]}")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                               bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    MODELS["oracle_processor"] = AutoProcessor.from_pretrained(qwen_id)
    MODELS["oracle_model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_id, torch_dtype=dtype, quantization_config=quant,
        device_map={"": device_1}).eval()


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════
def extract_img_features(model, processor, device, pil_imgs):
    inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.no_grad():
        # `projected`: on transformers 5.x these two methods return the whole
        # model output and the embedding is `.pooler_output`. This is the side
        # that *writes* the index, so getting it wrong does not raise — it fills
        # `frame_vector` with vectors no query can ever match. See atlas/hfcompat.
        features = projected(model.get_image_features(**inputs),
                             "image features")
        if features.ndim == 3:
            features = features.mean(dim=1)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features


def extract_txt_features(model, processor, device, text):
    inputs = processor(text=text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        features = projected(model.get_text_features(**inputs),
                             "text features")
        if features.ndim == 3:
            features = features.mean(dim=1)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features


def bge_encode(text):
    return np.array(MODELS["bge_model"].encode(text, normalize_embeddings=True)).tolist()


def siglip_text_vec(text):
    return extract_txt_features(MODELS["siglip_model"], MODELS["siglip_processor"],
                                device_0, text).cpu().numpy()[0].tolist()


def clip_text_vec(text):
    return extract_txt_features(MODELS["clip_model"], MODELS["clip_processor"],
                                device_0, text).cpu().numpy()[0].tolist()


# ═══════════════════════════════════════════════════════════
# QWEN ORACLE — serialized video-to-text generation
# ═══════════════════════════════════════════════════════════
def _trim_to_sentence(text, min_keep=0.4):
    """Drop a trailing fragment left behind by a max_new_tokens cutoff.

    Generation stops on a token count, not on grammar, so a capped narrative
    ends mid-word or mid-quote — the UI showed things like `such as: 1. "YOU `.
    If the text already ends on terminal punctuation it is returned untouched.
    Otherwise we cut back to the last real sentence end, but only when that
    keeps `min_keep` of the text: for a single long unpunctuated sentence,
    trimming would throw away everything the model actually saw, so it is kept
    with an ellipsis marking the truncation honestly.
    """
    text = (text or "").strip()
    if not text:
        return text

    def _terminal(s):
        """True when s ends a real sentence — a period right after a digit is
        a list marker ("1.") or a decimal, not an ending."""
        if s[-1:] not in ".!?…":
            return False
        return not (s[-1] == "." and len(s) > 1 and s[-2].isdigit())

    # Unbalanced quote from a cut-off on-screen-text citation.
    if text.count('"') % 2:
        text = text[:text.rfind('"')].rstrip()
        if not text:
            return text
    if _terminal(text):
        return text

    # Candidate sentence ends, latest first.
    best = -1
    for m in re.finditer(r'[.!?](?=[\s"]|$)', text):
        i = m.start()
        if text[i] == "." and i and text[i - 1].isdigit():
            continue
        best = i
    if best > len(text) * min_keep:
        return text[:best + 1].strip()
    return text.rstrip(" ,;:-—([{\"'0123456789.") + "…"


def qwen_describe_video(video_path, prompt_text, fps=2.0, max_new_tokens=150,
                        max_pixels=360000, temperature=0.7):
    """Run Qwen2.5-VL over a video file. Thread-safe (QWEN_LOCK)."""
    from qwen_vl_utils import process_vision_info
    processor = MODELS["oracle_processor"]
    model = MODELS["oracle_model"]
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "max_pixels": max_pixels, "fps": fps},
        {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    # A chunk that decodes to zero frames makes the model answer from the text
    # prompt alone, which degenerates into the same generic paragraph for every
    # video. Fail loudly instead — the job retries with a fresh chunk render.
    _v0 = video_inputs[0] if video_inputs else None
    _nframes = (_v0.shape[0] if hasattr(_v0, "shape") else len(_v0)) if _v0 is not None else 0
    if _nframes < 1:
        raise ValueError(f"no video frames decoded from {video_path}")
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device_1)
    for k, v in inputs.items():
        inputs[k] = v.to(model.dtype) if torch.is_floating_point(v) else v
    with QWEN_LOCK:
        with torch.no_grad():
            # Explicit sampling: bare generate() fell back to greedy decoding,
            # which collapses visually similar clips onto identical narratives.
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=0.9,
                repetition_penalty=1.05)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return _trim_to_sentence(processor.batch_decode(trimmed, skip_special_tokens=True)[0])


# ═══════════════════════════════════════════════════════════
# HYBRID SPATIAL PROOF — OCR match → DINO box → SAM mask overlay
# ═══════════════════════════════════════════════════════════
def hybrid_spatial_proof(image, text_query, ocr_results):
    """Returns (annotated ndarray, proven bool, message)."""
    img_arr = np.array(image)
    query_lower = text_query.lower()
    best_box, box_type = None, None

    for (bbox, text, conf) in ocr_results:
        if query_lower in text.lower() or any(
                word in text.lower() for word in query_lower.split() if len(word) > 3):
            x_coords, y_coords = [pt[0] for pt in bbox], [pt[1] for pt in bbox]
            best_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            box_type = "OCR"
            break

    if not best_box and "dino_model" in MODELS:
        inputs = MODELS["dino_processor"](images=image, text=query_lower + ".",
                                          return_tensors="pt").to(device_0)
        with torch.no_grad():
            outputs = MODELS["dino_model"](**inputs)
        dino_results = MODELS["dino_processor"].post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.15, text_threshold=0.15,
            target_sizes=[image.size[::-1]])[0]
        if len(dino_results["boxes"]) > 0:
            best_box = dino_results["boxes"][0].tolist()
            box_type = "DINO"

    if not best_box:
        return img_arr.astype(np.uint8), False, "❌ No Text or Physical Boundaries detected."

    if "sam_model" not in MODELS:
        color = np.array([255, 50, 50]) if box_type == "OCR" else np.array([50, 255, 50])
        cv2.rectangle(img_arr, (int(best_box[0]), int(best_box[1])),
                      (int(best_box[2]), int(best_box[3])), color.tolist()[::-1], 3)
        return img_arr.astype(np.uint8), True, f"📦 {box_type} box drawn (SAM offline)."

    sam_inputs = MODELS["sam_processor"](images=image, input_boxes=[[best_box]],
                                         return_tensors="pt").to(device_0)
    with torch.no_grad():
        sam_outputs = MODELS["sam_model"](**sam_inputs)
    mask = MODELS["sam_processor"].image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu())[0][0][0].numpy()

    color = np.array([255, 50, 50]) if box_type == "OCR" else np.array([50, 255, 50])
    msg = ("🔤 EasyOCR Matched Text: Drew BLUE spatial mask." if box_type == "OCR"
           else "🧩 Grounding DINO Matched Object: Drew GREEN spatial mask.")
    img_arr[mask] = img_arr[mask] * 0.6 + color * 0.4
    cv2.rectangle(img_arr, (int(best_box[0]), int(best_box[1])),
                  (int(best_box[2]), int(best_box[3])), color.tolist()[::-1], 3)
    return img_arr.astype(np.uint8), True, msg
