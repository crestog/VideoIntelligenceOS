import os
import gc
import json
import time
import torch
import psutil
import redis
import threading
import traceback

# ==========================================
# 1. INITIALIZE MESSAGE BROKER
# ==========================================
print("🗄️ Connecting Message Broker...")
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
redis_client.set("SYSTEM_PAUSED", "FALSE")

os.environ["HF_HOME"] = "/kaggle/working/huggingface_cache"
device_0 = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global dictionary to hold the heavy weights in VRAM safely
WARM_MODELS = {}

def clear_ram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def log_memory(step_name):
    ram = psutil.virtual_memory()
    print(f"\n📊 [{step_name}] CPU RAM: {ram.used / 1e9:.2f}GB / {ram.total / 1e9:.2f}GB")
    if torch.cuda.is_available():
        print(f"   🖥️ GPU 0 VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
    print("-" * 40)

# ==========================================
# 2. ALL 7 VISION & AUDIO MODELS
# ==========================================
def download_siglip():
    from transformers import AutoProcessor, AutoModel
    print("👁️ Downloading SigLIP2-So400m...")
    model_id = "google/siglip-so400m-patch14-384"
    WARM_MODELS['siglip_processor'] = AutoProcessor.from_pretrained(model_id)
    WARM_MODELS['siglip_model'] = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device_0).eval()

def download_clip():
    from transformers import CLIPProcessor, CLIPModel
    print("👁️ Downloading CLIP ViT-L/14...")
    model_id = "openai/clip-vit-large-patch14"
    WARM_MODELS['clip_processor'] = CLIPProcessor.from_pretrained(model_id)
    WARM_MODELS['clip_model'] = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device_0).eval()

def download_dinov2():
    from transformers import AutoImageProcessor, AutoModel
    print("🧩 Downloading DINOv2-Large...")
    model_id = "facebook/dinov2-large"
    WARM_MODELS['dino_processor'] = AutoImageProcessor.from_pretrained(model_id)
    WARM_MODELS['dino_model'] = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device_0).eval()

def download_whisper():
    from faster_whisper import WhisperModel
    print("🎙️ Downloading Whisper-Large-v3...")
    WARM_MODELS['whisper_model'] = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")

def download_raft():
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    print("🌊 Downloading RAFT-Large...")
    weights = Raft_Large_Weights.DEFAULT
    WARM_MODELS['raft_transforms'] = weights.transforms()
    WARM_MODELS['raft_model'] = raft_large(weights=weights, progress=False).to(device_0).eval()

def download_yolo():
    from ultralytics import YOLO
    print("🎯 Downloading YOLOv11x...")
    WARM_MODELS['yolo_model'] = YOLO("yolo11x.pt")
    WARM_MODELS['yolo_model'].to(device_0)

def download_easyocr():
    import easyocr
    print("🔤 Downloading EasyOCR...")
    WARM_MODELS['ocr_reader'] = easyocr.Reader(['en'], gpu=True)

MODEL_REGISTRY = {
    "siglip": download_siglip,
    "clip": download_clip,
    "dinov2": download_dinov2,
    "whisper": download_whisper,
    "raft": download_raft,
    "yolo": download_yolo,
    "easyocr": download_easyocr
}

# ==========================================
# 3. MICROSERVICE WORKER (TASK HANDLING)
# ==========================================
def model_worker_loop():
    print("⚙️ Worker (Model Loader) Online. Waiting for jobs...")
    log_memory("Baseline")
    
    while True:
        # Exact logic from your original script: Check if system is frozen
        if redis_client.get("SYSTEM_PAUSED") == "TRUE":
            time.sleep(2)
            continue
            
        # blpop blocks until a job is available in QUEUE_MODELS
        job_raw = redis_client.blpop("QUEUE_MODELS", timeout=2)
        if not job_raw: continue
        
        job = json.loads(job_raw[1])
        model_name = job.get("model_name")
        
        if model_name not in MODEL_REGISTRY:
            print(f"⚠️ Unknown model requested: {model_name}")
            continue

        redis_client.hset("status:models", model_name, "Loading ⏳")
        
        try:
            print(f"\n[Model Worker] Processing {model_name}...")
            MODEL_REGISTRY[model_name]()
            clear_ram()
            
            redis_client.hset("status:models", model_name, "DONE ✅")
            log_memory(f"Post-{model_name.upper()} Load")
            
        except Exception as e:
            print(f"❌ Model Worker Failed: {traceback.format_exc()}")
            redis_client.hset("status:models", model_name, f"ERROR: {e}")
            clear_ram()

# ==========================================
# 4. STARTING THE WORKER THREADS
# ==========================================
def start_workers():
    # Matches your exact threading setup
    threading.Thread(target=model_worker_loop, daemon=True).start()

if __name__ == "__main__":
    start_workers()
    # Keep main thread alive just like your original code
    while True: time.sleep(1)
