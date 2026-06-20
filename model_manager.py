import os, gc, torch
# Professional Logging Filter: Mute raw library noise
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from queue_manager import pop_job, push_job

os.environ["HF_HOME"] = "/kaggle/working/huggingface_cache"
device_0 = "cuda:0" if torch.cuda.is_available() else "cpu"
WARM_MODELS = {}

def clear_ram():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# --- THE 7 SOTA VISION/AUDIO ENGINES ---
def load_siglip():
    from transformers import AutoModel
    print("👁️ Downloading SigLIP2...")
    WARM_MODELS['siglip_model'] = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.float16).to(device_0).eval()

def load_clip():
    from transformers import CLIPModel
    print("👁️ Downloading CLIP...")
    WARM_MODELS['clip_model'] = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device_0).eval()

def load_dinov2():
    from transformers import AutoModel
    print("🧩 Downloading DINOv2-Large...")
    WARM_MODELS['dino_model'] = AutoModel.from_pretrained("facebook/dinov2-large", torch_dtype=torch.float16).to(device_0).eval()

def load_whisper():
    from faster_whisper import WhisperModel
    print("🎙️ Downloading Whisper-Large-v3...")
    WARM_MODELS['whisper_model'] = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")

def load_raft():
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    print("🌊 Downloading RAFT-Large...")
    weights = Raft_Large_Weights.DEFAULT
    WARM_MODELS['raft_transforms'] = weights.transforms()
    WARM_MODELS['raft_model'] = raft_large(weights=weights, progress=False).to(device_0).eval()

def load_yolo():
    from ultralytics import YOLO
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR) # Mute YOLO noise
    print("🎯 Downloading YOLOv11x...")
    WARM_MODELS['yolo_model'] = YOLO("yolo11x.pt")
    WARM_MODELS['yolo_model'].to(device_0)

def load_easyocr():
    import easyocr
    import logging
    logging.getLogger("easyocr").setLevel(logging.ERROR)
    print("🔤 Downloading EasyOCR...")
    WARM_MODELS['ocr_reader'] = easyocr.Reader(['en'], gpu=True, verbose=False)

# --- THE REGISTRY ---
MODEL_REGISTRY = {
    "siglip": load_siglip,
    "clip": load_clip,
    "dinov2": load_dinov2,
    "whisper": load_whisper,
    "raft": load_raft,
    "yolo": load_yolo,
    "easyocr": load_easyocr
}

if __name__ == "__main__":
    print("🚀 Model Manager Booting: Auto-queueing all 7 foundational models...")
    for model in ["yolo", "whisper", "dinov2", "siglip", "clip", "raft", "easyocr"]: 
        push_job("QUEUE_MODELS", {"model_name": model})
    
    while True:
        job = pop_job("QUEUE_MODELS", timeout=2)
        if not job: continue
        
        m_name = job.get("model_name")
        if m_name in MODEL_REGISTRY:
            try:
                MODEL_REGISTRY[m_name]()
                clear_ram()
                print(f"✅ {m_name.upper()} loaded securely into VRAM.")
            except Exception as e:
                print(f"❌ Failed to load {m_name}: {e}")
                clear_ram()
