import os, gc, torch, psutil
from queue_manager import pop_job, push_job

os.environ["HF_HOME"] = "/kaggle/working/huggingface_cache"
device_0 = "cuda:0" if torch.cuda.is_available() else "cpu"
WARM_MODELS = {}

def clear_ram():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def load_siglip():
    from transformers import AutoProcessor, AutoModel
    print("👁️ Downloading SigLIP2...")
    WARM_MODELS['siglip_model'] = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.float16).to(device_0).eval()

def load_clip():
    from transformers import CLIPModel
    print("👁️ Downloading CLIP...")
    WARM_MODELS['clip_model'] = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device_0).eval()

def load_whisper():
    from faster_whisper import WhisperModel
    print("🎙️ Downloading Whisper...")
    WARM_MODELS['whisper_model'] = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")

MODEL_REGISTRY = {"siglip": load_siglip, "clip": load_clip, "whisper": load_whisper}

if __name__ == "__main__":
    print("🚀 Model Manager Booting: Auto-queueing models...")
    for model in MODEL_REGISTRY.keys(): push_job("QUEUE_MODELS", {"model_name": model})
    
    while True:
        job = pop_job("QUEUE_MODELS", timeout=2)
        if not job: continue
        m_name = job.get("model_name")
        if m_name in MODEL_REGISTRY:
            try:
                MODEL_REGISTRY[m_name]()
                clear_ram()
                print(f"✅ {m_name.upper()} loaded into VRAM.")
            except Exception as e:
                print(f"❌ Failed {m_name}: {e}")
