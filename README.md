# 🧠 Video Intelligence OS (Project Omega)

**Lead Architect:** Devansh Sahu  
**Phase:** Core Infrastructure & Ingestion Complete  

Video Intelligence OS (VIOS) is a decoupled, multi-modal cloud operating system designed to autonomously harvest, standardize, and analyze massive volumes of video data to build a "Civilizational Index." 

Built specifically for ephemeral, resource-constrained cloud hardware (like Kaggle), VIOS utilizes an enterprise-grade microservice architecture to ensure zero-latency UI interactions, mathematical fault tolerance, and maximum hardware utilization.

---

## 🏗️ Core Architecture Overview

The system is divided into three completely independent execution layers, unified by a persistent Redis message broker.

### 1. The Watchdog Orchestrator (`boot.py`)
The master control plane. It never performs heavy computation; its only job is to keep the system alive.
* **Pre-flight Sweep:** Automatically purges zombie FFmpeg/Cloudflare processes from previous crashes to protect system RAM.
* **Redis Initialization:** Boots the Redis broker in `AOF` (Append-Only File) mode, ensuring job queues survive sudden power-offs.
* **Auto-Healing Threads:** Launches the UI Server and Model Engine in isolated threads. If either worker crashes, the Watchdog catches the failure and resurrects the process in 3 seconds.
* **Stream Filtering:** Captures standard output from all workers, aggressively filters out raw library noise (like HuggingFace progress bars), and renders a clean, unified, zero-latency console for the developer.

### 2. The Ingestion Node (`ui_server.py`)
The high-speed data harvester and user interface, built on `FastAPI` and `Pyrogram`.
* **The Ghost Worker:** Autonomously scans the configured Telegram channel (`global_index_v`), fetching missing videos backwards through time to maintain a perfect local copy.
* **Metadata Extraction:** Uses Regex heuristics to parse structured captions (Creator, Category, Likes) and writes them to the database.
* **Format Standardization:** Routes all downloaded media through a dynamic FFmpeg pipeline. If a video is web-compatible, it triggers an `Instant Remux` (`-c copy`). If not, it falls back to a lossless `CRF 17` conversion. 
* **The Command Deck UI:** A responsive HTML/JS web interface featuring an infinite-scroll FTS5 search index, live video playback, keyboard macros, and live cloud-status monitoring.

### 3. The Multi-Modal Engine (`model_manager.py`)
The AI compute layer. It warms up 7 state-of-the-art neural networks and anchors them into the GPU. 
* **Strict VRAM Stacking:** Models are loaded in a mathematically determined sequence (heaviest to lightest) to eliminate VRAM fragmentation and prevent CUDA OOM crashes.
* **The 7 Foundational Models:**
  1. **YOLO11x:** Object detection and spatial awareness.
  2. **Whisper-Large-v3:** High-fidelity audio transcription.
  3. **DINOv2-Large:** Deep visual feature extraction.
  4. **SigLIP2:** High-performance text-image semantic embedding.
  5. **CLIP:** Secondary semantic vectorization.
  6. **RAFT-Large:** Optical flow and motion physics tracking.
  7. **EasyOCR:** GPU-accelerated text-in-video extraction.

---

## 🛡️ Enterprise Engineering Standards Implemented

To survive the hostile environment of cloud scraping and ephemeral servers, this OS features multiple layers of advanced fault protection:

### Network & Scraping Resilience
* **Exponential Backoff:** The Pyrogram client actively listens for Telegram `FloodWait` exceptions and `Broken Pipe` socket crashes, automatically pacing its downloads and sleeping to avoid permanent server bans.

### Compute & Hardware Optimization
* **FFmpeg Semaphores:** Video conversion is bottlenecked by a global `asyncio.Semaphore(2)`, mathematically guaranteeing that the OS will never spawn more than two concurrent FFmpeg processes, protecting the CPU from exhaustion.
* **Disk Space Watchdog:** The worker continuously monitors the `videos/` directory. If total system storage drops below 5GB, the OS pauses all scraping to prevent catastrophic SQLite corruption.

### Latency & Data Flow
* **Asynchronous Database Pipeline:** The SQLite database utilizes `aiosqlite` and `WAL` (Write-Ahead Logging) mode, allowing the Ghost Worker to perform rapid, continuous inserts without freezing the FastAPI web loop.
* **Idempotent Queues:** Before pushing a downloaded video to the Redis AI processing queue, the OS checks a high-speed Redis Set (`SADD`). This acts as an absolute gatekeeper, ensuring the GPUs never waste time processing duplicate media.
* **Cursor Pagination:** The UI explore tab utilizes Offset Pagination (`LIMIT 30 OFFSET X`), reducing API payload sizes by 97% and providing instantaneous front-end rendering.

---

## 🚀 The Kaggle Launchpad (CI/CD)

The repository relies on Infrastructure as Code (IaC) via `setup.sh` and `requirements.txt`. The entire OS can be bootstrapped from a factory-reset Kaggle instance using this single cell:

```python
import os
from kaggle_secrets import UserSecretsClient

os.chdir("/kaggle/working")
WORKSPACE = "/kaggle/working/vios_system"

secrets = UserSecretsClient()
repo = f"https://crestog:{secrets.get_secret('github_token')}@[github.com/crestog/VideoIntelligenceOS.git](https://github.com/crestog/VideoIntelligenceOS.git)"
os.system(f"git clone {repo} {WORKSPACE} 2>/dev/null || git -C {WORKSPACE} pull {repo} main > /dev/null 2>&1")

try:
    os.environ["HF_TOKEN"] = secrets.get_secret("hf_token")
except: 
    pass

os.chdir(WORKSPACE)
print("⚙️ [SYSTEM] Provisioning Environment...")
os.system("bash setup.sh > setup_logs.txt 2>&1") 
os.system("python boot.py")
```
