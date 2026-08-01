"""
VIOS Configuration — Single Source of Truth

All paths, constants, queue names, and thresholds.
Every module imports from here instead of defining its own paths.

═══════════════════════════════════════════════════════════════════════════
STORAGE TIERING — why this matters, and why it broke
═══════════════════════════════════════════════════════════════════════════
Kaggle gives you two very different pools of space:

  /kaggle/working  → the OUTPUT quota (~20 GB). Survives the session, is what
                     you can download, and is *small*.
  /kaggle/temp     → scratch on the container disk (the rest of the ~57.6 GB).
                     Vanishes with the session. Large.

The Omniscient merge put HF_HOME on /kaggle/working. Each model stack fits
there alone — model_manager.py pulls ~9.5 GB, omni_models.py pulls ~19 GB
(Qwen2.5-VL-7B is 16.6 GB by itself) — but together they need ~28 GB in a
20 GB quota. So the merged system died at "No space left on device" on the
first Qwen shard, and every downstream failure (bge, siglip, "database or
disk is full", the ASGI traceback) was that same wall.

The rule now: anything re-downloadable or re-derivable lives on SCRATCH_DIR.
Only irreplaceable state (lake.db, the Telegram session) stays on the output
quota, since that is the one thing a finished Kaggle session leaves behind.

  SCRATCH (big, ephemeral)   model weights, HF/torch caches, extracted frames,
                             thumbnails, Qdrant vectors, Neo4j store, Postgres
  OUTPUT  (small, kept)      lake.db, bot_session, state.txt, flagged dataset

Override any of it with VIOS_BASE_DIR / VIOS_SCRATCH_DIR.
"""

import os
import shutil

# ═══════════════════════════════════════════════════════════
# FILESYSTEM PATHS
# ═══════════════════════════════════════════════════════════
_KAGGLE_DEFAULT = '/kaggle/working/Insta-Vault'

if os.environ.get('VIOS_BASE_DIR'):
    BASE_DIR = os.environ['VIOS_BASE_DIR']
elif os.path.isdir('/kaggle/working'):
    BASE_DIR = _KAGGLE_DEFAULT
else:
    # Local development fallback — keeps the same layout relative to the repo
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Insta-Vault')

LAKE_DIR    = os.path.join(BASE_DIR, 'DataLake')
DB_PATH     = os.path.join(LAKE_DIR, 'lake.db')
FLAG_DIR    = os.path.join(LAKE_DIR, '_Flagged_Dataset')
SESSION_DIR = os.path.join(LAKE_DIR, 'bot_session')
STATE_FILE  = os.path.join(LAKE_DIR, 'state.txt')


# ═══════════════════════════════════════════════════════════
# SCRATCH DISK — the big ephemeral pool
# ═══════════════════════════════════════════════════════════
def _free_gb(path):
    """Free space in GB for the mount holding `path` (0.0 if unmeasurable)."""
    try:
        probe = path
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        return shutil.disk_usage(probe).free / (1024 ** 3)
    except OSError:
        return 0.0


def _usable(path):
    """True if `path` can be created and written to."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, '.vios_write_probe')
        with open(probe, 'w') as f:
            f.write('ok')
        os.remove(probe)
        return True
    except OSError:
        return False


def _pick_scratch():
    """
    Choose the roomiest writable scratch location.

    Preference order is deliberate — /kaggle/temp is Kaggle's documented
    scratch mount and does not count against the output quota — but the final
    pick is by measured free space, so this stays correct even if Kaggle
    changes its disk geometry or we run somewhere else entirely.
    """
    if os.environ.get('VIOS_SCRATCH_DIR'):
        return os.environ['VIOS_SCRATCH_DIR']

    candidates = ['/kaggle/temp/vios_scratch', '/tmp/vios_scratch']
    # Last resort: sit next to BASE_DIR (same quota, but always writable).
    candidates.append(os.path.join(BASE_DIR, '_scratch'))

    best, best_free = None, -1.0
    for cand in candidates:
        if not _usable(cand):
            continue
        free = _free_gb(cand)
        if free > best_free:
            best, best_free = cand, free
    return best or os.path.join(BASE_DIR, '_scratch')


SCRATCH_DIR = _pick_scratch()

# Heavy, regenerable data — all on scratch.
MODEL_CACHE_DIR = os.path.join(SCRATCH_DIR, 'model_cache')
VIDEO_DIR       = os.path.join(SCRATCH_DIR, 'videos')       # videos + extracted frames
THUMB_DIR       = os.path.join(SCRATCH_DIR, 'thumbnails')
NEO4J_DATA_DIR  = os.path.join(SCRATCH_DIR, 'neo4j')


# ═══════════════════════════════════════════════════════════
# CACHE + LOG ENVIRONMENT
# Must be set BEFORE torch/transformers import, so every entrypoint imports
# config first. Uses setdefault throughout — an explicit env var always wins.
# ═══════════════════════════════════════════════════════════
def configure_environment():
    """Point every model cache at scratch and mute duplicate-registration noise."""
    hf_home = os.path.join(MODEL_CACHE_DIR, 'huggingface')
    os.environ.setdefault('HF_HOME', hf_home)
    os.environ.setdefault('HF_HUB_CACHE', os.path.join(hf_home, 'hub'))
    os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME',
                          os.path.join(MODEL_CACHE_DIR, 'sentence_transformers'))
    os.environ.setdefault('TORCH_HOME', os.path.join(MODEL_CACHE_DIR, 'torch'))
    os.environ.setdefault('EASYOCR_MODULE_PATH', os.path.join(MODEL_CACHE_DIR, 'easyocr'))
    os.environ.setdefault('YOLO_CONFIG_DIR', os.path.join(MODEL_CACHE_DIR, 'ultralytics'))
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(MODEL_CACHE_DIR, 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', os.path.join(MODEL_CACHE_DIR, 'xdg'))

    # transformers probes for TensorFlow at import and loads it if installed.
    # TF then re-registers the CUDA factories torch already registered, which is
    # the entire source of the "Unable to register cuDNN/cuBLAS/cuFFT factory"
    # and "computation placer already registered" spam. Nothing in VIOS uses TF,
    # so refusing the import removes the cause instead of hiding the symptom.
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_FLAX', '0')
    os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
    os.environ.setdefault('GLOG_minloglevel', '2')

    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    # Create the cache dirs now. Several libraries write a settings file on
    # first import and silently fall back to /tmp if the directory is not
    # already there — Ultralytics does exactly that ("user config directory
    # ... is not writable, using /tmp/Ultralytics"), which would scatter caches
    # back off the scratch tier one library at a time.
    for _var in ('HF_HOME', 'HF_HUB_CACHE', 'SENTENCE_TRANSFORMERS_HOME',
                 'TORCH_HOME', 'EASYOCR_MODULE_PATH', 'YOLO_CONFIG_DIR',
                 'MPLCONFIGDIR', 'XDG_CACHE_HOME'):
        try:
            os.makedirs(os.environ[_var], exist_ok=True)
        except OSError:
            pass  # unwritable tier is reported by the boot storage budget


configure_environment()


def disk_report():
    """
    Per-tier free space, for the boot banner.
    Returns [(label, path, free_gb), ...].
    """
    return [
        ('OUTPUT  (kept, quota)', BASE_DIR,       _free_gb(BASE_DIR)),
        ('SCRATCH (ephemeral)',   SCRATCH_DIR,    _free_gb(SCRATCH_DIR)),
        ('MODELS  (on scratch)',  MODEL_CACHE_DIR, _free_gb(MODEL_CACHE_DIR)),
    ]

# ═══════════════════════════════════════════════════════════
# QUEUE NAMES
# ═══════════════════════════════════════════════════════════
QUEUE_VISION  = 'QUEUE_VISION'
QUEUE_MODELS  = 'QUEUE_MODELS'
QUEUE_ANALYZE = 'QUEUE_ANALYZE'   # GPU analysis jobs (after frame extraction)

# ═══════════════════════════════════════════════════════════
# FRAME EXTRACTION TIERS
# ═══════════════════════════════════════════════════════════
PREVIEW_DIR_NAME = '.preview'     # per-video subfolder for the low-res tier
PREVIEW_WIDTH    = 320            # preview tier width (height auto, keeps aspect)
PREVIEW_QUALITY  = 6              # ffmpeg -q:v for preview jpegs (2 best..31 worst)
FULL_QUALITY     = 2              # ffmpeg -q:v for full-res jpegs

# ═══════════════════════════════════════════════════════════
# WORKER CONFIGURATION
# ═══════════════════════════════════════════════════════════
MAX_RETRIES              = 3      # Retry attempts before dead-lettering
DISK_PAUSE_THRESHOLD_GB  = 2.0    # CV engine pauses extraction below this (on scratch)
DISK_WARN_THRESHOLD_GB   = 5.0    # CV engine warns below this
DISK_DL_PAUSE_GB         = 3.0    # Ghost Worker pauses Telegram downloads below this
BATCH_FRAME_COUNT        = 200    # Frames per binary batch fetch in V17

# ═══════════════════════════════════════════════════════════
# SQLITE — shared connection settings (prevents "database is locked")
# ═══════════════════════════════════════════════════════════
SQLITE_TIMEOUT = 30  # seconds; ALL sqlite3.connect calls must pass this

# ═══════════════════════════════════════════════════════════
# TELEGRAM (shared by Ghost Worker + Snapshot Manager)
# ═══════════════════════════════════════════════════════════
API_ID     = int(os.environ.get('VIOS_API_ID', 37392880))
API_HASH   = os.environ.get('VIOS_API_HASH', '4037344084ae998be2cdaee3192bd8f8')
BOT_TOKEN  = os.environ.get('VIOS_BOT_TOKEN', '8269867642:AAH76B2_aFbqc6OqNiCAm-NenTTmG_SWavU')
CHANNEL_ID = int(os.environ.get('VIOS_CHANNEL_ID', -1003762735924))

# ═══════════════════════════════════════════════════════════
# REDIS
# ═══════════════════════════════════════════════════════════
REDIS_HOST = os.environ.get('VIOS_REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('VIOS_REDIS_PORT', 6379))

# ═══════════════════════════════════════════════════════════
# OMNISCIENT LAYER (tri-partite DB + GraphRAG engine)
#   Set VIOS_OMNI=0 to disable the whole subsystem.
# ═══════════════════════════════════════════════════════════
OMNI_ENABLED = os.environ.get('VIOS_OMNI', '1') != '0'

# Filesystem — both on scratch: bot downloads are re-fetchable from Telegram and
# the vector store is rebuildable from the frames, so neither belongs in the
# 20 GB output quota.
ARCHIVE_DIR  = os.path.join(SCRATCH_DIR, 'omni_archive')     # bot downloads, frames, chunks
QDRANT_PATH  = os.path.join(SCRATCH_DIR, 'qdrant_storage')   # embedded Qdrant store

# Queues (dual-lane: bot uploads = PRIORITY, Ghost Worker harvest = DEFAULT)
QUEUE_OMNI_VISION = 'QUEUE_OMNI_VISION'
QUEUE_OMNI_ORACLE = 'QUEUE_OMNI_ORACLE'
OMNI_DEDUP_SET    = 'OMNI_PROCESSED_SET'

# Hugging Face (for gated model downloads).
#   Supplied by the launcher from Kaggle Secrets -> os.environ["HF_TOKEN"].
#   Empty is fine: every model in the stack is currently public.
HF_TOKEN = os.environ.get('HF_TOKEN', '')

# NVIDIA NIM API (GraphRAG extraction + query rewrite + synthesis).
#   Supplied by the launcher from Kaggle Secrets -> os.environ["VIOS_NIM_API_KEY"].
#   Empty disables GraphRAG/synthesis only; vision + oracle pipelines still run.
NIM_API_KEY  = os.environ.get('VIOS_NIM_API_KEY', '')
NIM_BASE_URL = os.environ.get('VIOS_NIM_BASE_URL', 'https://integrate.api.nvidia.com/v1')
NIM_MODEL    = os.environ.get('VIOS_NIM_MODEL', 'nvidia/nemotron-3-ultra-550b-a55b')

# PostgreSQL (Omniscient relational store)
OMNI_PG_DB       = os.environ.get('VIOS_PG_DB', 'omnidb')
OMNI_PG_USER     = os.environ.get('VIOS_PG_USER', 'omni')
OMNI_PG_PASSWORD = os.environ.get('VIOS_PG_PASSWORD', 'omni')
OMNI_PG_HOST     = os.environ.get('VIOS_PG_HOST', 'localhost')

# Neo4j (knowledge graph). NEO4J_HOME = extracted community tarball.
NEO4J_HOME = os.environ.get('VIOS_NEO4J_HOME',
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'neo4j-community-5.18.0'))
NEO4J_BOLT = os.environ.get('VIOS_NEO4J_BOLT', 'bolt://127.0.0.1:7687')
JAVA_HOME  = os.environ.get('VIOS_JAVA_HOME', '/usr/lib/jvm/java-17-openjdk-amd64')

# God-Mode Explorer dashboard (Flask inside omni_engine, proxied at /omni)
OMNI_DASHBOARD_PORT = int(os.environ.get('VIOS_OMNI_DASHBOARD_PORT', 5000))

# Processing modes: (chunk_seconds, qwen_fps, qwen_max_tokens, frame_step_fps)
OMNI_MODE_OMNI  = {'chunk': 5.0,  'fps': 2.0, 'tokens': 150}
OMNI_MODE_BLITZ = {'chunk': 15.0, 'fps': 1.0, 'tokens': 75}
OMNI_BLITZ_SAMPLE_FPS = 0.4       # blitz-mode frame sampling rate

# ═══════════════════════════════════════════════════════════
# ENSURE DIRECTORIES EXIST
# ═══════════════════════════════════════════════════════════
for d in [LAKE_DIR, VIDEO_DIR, THUMB_DIR, FLAG_DIR, ARCHIVE_DIR]:
    os.makedirs(d, exist_ok=True)
