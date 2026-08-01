"""
VIOS Configuration — Single Source of Truth

All paths, constants, queue names, and thresholds.
Every module imports from here instead of defining its own paths.

BASE_DIR resolution order:
  1. VIOS_BASE_DIR environment variable (explicit override)
  2. /kaggle/working/Insta-Vault  (default Kaggle deployment path)
  3. ./Insta-Vault relative to this file (local development fallback)
"""

import os

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
VIDEO_DIR   = os.path.join(LAKE_DIR, 'videos')
THUMB_DIR   = os.path.join(LAKE_DIR, '.thumbnails')
FLAG_DIR    = os.path.join(LAKE_DIR, '_Flagged_Dataset')
SESSION_DIR = os.path.join(LAKE_DIR, 'bot_session')
STATE_FILE  = os.path.join(LAKE_DIR, 'state.txt')

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
# SNAPSHOT (Telegram DB export/import)
# ═══════════════════════════════════════════════════════════
SNAPSHOT_DIR        = (os.path.join('/tmp', 'vios_snapshots') if os.path.isdir('/tmp')
                       else os.path.join(LAKE_DIR, '.snapshots'))
SNAPSHOT_CHUNK_MB   = 1900        # stay under the 2 GB MTProto bot limit per part
SNAPSHOT_TAG        = '#VIOS_SNAPSHOT'   # caption tag used to find snapshots in the channel

# ═══════════════════════════════════════════════════════════
# WORKER CONFIGURATION
# ═══════════════════════════════════════════════════════════
MAX_RETRIES              = 3      # Retry attempts before dead-lettering
DISK_PAUSE_THRESHOLD_GB  = 0.5    # CV engine pauses below this
DISK_WARN_THRESHOLD_GB   = 1.5    # CV engine warns below this
DISK_DL_PAUSE_GB         = 1.0    # Ghost Worker pauses downloads below this
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

# Filesystem
ARCHIVE_DIR  = os.path.join(LAKE_DIR, 'omni_archive')     # bot downloads, frames, chunks
QDRANT_PATH  = os.path.join(LAKE_DIR, 'qdrant_storage')   # embedded Qdrant store

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
