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
QUEUE_VISION       = 'QUEUE_VISION'
QUEUE_MODELS       = 'QUEUE_MODELS'
QUEUE_ANALYZE      = 'QUEUE_ANALYZE'       # GPU analysis jobs (after frame extraction)
QUEUE_ORACLE       = 'QUEUE_ORACLE'        # Qwen2.5-VL narrative generation
QUEUE_VISION_EMBED = 'QUEUE_VISION_EMBED'  # SigLIP/CLIP/Depth/RAFT embedding

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
DISK_PAUSE_THRESHOLD_GB  = 2.0    # CV engine pauses below this (one long video can eat 300+ MB of frames)
DISK_WARN_THRESHOLD_GB   = 4.0    # CV engine warns below this
DISK_DL_PAUSE_GB         = 3.0    # Ghost Worker pauses downloads below this
BATCH_FRAME_COUNT        = 200    # Frames per binary batch fetch in V17

# ── Downloader backpressure: don't outrun the GPU pipeline ──
# The Ghost Worker stops fetching new videos while the extraction queue or
# the combined GPU backlog (analyze + oracle + embed) is above these caps,
# so disk usage stays bounded by work-in-flight instead of channel size.
QUEUE_VISION_MAX_PENDING   = int(os.environ.get('VIOS_VISION_MAX_PENDING', 4))
DOWNSTREAM_MAX_BACKLOG     = int(os.environ.get('VIOS_DOWNSTREAM_MAX_BACKLOG', 30))

# ── Frame lifecycle: full-res frames are a processing artifact ──
# After a video is BOTH analyzed (YOLO/OCR) and embedded (SigLIP/CLIP/etc),
# the full-res JPEG tier is deleted; the ~8 KB preview tier + thumbnails stay.
# Spatial-proof re-extracts single full-res frames from the video on demand.
PURGE_FULL_FRAMES = os.environ.get('VIOS_PURGE_FULL_FRAMES', '1') != '0'
FRAME_INDEX_NAME  = 'frames_index.json'   # written at purge time so the UI keeps its frame list

# ═══════════════════════════════════════════════════════════
# SQLITE — shared connection settings (prevents "database is locked")
# ═══════════════════════════════════════════════════════════
SQLITE_TIMEOUT = 30  # seconds; ALL sqlite3.connect calls must pass this

# ═══════════════════════════════════════════════════════════
# TELEGRAM (shared by Ghost Worker + Snapshot Manager)
# SECURITY: no hardcoded defaults. Set these via environment variables
# (on Kaggle: UserSecretsClient → os.environ in the launcher cell).
# ═══════════════════════════════════════════════════════════
API_ID     = int(os.environ.get('VIOS_API_ID', 0) or 0)
API_HASH   = os.environ.get('VIOS_API_HASH', '')
BOT_TOKEN  = os.environ.get('VIOS_BOT_TOKEN', '')
CHANNEL_ID = int(os.environ.get('VIOS_CHANNEL_ID', 0) or 0)

# ═══════════════════════════════════════════════════════════
# REDIS
# ═══════════════════════════════════════════════════════════
REDIS_HOST = os.environ.get('VIOS_REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('VIOS_REDIS_PORT', 6379))

# ═══════════════════════════════════════════════════════════
# QDRANT (local on-disk vector store)
# ═══════════════════════════════════════════════════════════
QDRANT_PATH = os.environ.get('VIOS_QDRANT_PATH', os.path.join(BASE_DIR, 'qdrant_storage'))

# ═══════════════════════════════════════════════════════════
# NEO4J (community, launched as subprocess)
# ═══════════════════════════════════════════════════════════
NEO4J_HOME       = os.environ.get('VIOS_NEO4J_HOME', '/kaggle/working/neo4j-community-5.18.0')
NEO4J_JAVA_HOME  = os.environ.get('VIOS_NEO4J_JAVA_HOME', '/usr/lib/jvm/java-17-openjdk-amd64')
NEO4J_BOLT_URL   = os.environ.get('VIOS_NEO4J_BOLT_URL', 'bolt://127.0.0.1:7687')

# ═══════════════════════════════════════════════════════════
# NVIDIA NIM API (GraphRAG entity extraction)
# ═══════════════════════════════════════════════════════════
NVIDIA_API_KEY = os.environ.get('VIOS_NVIDIA_API_KEY', '')
NIM_MODEL      = os.environ.get('VIOS_NIM_MODEL', 'nvidia/nemotron-3-ultra-550b-a55b')
NIM_BASE_URL   = 'https://integrate.api.nvidia.com/v1'

# ═══════════════════════════════════════════════════════════
# ORACLE PROCESSING MODES
# blitz: 15s chunks, 1 fps, 75 tokens — fast overview
# omni:  5s chunks, 2 fps, 150 tokens — rich detail
# ═══════════════════════════════════════════════════════════
DEFAULT_ORACLE_MODE    = os.environ.get('VIOS_ORACLE_MODE', 'blitz')  # 'blitz' | 'omni'
GRAPHRAG_IN_OMNI_ONLY = os.environ.get('VIOS_GRAPHRAG_OMNI_ONLY', '1') != '0'  # default: True

# ═══════════════════════════════════════════════════════════
# ENSURE DIRECTORIES EXIST
# ═══════════════════════════════════════════════════════════
for d in [LAKE_DIR, VIDEO_DIR, THUMB_DIR, FLAG_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# CONFIG VALIDATION — fail loudly and clearly, never mysteriously
# ═══════════════════════════════════════════════════════════
# (name, value, required_for, is_critical)
_SECRET_SPECS = [
    ('VIOS_API_ID',         API_ID,         'Telegram sync (Ghost Worker, snapshots)', True),
    ('VIOS_API_HASH',       API_HASH,       'Telegram sync (Ghost Worker, snapshots)', True),
    ('VIOS_BOT_TOKEN',      BOT_TOKEN,      'Telegram sync (Ghost Worker, snapshots)', True),
    ('VIOS_CHANNEL_ID',     CHANNEL_ID,     'Telegram sync (Ghost Worker, snapshots)', True),
    ('VIOS_NVIDIA_API_KEY', NVIDIA_API_KEY, 'GraphRAG entity extraction (optional)',   False),
]


def validate_config(strict=False):
    """
    Check that required secrets are present. Returns a report dict:
      {"ok": bool, "missing_critical": [...], "missing_optional": [...], "set": [...]}

    strict=True raises RuntimeError listing every missing critical secret,
    so boot fails with ONE clear message instead of a cryptic mid-run error.
    """
    missing_critical, missing_optional, present = [], [], []
    for name, value, purpose, critical in _SECRET_SPECS:
        if not value:
            (missing_critical if critical else missing_optional).append(
                {'name': name, 'needed_for': purpose})
        else:
            present.append(name)

    report = {
        'ok': len(missing_critical) == 0,
        'missing_critical': missing_critical,
        'missing_optional': missing_optional,
        'set': present,
    }

    if strict and missing_critical:
        lines = ['[CONFIG] Missing required environment variables:']
        for m in missing_critical:
            lines.append(f"  - {m['name']}  (needed for: {m['needed_for']})")
        lines.append('Set them as Kaggle Secrets and export them before running boot.py.')
        raise RuntimeError('\n'.join(lines))

    return report


def env_check_masked():
    """Masked env/secret status for the /api/system/env-check endpoint."""
    def _mask(v):
        s = str(v)
        if not s or s == '0':
            return None
        return (s[:3] + '…' + s[-2:]) if len(s) > 6 else '•••'

    return {
        name: {'set': bool(value), 'masked': _mask(value), 'needed_for': purpose, 'critical': critical}
        for name, value, purpose, critical in _SECRET_SPECS
    }
