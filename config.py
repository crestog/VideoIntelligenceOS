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
# ENSURE DIRECTORIES EXIST
# ═══════════════════════════════════════════════════════════
for d in [LAKE_DIR, VIDEO_DIR, THUMB_DIR, FLAG_DIR]:
    os.makedirs(d, exist_ok=True)
