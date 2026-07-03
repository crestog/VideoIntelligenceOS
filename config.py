"""
VIOS Configuration — Single Source of Truth

All paths, constants, queue names, and thresholds.
Every module imports from here instead of defining its own paths.
"""

import os

# ═══════════════════════════════════════════════════════════
# FILESYSTEM PATHS
# ═══════════════════════════════════════════════════════════
BASE_DIR   = '/kaggle/working/Insta-Vault'
LAKE_DIR   = os.path.join(BASE_DIR, 'DataLake')
DB_PATH    = os.path.join(LAKE_DIR, 'lake.db')
VIDEO_DIR  = os.path.join(LAKE_DIR, 'videos')
THUMB_DIR  = os.path.join(LAKE_DIR, '.thumbnails')
FLAG_DIR   = os.path.join(LAKE_DIR, '_Flagged_Dataset')
SESSION_DIR = os.path.join(LAKE_DIR, 'bot_session')
STATE_FILE = os.path.join(LAKE_DIR, 'state.txt')

# ═══════════════════════════════════════════════════════════
# QUEUE NAMES
# ═══════════════════════════════════════════════════════════
QUEUE_VISION = 'QUEUE_VISION'
QUEUE_MODELS = 'QUEUE_MODELS'

# ═══════════════════════════════════════════════════════════
# WORKER CONFIGURATION
# ═══════════════════════════════════════════════════════════
MAX_RETRIES              = 3       # Retry attempts before dead-lettering
DISK_PAUSE_THRESHOLD_GB  = 0.5    # CV engine pauses below this
DISK_WARN_THRESHOLD_GB   = 1.5    # CV engine warns below this
DISK_DL_PAUSE_GB         = 1.0    # Ghost Worker pauses downloads below this
BATCH_FRAME_COUNT        = 200    # Frames per binary batch fetch in V17

# ═══════════════════════════════════════════════════════════
# REDIS
# ═══════════════════════════════════════════════════════════
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# ═══════════════════════════════════════════════════════════
# ENSURE DIRECTORIES EXIST
# ═══════════════════════════════════════════════════════════
for d in [LAKE_DIR, VIDEO_DIR, THUMB_DIR, FLAG_DIR]:
    os.makedirs(d, exist_ok=True)
