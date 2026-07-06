"""
VIOS Structured Logger — Unified Logging System

All subsystems use this logger for consistent, professional Kaggle output.
Logs are also stored in a Redis list for real-time UI access (Admin panel).
"""

import time
import redis
import json
from collections import deque

# ═══════════════════════════════════════════════════════════
# SUBSYSTEM PREFIXES
# ═══════════════════════════════════════════════════════════
SUBSYSTEMS = {
    "SYS":   "⚙️  [SYSTEM]",
    "UI":    "🖥️  [UI]",
    "CV":    "🎞️  [CV-ENGINE]",
    "AI":    "🤖  [AI-ENGINE]",
    "ADMIN": "🛡️  [ADMIN]",
    "QUEUE": "📡  [QUEUE]",
    "SNAP":  "📸  [SNAPSHOT]",
}

LEVELS = {
    "INFO":    "",
    "SUCCESS": "✅ ",
    "WARN":    "⚠️ ",
    "ERROR":   "❌ ",
}

# ═══════════════════════════════════════════════════════════
# IN-MEMORY LOG BUFFER (for Admin panel)
# ═══════════════════════════════════════════════════════════
LOG_BUFFER = deque(maxlen=500)

# Redis connection (lazy init — may not be available at import time)
_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True,
                                        socket_timeout=2, socket_connect_timeout=2)
            _redis_client.ping()
        except:
            _redis_client = None
    return _redis_client


def vios_log(message, subsystem="SYS", level="INFO"):
    """
    Log a message with structured formatting.
    
    Args:
        message: The log message
        subsystem: One of SYS, UI, CV, AI, ADMIN, QUEUE
        level: One of INFO, SUCCESS, WARN, ERROR
    """
    ts = time.strftime('%H:%M:%S')
    prefix = SUBSYSTEMS.get(subsystem, f"[{subsystem}]")
    level_icon = LEVELS.get(level, "")

    formatted = f"[{ts}] {prefix} {level_icon}{message}"
    print(formatted, flush=True)

    # Store in memory buffer
    entry = {
        "ts": ts,
        "time": time.time(),
        "subsystem": subsystem,
        "level": level,
        "message": message,
    }
    LOG_BUFFER.append(entry)

    # Push to Redis for cross-process access (best-effort)
    try:
        r = _get_redis()
        if r:
            r.lpush("VIOS_LOGS", json.dumps(entry))
            r.ltrim("VIOS_LOGS", 0, 499)  # Keep last 500
    except:
        pass


def get_recent_logs(count=200):
    """Get recent logs from Redis (cross-process) or memory buffer."""
    try:
        r = _get_redis()
        if r:
            items = r.lrange("VIOS_LOGS", 0, count - 1)
            return [json.loads(item) for item in items]
    except:
        pass
    return list(LOG_BUFFER)[-count:]
