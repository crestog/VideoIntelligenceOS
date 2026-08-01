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
    "DEDUP": "♻️  [DEDUP]",
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


def _safe_print(msg):
    """
    print() that cannot raise. The prefixes above are emoji, and a console
    without UTF-8 (cp1252 Windows terminals) raises UnicodeEncodeError on
    encode — which would turn a log line into a worker crash. Kaggle is UTF-8,
    so this is a guard for local runs and any redirected/piped stdout.
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
    except Exception:
        pass

# Redis connection (lazy init — may not be available at import time)
_redis_client = None
_redis_next_try = 0.0        # monotonic deadline; don't reconnect before this
_REDIS_RETRY_BACKOFF = 30.0  # seconds to stay quiet after a failed connect


def _get_redis():
    """
    Best-effort Redis handle for the cross-process log buffer.

    Backs off for _REDIS_RETRY_BACKOFF after a failure. Without this, a Redis
    outage made *every* log line attempt a fresh connect (2s timeout plus the
    client's own retries), so logging itself became the bottleneck at exactly
    the moment the operator needs to read the logs.
    """
    global _redis_client, _redis_next_try
    if _redis_client is not None:
        return _redis_client
    if time.monotonic() < _redis_next_try:
        return None
    try:
        client = redis.Redis(host='localhost', port=6379, decode_responses=True,
                             socket_timeout=2, socket_connect_timeout=2,
                             retry_on_timeout=False)
        client.ping()
        _redis_client = client
    except Exception:
        _redis_client = None
        _redis_next_try = time.monotonic() + _REDIS_RETRY_BACKOFF
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
    _safe_print(formatted)

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
