"""
VIOS — Dedup Set Rebuild

Redis is ephemeral (flushed on every fresh Kaggle session); the SQLite lake is
the source of truth for what has already been processed. This module reconciles
the two on boot so a restart never reprocesses work that is already in the DB.

Previously this lived in snapshot_manager.py alongside the Telegram DB
export/import cycle. That feature was removed, but dedup rebuild never depended
on it — it only reads the local `videos` table.
"""

import sqlite3

from config import DB_PATH, SQLITE_TIMEOUT
from logger import vios_log


def log(msg, level="INFO"):
    vios_log(msg, "DEDUP", level)


def rebuild_dedup_set():
    """PROCESSED_VIDEOS_SET ← every msg_id in the `videos` table.
    This is what makes 'never reprocess' true across restarts."""
    try:
        from queue_manager import get_redis
        r = get_redis()
        conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        try:
            rows = conn.execute("SELECT msg_id FROM videos").fetchall()
        except sqlite3.OperationalError:
            rows = []  # fresh DB, no videos table yet
        conn.close()
        if rows:
            r.sadd("PROCESSED_VIDEOS_SET", *[str(row[0]) for row in rows])
        log(f"Dedup set rebuilt: {len(rows)} processed videos registered")
        return len(rows)
    except Exception as e:
        log(f"Dedup rebuild failed: {e}", "ERROR")
        return 0


if __name__ == "__main__":
    rebuild_dedup_set()
