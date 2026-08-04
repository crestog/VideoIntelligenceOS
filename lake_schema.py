"""
lake.db schema — one definition, callable from any process.

Every table in lake.db used to be created by whichever process happened to
need it first: `posts`/`creators`/`categories` by ui_server at __main__,
`videos` by frame_worker on its first write, the analysis tables by
model_manager at warm-up. That works while the file only ever grows.

It stops working the moment the file is deleted. A factory reset removes
lake.db while ui_server is still running — ui_server is the process running
the reset, so it is deliberately not restarted, and nothing calls its
init_db() again. The harvester's next INSERT then hits "no such table: posts"
and keeps hitting it until the whole container is rebooted.

So the schema lives here instead, and the reset re-applies it the moment the
file is gone. Everything is CREATE ... IF NOT EXISTS: calling this against a
populated database is a no-op, which is what lets the reset call it blindly.
"""

import sqlite3

from config import DB_PATH, SQLITE_TIMEOUT

# ── Tables ────────────────────────────────────────────────────────────────
# Kept as plain DDL strings so the same list can be applied by any caller.
# `videos` matches frame_worker/v17_backend exactly — the columns added later
# (created_at, fps, width, height) are in the CREATE here, and both of those
# modules still run their own ALTER-if-missing loop for databases that predate
# them, so an older file migrates and a fresh one is already correct.
_TABLES = (
    "CREATE TABLE IF NOT EXISTS categories "
    "(id INTEGER PRIMARY KEY, name TEXT UNIQUE)",

    "CREATE TABLE IF NOT EXISTS creators "
    "(id INTEGER PRIMARY KEY, username TEXT UNIQUE)",

    "CREATE TABLE IF NOT EXISTS posts "
    "(video_id INTEGER PRIMARY KEY, category_id INTEGER, creator_id INTEGER, "
    "likes INTEGER, caption TEXT, local_video_path TEXT, status TEXT)",

    # Scan coverage: every message id we have ALREADY asked Telegram about,
    # video or not. Without this, non-video ids are absent from `posts`
    # forever, so each scan re-fetched them and replayed the whole history.
    "CREATE TABLE IF NOT EXISTS scanned_ids (video_id INTEGER PRIMARY KEY)",

    "CREATE TABLE IF NOT EXISTS videos "
    "(msg_id INTEGER PRIMARY KEY, folder_id TEXT, title TEXT, "
    "frames INTEGER, duration_sec REAL, duration_str TEXT, "
    "thumb TEXT, first_frame TEXT, file_size_mb REAL, abs_path TEXT, "
    "created_at REAL, fps REAL, width INTEGER, height INTEGER)",

    "CREATE TABLE IF NOT EXISTS transcripts "
    "(id INTEGER PRIMARY KEY AUTOINCREMENT, msg_id INTEGER, "
    "start_sec REAL, end_sec REAL, text TEXT, created_at REAL)",

    "CREATE TABLE IF NOT EXISTS frame_notes "
    "(id INTEGER PRIMARY KEY AUTOINCREMENT, msg_id INTEGER, frame_idx INTEGER, "
    "ts_sec REAL, objects TEXT, ocr_text TEXT, description TEXT, "
    "created_at REAL)",
)

# ── Full-text indexes ─────────────────────────────────────────────────────
# Separate from _TABLES because fts5 can be missing from a stripped SQLite
# build. If it is, the rest of the schema must still apply.
_FTS = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS posts_search USING fts5"
    "(video_id UNINDEXED, caption, creator, category)",

    "CREATE VIRTUAL TABLE IF NOT EXISTS moments_search USING fts5"
    "(msg_id UNINDEXED, ts_sec UNINDEXED, source UNINDEXED, content)",
)

_TRIGGERS = (
    "CREATE TRIGGER IF NOT EXISTS sync_posts_search AFTER INSERT ON posts "
    "BEGIN INSERT INTO posts_search(video_id, caption, creator, category) "
    "VALUES (new.video_id, new.caption, "
    "(SELECT username FROM creators WHERE id = new.creator_id), "
    "(SELECT name FROM categories WHERE id = new.category_id)); END",
)

# ── Indexes ───────────────────────────────────────────────────────────────
# Every hot read filters on `local_video_path IS NOT NULL` and then joins
# creators/categories. Without these three, each poll from the player is a
# full table scan plus two nested-loop joins with no index on either foreign
# key — that is the bulk of what the UI shows as "buffering".
_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_posts_have_file ON posts(local_video_path)",
    "CREATE INDEX IF NOT EXISTS idx_posts_category  ON posts(category_id)",
    "CREATE INDEX IF NOT EXISTS idx_posts_creator   ON posts(creator_id)",
    "CREATE INDEX IF NOT EXISTS idx_transcripts_msg ON transcripts(msg_id)",
    "CREATE INDEX IF NOT EXISTS idx_frame_notes_msg ON frame_notes(msg_id)",
)


def ensure_lake_schema(db_path: str = None, log=None) -> dict:
    """Create anything missing in lake.db. Safe to call repeatedly.

    Returns {"ok", "applied", "skipped"} rather than raising on a partial
    failure: an index or an fts5 table that will not build should not stop the
    tables the harvester needs from existing.
    """
    path = db_path or DB_PATH
    applied, skipped = 0, []

    conn = sqlite3.connect(path, timeout=SQLITE_TIMEOUT)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        cur = conn.cursor()
        # Order matters: the trigger references posts_search, and the indexes
        # reference tables, so tables → fts → triggers → indexes.
        for ddl in _TABLES + _FTS + _TRIGGERS + _INDEXES:
            try:
                cur.execute(ddl)
                applied += 1
            except sqlite3.Error as e:
                name = ddl.split("EXISTS", 1)[-1].strip().split()[0]
                skipped.append(f"{name}: {e}")
                if log:
                    log(f"lake schema — skipped {name} ({e})")
        conn.commit()
    finally:
        conn.close()

    return {"ok": not skipped, "applied": applied, "skipped": skipped}


if __name__ == "__main__":
    print(ensure_lake_schema())
