"""
VIOS Central SQLite Schema — single idempotent initializer

Every table the system uses is created here at boot (CREATE TABLE IF NOT
EXISTS + ALTER TABLE migration guards), so no worker ever races another to
create a table, and a fresh Kaggle session starts with a complete schema.

Also configures WAL journal mode + busy_timeout on the database file so
concurrent workers never see "database is locked".

Workers keep their own defensive CREATE IF NOT EXISTS calls (harmless), but
the authoritative schema lives here.
"""

import sqlite3
import time

from config import DB_PATH, SQLITE_TIMEOUT
from logger import vios_log


def log(msg, level="INFO"):
    vios_log(msg, "SCHEMA", level)


def get_conn(row_factory=True):
    """Open a properly-configured SQLite connection (WAL, busy timeout)."""
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    conn.execute("PRAGMA busy_timeout = %d" % (SQLITE_TIMEOUT * 1000))
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


# ─── Migration guard: add a column if it doesn't exist ───
def _ensure_column(c, table, col, sqltype):
    cols = {row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()}
    if col not in cols:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sqltype}")
        log(f"migrated: {table}.{col} ({sqltype}) added")


def init_sqlite_schema():
    """
    Create/upgrade the full SQLite schema. Idempotent — safe to call from
    every process at startup. Returns True on success.
    """
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
    try:
        c = conn.cursor()

        # WAL survives on the file; set it once here
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError as e:
            log(f"WAL pragma skipped: {e}", "WARN")

        # ── Core video registry (written by frame_worker / v17_backend) ──
        c.execute('''CREATE TABLE IF NOT EXISTS videos
                     (msg_id INTEGER PRIMARY KEY, folder_id TEXT, title TEXT,
                      frames INTEGER, duration_sec REAL, duration_str TEXT,
                      thumb TEXT, first_frame TEXT, file_size_mb REAL, abs_path TEXT,
                      created_at REAL, fps REAL, width INTEGER, height INTEGER)''')
        for col, sqltype in (('created_at', 'REAL'), ('fps', 'REAL'),
                             ('width', 'INTEGER'), ('height', 'INTEGER')):
            _ensure_column(c, 'videos', col, sqltype)

        # ── Legacy vault tables (ui_server) ──
        c.execute("CREATE TABLE IF NOT EXISTS categories (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
        c.execute("CREATE TABLE IF NOT EXISTS creators (id INTEGER PRIMARY KEY, username TEXT UNIQUE)")
        c.execute('''CREATE TABLE IF NOT EXISTS posts
                     (video_id INTEGER PRIMARY KEY, category_id INTEGER, creator_id INTEGER,
                      likes INTEGER, caption TEXT, local_video_path TEXT, status TEXT)''')
        c.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS posts_search USING fts5
                     (video_id UNINDEXED, caption, creator, category)''')
        c.execute('''CREATE TRIGGER IF NOT EXISTS sync_posts_search AFTER INSERT ON posts BEGIN
                       INSERT INTO posts_search(video_id, caption, creator, category)
                       VALUES (new.video_id, new.caption,
                               (SELECT username FROM creators WHERE id = new.creator_id),
                               (SELECT name FROM categories WHERE id = new.category_id));
                     END;''')

        # ── Analysis tables (model_manager) ──
        c.execute('''CREATE TABLE IF NOT EXISTS transcripts
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      msg_id INTEGER, start_sec REAL, end_sec REAL,
                      text TEXT, created_at REAL)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_transcripts_msg ON transcripts(msg_id)')
        c.execute('''CREATE TABLE IF NOT EXISTS frame_notes
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      msg_id INTEGER, frame_idx INTEGER, ts_sec REAL,
                      objects TEXT, ocr_text TEXT, description TEXT,
                      created_at REAL)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_frame_notes_msg ON frame_notes(msg_id)')

        # ── FTS moment index (searched by /api/search/moments) ──
        c.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS moments_search USING fts5
                     (msg_id UNINDEXED, ts_sec UNINDEXED, source UNINDEXED, content)''')

        # ── Oracle narrative chunks (oracle_worker SQLite mirror) ──
        c.execute('''CREATE TABLE IF NOT EXISTS oracle_chunks
                     (chunk_id TEXT PRIMARY KEY, video_uuid TEXT, video_path TEXT,
                      msg_id INTEGER, start_t REAL, end_t REAL, mode TEXT, description TEXT)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_oracle_chunks_msg ON oracle_chunks(msg_id)')
        _ensure_column(c, 'oracle_chunks', 'created_at', 'REAL')

        # ── Processing event log (chunk/video timeline for the Data Inspector) ──
        c.execute('''CREATE TABLE IF NOT EXISTS processing_events
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      msg_id INTEGER, chunk_id TEXT, stage TEXT, status TEXT,
                      worker TEXT, detail TEXT, duration_sec REAL, created_at REAL)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_events_msg ON processing_events(msg_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_events_chunk ON processing_events(chunk_id)')

        conn.commit()
        log("SQLite schema ready (all tables + FTS + WAL)", "SUCCESS")
        return True
    except Exception as e:
        log(f"SQLite schema init failed: {e}", "ERROR")
        return False
    finally:
        conn.close()


def record_event(msg_id=None, chunk_id=None, stage="", status="ok",
                 worker="", detail="", duration_sec=None):
    """
    Append a processing event (best-effort — never raises, never blocks a
    worker). Powers the per-chunk 'everything that happened' timeline.
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        conn.execute("PRAGMA busy_timeout = %d" % (SQLITE_TIMEOUT * 1000))
        conn.execute(
            """INSERT INTO processing_events
               (msg_id, chunk_id, stage, status, worker, detail, duration_sec, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, chunk_id, stage, status, worker, str(detail)[:500],
             duration_sec, time.time()))
        conn.commit()
        conn.close()
    except Exception:
        pass  # observability must never break the pipeline


def sanitize_fts_query(q):
    """
    Turn arbitrary user input into a safe FTS5 MATCH expression.
    Every token is quoted (phrase syntax) so FTS operators/quotes in user
    input can never cause a syntax error. Returns '' for empty input.
    """
    if not q:
        return ""
    tokens = [t.replace('"', '') for t in q.strip().split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)
