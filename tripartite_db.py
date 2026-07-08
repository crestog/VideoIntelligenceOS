"""
VIOS Tri-Partite Database Layer — Qdrant + PostgreSQL + Neo4j

Manages the three non-SQLite databases used by the Layer-5 intelligence pipeline:

  • Qdrant  — on-disk vector store (frames_siglip, frames_clip, chunks_bge)
  • PostgreSQL — relational store (frames table, chunks table)
  • Neo4j   — knowledge graph (Video → Chunk → Narrative → Entity)

All init functions fail gracefully with a WARNING log so the existing
SQLite-only VIOS pipeline continues to work on machines that don't have
these services installed (e.g. local Windows dev machines).

On Kaggle: boot.py starts PostgreSQL + Redis, the notebook cell starts Neo4j,
and Qdrant runs embedded (no server needed — just a local path).
"""

import os
import subprocess
import time
import traceback

from logger import vios_log
from config import (
    QDRANT_PATH,
    NEO4J_HOME, NEO4J_JAVA_HOME, NEO4J_BOLT_URL,
)


def log(msg, level="INFO"):
    vios_log(msg, "DB3", level)


# ═══════════════════════════════════════════════════════════
# QDRANT — embedded, on-disk vector store
# ═══════════════════════════════════════════════════════════
_qdrant_client = None

# Collection specs: name → vector dimension
QDRANT_COLLECTIONS = {
    "frames_siglip": 1152,   # SigLIP-SO400M-patch14-384
    "frames_clip":    768,   # CLIP-ViT-Large-patch14
    "chunks_bge":    1024,   # BGE-large-en-v1.5
}


def init_qdrant():
    """
    Create (or reuse) the embedded Qdrant client and ensure all 3 collections exist.
    Returns the QdrantClient instance, or None if qdrant-client is not installed.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        os.makedirs(QDRANT_PATH, exist_ok=True)
        client = QdrantClient(path=QDRANT_PATH)

        for name, dim in QDRANT_COLLECTIONS.items():
            if not client.collection_exists(name):
                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                log(f"✅ Qdrant collection '{name}' created (dim={dim})")
            else:
                log(f"📦 Qdrant collection '{name}' already exists")

        _qdrant_client = client
        log("✅ Qdrant online (embedded, on-disk)", "SUCCESS")
        return client

    except ImportError:
        log("⚠️ qdrant-client not installed — vector search disabled", "WARN")
        return None
    except Exception as e:
        log(f"❌ Qdrant init failed: {e}", "ERROR")
        traceback.print_exc()
        return None


def get_qdrant():
    """Return the cached Qdrant client (or initialize it on first call)."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = init_qdrant()
    return _qdrant_client


# ═══════════════════════════════════════════════════════════
# POSTGRESQL — relational store for frames + chunks metadata
# ═══════════════════════════════════════════════════════════
_PG_DSN = {
    "dbname":   os.environ.get("VIOS_PG_DB",   "omnidb"),
    "user":     os.environ.get("VIOS_PG_USER",  "omni"),
    "password": os.environ.get("VIOS_PG_PASS",  "omni"),
    "host":     os.environ.get("VIOS_PG_HOST",  "localhost"),
    "port":     int(os.environ.get("VIOS_PG_PORT", 5432)),
}


def get_pg_conn():
    """
    Return a new psycopg2 connection.
    Raises ImportError or psycopg2.OperationalError when unavailable — callers
    must handle this so the main pipeline isn't blocked.
    """
    import psycopg2  # noqa: F401 — intentional runtime import
    return psycopg2.connect(**_PG_DSN)


def init_postgres():
    """
    Ensure the `frames` and `chunks` tables exist in PostgreSQL.
    Returns True on success, False if PostgreSQL is unavailable.
    """
    try:
        conn = get_pg_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS frames (
                        frame_id   TEXT PRIMARY KEY,
                        video_uuid TEXT,
                        video_path TEXT,
                        msg_id     BIGINT,
                        timestamp  REAL,
                        frame_idx  INTEGER,
                        depth      REAL,
                        motion     REAL
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id   TEXT PRIMARY KEY,
                        video_uuid TEXT,
                        video_path TEXT,
                        msg_id     BIGINT,
                        start_t    REAL,
                        end_t      REAL,
                        mode       TEXT,
                        description TEXT
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_frames_msg ON frames(msg_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_msg ON chunks(msg_id)")
        conn.close()
        log("✅ PostgreSQL schema ready (frames + chunks)", "SUCCESS")
        return True
    except ImportError:
        log("⚠️ psycopg2 not installed — PostgreSQL disabled", "WARN")
        return False
    except Exception as e:
        log(f"⚠️ PostgreSQL init failed (service may not be running): {e}", "WARN")
        return False


# ═══════════════════════════════════════════════════════════
# NEO4J — knowledge graph subprocess lifecycle
# ═══════════════════════════════════════════════════════════
_neo4j_process = None


def start_neo4j(timeout_sec: int = 30) -> bool:
    """
    Start the Neo4j community server as a background subprocess.
    Blocks until the bolt port is reachable or timeout elapses.
    Returns True on success, False if unavailable.
    """
    global _neo4j_process

    console_bin = os.path.join(NEO4J_HOME, "bin", "neo4j")
    if not os.path.exists(console_bin):
        log(f"⚠️ Neo4j binary not found at {console_bin} — graph store disabled", "WARN")
        return False

    try:
        neo4j_env = os.environ.copy()
        neo4j_env["JAVA_HOME"] = NEO4J_JAVA_HOME

        _neo4j_process = subprocess.Popen(
            [console_bin, "console"],
            env=neo4j_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log("⏳ Neo4j starting...")

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                _check_neo4j_bolt()
                log("✅ Neo4j bolt interface online", "SUCCESS")
                return True
            except Exception:
                time.sleep(2)

        log(f"❌ Neo4j did not become ready within {timeout_sec}s", "ERROR")
        return False

    except Exception as e:
        log(f"❌ Neo4j startup error: {e}", "ERROR")
        return False


def stop_neo4j():
    """Terminate the Neo4j subprocess if it was started by this process."""
    global _neo4j_process
    if _neo4j_process:
        _neo4j_process.terminate()
        try:
            _neo4j_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _neo4j_process.kill()
        _neo4j_process = None
        log("🛑 Neo4j stopped")


def _check_neo4j_bolt():
    """Raise an exception if Neo4j bolt is not reachable."""
    from neo4j import GraphDatabase  # noqa: F401
    with GraphDatabase.driver(NEO4J_BOLT_URL, auth=None) as driver:
        driver.verify_connectivity()


def get_neo4j_driver():
    """
    Return a new Neo4j driver connected to NEO4J_BOLT_URL.
    Returns None if neo4j package is missing or bolt is unreachable.
    """
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_BOLT_URL, auth=None)
        driver.verify_connectivity()
        return driver
    except ImportError:
        log("⚠️ neo4j package not installed — graph store disabled", "WARN")
        return None
    except Exception as e:
        log(f"⚠️ Neo4j bolt unreachable: {e}", "WARN")
        return None


def ensure_neo4j_schema(driver):
    """
    Create constraints and indexes for the knowledge graph schema.
    Call once after Neo4j is online.
    """
    if driver is None:
        return
    try:
        with driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video)  REQUIRE v.uuid IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk)  REQUIRE c.id   IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        log("✅ Neo4j schema constraints ensured")
    except Exception as e:
        log(f"⚠️ Neo4j schema setup failed: {e}", "WARN")


# ═══════════════════════════════════════════════════════════
# CONVENIENCE — initialise everything at once
# ═══════════════════════════════════════════════════════════
def init_all(start_graph: bool = False):
    """
    Init Qdrant + PostgreSQL (and optionally start Neo4j).
    Returns (qdrant_client, pg_ok, neo4j_ok).
    """
    qdrant = init_qdrant()
    pg_ok = init_postgres()
    neo4j_ok = False
    if start_graph:
        neo4j_ok = start_neo4j()
        if neo4j_ok:
            drv = get_neo4j_driver()
            ensure_neo4j_schema(drv)
            if drv:
                drv.close()
    return qdrant, pg_ok, neo4j_ok
