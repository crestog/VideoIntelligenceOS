"""
VIOS Omniscient — Tri-Partite Database Layer

  PostgreSQL → relational truth (frames, chunks + Qwen narratives)
  Qdrant     → vector memory (SigLIP / CLIP frame vectors, BGE chunk vectors)
  Neo4j      → knowledge graph (Video→Chunk→Narrative + GraphRAG entities)

Everything degrades gracefully: each backend exposes an availability flag so
the engine keeps running (and reports clearly) when a service is down.

Qdrant runs EMBEDDED (single-process file lock) — only omni_engine.py may
import get_qdrant(). The dashboard lives inside the same process, so this
constraint is invisible in practice.
"""

import hashlib
import os
import subprocess
import time

from config import (QDRANT_PATH, OMNI_PG_DB, OMNI_PG_USER, OMNI_PG_PASSWORD,
                    OMNI_PG_HOST, NEO4J_HOME, NEO4J_BOLT, JAVA_HOME)
from logger import vios_log


def log(msg, level="INFO"):
    vios_log(msg, "OMNI", level)


# Availability flags — set by ensure_services()/init_*; read by the engine
AVAILABLE = {"postgres": False, "qdrant": False, "neo4j": False}


# ═══════════════════════════════════════════════════════════
# STABLE 63-BIT IDS
# The notebook used Python hash(), which is salted per process — vector IDs
# changed on every restart, orphaning old points and breaking lookups.
# md5-based IDs are deterministic forever.
# ═══════════════════════════════════════════════════════════
def stable_id64(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


# ═══════════════════════════════════════════════════════════
# SERVICE MANAGEMENT (idempotent — safe to call on every engine boot)
# ═══════════════════════════════════════════════════════════
def _start_postgres():
    """Start PostgreSQL and ensure the omni user/database exist."""
    try:
        subprocess.run(["service", "postgresql", "start"], capture_output=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # not a Debian-style host (local dev) — connection attempt decides
    # Idempotent role/db creation (|| true semantics)
    for sql in (f"CREATE USER {OMNI_PG_USER} WITH PASSWORD '{OMNI_PG_PASSWORD}';",
                f"CREATE DATABASE {OMNI_PG_DB} OWNER {OMNI_PG_USER};"):
        try:
            subprocess.run(["sudo", "-u", "postgres", "psql", "-c", sql],
                           capture_output=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            break


def _start_neo4j():
    """Launch Neo4j from the community tarball if present (daemonizes itself)."""
    neo4j_bin = os.path.join(NEO4J_HOME, "bin", "neo4j")
    if not os.path.exists(neo4j_bin):
        log(f"Neo4j not installed at {NEO4J_HOME} — knowledge graph disabled", "WARN")
        return
    conf = os.path.join(NEO4J_HOME, "conf", "neo4j.conf")
    try:  # one-time config: no auth, bind bolt on all interfaces
        with open(conf, "r+", encoding="utf-8") as f:
            content = f.read()
            if "dbms.security.auth_enabled=false" not in content:
                f.write("\ndbms.security.auth_enabled=false\n"
                        "server.bolt.listen_address=0.0.0.0:7687\n")
    except OSError:
        pass
    env = os.environ.copy()
    env["JAVA_HOME"] = JAVA_HOME
    try:
        subprocess.run([neo4j_bin, "start"], env=env, capture_output=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log(f"Neo4j start failed: {e}", "WARN")


def ensure_services():
    """Start Postgres + Neo4j (idempotent), verify connectivity, set flags."""
    _start_postgres()
    _start_neo4j()

    # Postgres: quick probe with retries (service start is fast)
    for _ in range(10):
        try:
            get_pg_conn().close()
            AVAILABLE["postgres"] = True
            break
        except Exception:
            time.sleep(2)
    log(f"PostgreSQL: {'✅ online' if AVAILABLE['postgres'] else '❌ unreachable'}",
        "SUCCESS" if AVAILABLE["postgres"] else "WARN")

    # Neo4j: JVM boot takes ~15-30s — poll bolt connectivity
    try:
        from neo4j import GraphDatabase
        for _ in range(20):
            try:
                with GraphDatabase.driver(NEO4J_BOLT, auth=None) as driver:
                    driver.verify_connectivity()
                AVAILABLE["neo4j"] = True
                break
            except Exception:
                time.sleep(3)
    except ImportError:
        pass
    log(f"Neo4j: {'✅ online' if AVAILABLE['neo4j'] else '❌ unreachable (graph features off)'}",
        "SUCCESS" if AVAILABLE["neo4j"] else "WARN")
    return AVAILABLE


# ═══════════════════════════════════════════════════════════
# POSTGRESQL
# ═══════════════════════════════════════════════════════════
def get_pg_conn():
    import psycopg2
    return psycopg2.connect(dbname=OMNI_PG_DB, user=OMNI_PG_USER,
                            password=OMNI_PG_PASSWORD, host=OMNI_PG_HOST,
                            connect_timeout=10)


def init_pg_schema():
    """Create the frames/chunks tables (idempotent)."""
    if not AVAILABLE["postgres"]:
        return False
    try:
        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute('''CREATE TABLE IF NOT EXISTS frames (
                    frame_id TEXT PRIMARY KEY, video_uuid TEXT, video_path TEXT,
                    timestamp REAL, frame_idx INTEGER, depth REAL, motion REAL
                )''')
                cur.execute('''CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, video_uuid TEXT, video_path TEXT,
                    start_t REAL, end_t REAL, description TEXT
                )''')
            conn.commit()
        log("PostgreSQL schema ready (frames, chunks)", "SUCCESS")
        return True
    except Exception as e:
        log(f"PostgreSQL schema init failed: {e}", "ERROR")
        AVAILABLE["postgres"] = False
        return False


# ═══════════════════════════════════════════════════════════
# QDRANT (embedded — single process only)
# ═══════════════════════════════════════════════════════════
_qdrant = None

COLLECTIONS = [("frames_siglip", 1152), ("frames_clip", 768), ("chunks_bge", 1024)]


def get_qdrant():
    """Embedded Qdrant client with the three Omniscient collections ensured."""
    global _qdrant
    if _qdrant is not None:
        return _qdrant
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        client = QdrantClient(path=QDRANT_PATH)
        for name, dim in COLLECTIONS:
            if not client.collection_exists(name):
                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        _qdrant = client
        AVAILABLE["qdrant"] = True
        log("Qdrant vector store ready (frames_siglip, frames_clip, chunks_bge)", "SUCCESS")
    except Exception as e:
        log(f"Qdrant init failed: {e}", "ERROR")
        AVAILABLE["qdrant"] = False
    return _qdrant


# ═══════════════════════════════════════════════════════════
# NEO4J
# ═══════════════════════════════════════════════════════════
_neo4j_driver = None


def get_neo4j():
    """Singleton Neo4j driver, or None when the graph DB is unavailable."""
    global _neo4j_driver
    if not AVAILABLE["neo4j"]:
        return None
    if _neo4j_driver is None:
        try:
            from neo4j import GraphDatabase
            _neo4j_driver = GraphDatabase.driver(NEO4J_BOLT, auth=None)
            _neo4j_driver.verify_connectivity()
        except Exception as e:
            log(f"Neo4j driver init failed: {e}", "WARN")
            AVAILABLE["neo4j"] = False
            _neo4j_driver = None
    return _neo4j_driver
