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
import shutil
import subprocess
import time

from config import (QDRANT_PATH, OMNI_PG_DB, OMNI_PG_USER, OMNI_PG_PASSWORD,
                    OMNI_PG_HOST, NEO4J_HOME, NEO4J_BOLT, JAVA_HOME,
                    NEO4J_DATA_DIR)
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


def _find_java():
    """
    Locate a JRE for Neo4j. Returns a JAVA_HOME path, or None.

    Neo4j 5.x needs Java 17+. The configured default is Kaggle's usual OpenJDK
    17 location, but that moves between image builds, so fall back to probing.
    Returning None lets the caller report "no JRE" instead of Neo4j failing with
    an opaque error 30-90 seconds later.
    """
    if os.path.isfile(os.path.join(JAVA_HOME, "bin", "java")):
        return JAVA_HOME

    for base in ("/usr/lib/jvm",):
        if not os.path.isdir(base):
            continue
        # Newest version first — Neo4j rejects anything below 17.
        for name in sorted(os.listdir(base), reverse=True):
            cand = os.path.join(base, name)
            if os.path.isfile(os.path.join(cand, "bin", "java")):
                return cand

    java = shutil.which("java")
    if java:  # .../<home>/bin/java
        return os.path.dirname(os.path.dirname(os.path.realpath(java)))
    return None


# Config lines appended once to neo4j.conf. Kept as a list so the marker check
# below can detect a partially-configured file from an older revision.
_NEO4J_SETTINGS = [
    "dbms.security.auth_enabled=false",
    "server.bolt.listen_address=0.0.0.0:7687",
    # Storage on the scratch disk: the graph is fully rebuildable from
    # Postgres + the frames, so it must not consume the 20 GB output quota.
    f"server.directories.data={NEO4J_DATA_DIR}/data",
    f"server.directories.logs={NEO4J_DATA_DIR}/logs",
    f"server.directories.transaction.logs.root={NEO4J_DATA_DIR}/transactions",
    # Explicit, modest heap. Unset, the JVM claims 25% of the 15 GB box (~3.8 GB)
    # on top of two model stacks already competing for RAM — that OOM is the most
    # likely reason the graph never came up.
    "server.memory.heap.initial_size=512m",
    "server.memory.heap.max_size=1g",
    "server.memory.pagecache.size=512m",
    # A single small instance does not need 4 GC threads on a 4-vCPU box.
    "server.jvm.additional=-XX:ParallelGCThreads=2",
]
_NEO4J_MARKER = "# ── VIOS managed settings ──"


def _write_neo4j_conf():
    """Append VIOS settings to neo4j.conf exactly once. Returns True on success."""
    conf = os.path.join(NEO4J_HOME, "conf", "neo4j.conf")
    try:
        with open(conf, "r", encoding="utf-8") as f:
            content = f.read()
        if _NEO4J_MARKER not in content:
            with open(conf, "a", encoding="utf-8") as f:
                f.write("\n" + _NEO4J_MARKER + "\n")
                f.write("\n".join(_NEO4J_SETTINGS) + "\n")
        return True
    except OSError as e:
        log(f"Neo4j config write failed ({e}) — graph disabled", "WARN")
        return False


def _start_neo4j():
    """
    Launch Neo4j from the community tarball if present (daemonizes itself).

    Every failure path logs a specific reason. The previous version swallowed
    stdout/stderr into capture_output and discarded it, so a failed start
    produced only "Neo4j: unreachable" 60s later with nothing to act on.
    """
    neo4j_bin = os.path.join(NEO4J_HOME, "bin", "neo4j")
    if not os.path.exists(neo4j_bin):
        log(f"Neo4j not installed at {NEO4J_HOME} — knowledge graph disabled "
            f"(setup.sh downloads it; check the tarball step)", "WARN")
        return False

    java_home = _find_java()
    if not java_home:
        log("No Java 17+ runtime found — Neo4j cannot start, knowledge graph "
            "disabled (install openjdk-17-jre)", "WARN")
        return False

    for sub in ("data", "logs", "transactions"):
        try:
            os.makedirs(os.path.join(NEO4J_DATA_DIR, sub), exist_ok=True)
        except OSError as e:
            log(f"Neo4j data dir {NEO4J_DATA_DIR} not writable ({e}) — graph disabled", "WARN")
            return False

    if not _write_neo4j_conf():
        return False

    env = os.environ.copy()
    env["JAVA_HOME"] = java_home
    # NEO4J_CONF is read by the launcher script; without it a relocated
    # install can pick up a stale conf from a different prefix.
    env["NEO4J_CONF"] = os.path.join(NEO4J_HOME, "conf")

    try:
        proc = subprocess.run([neo4j_bin, "start"], env=env, capture_output=True,
                              text=True, timeout=180)
    except FileNotFoundError as e:
        log(f"Neo4j launcher not executable: {e}", "WARN")
        return False
    except subprocess.TimeoutExpired:
        log("Neo4j start timed out after 180s — graph disabled", "WARN")
        return False

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().replace("\n", " · ")
        log(f"Neo4j start failed (exit {proc.returncode}): {detail[:300]}", "WARN")
        return False

    return True


def _tail_neo4j_log(lines=12):
    """Last few lines of neo4j.log — the only place the real boot error appears."""
    for candidate in (os.path.join(NEO4J_DATA_DIR, "logs", "neo4j.log"),
                      os.path.join(NEO4J_HOME, "logs", "neo4j.log")):
        try:
            with open(candidate, "r", encoding="utf-8", errors="replace") as f:
                tail = [ln.rstrip() for ln in f.readlines()[-lines:] if ln.strip()]
            if tail:
                return tail
        except OSError:
            continue
    return []


def ensure_services():
    """Start Postgres + Neo4j (idempotent), verify connectivity, set flags."""
    _start_postgres()
    neo4j_started = _start_neo4j()

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

    # Neo4j: JVM boot takes ~15-30s — poll bolt connectivity, but only if the
    # launcher actually reported success. Probing for 60s after a failed start
    # is what made this look like a mysterious timeout rather than a start error.
    last_err = None
    if neo4j_started:
        try:
            from neo4j import GraphDatabase
            for _ in range(20):
                try:
                    with GraphDatabase.driver(NEO4J_BOLT, auth=None) as driver:
                        driver.verify_connectivity()
                    AVAILABLE["neo4j"] = True
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(3)
        except ImportError:
            last_err = "neo4j Python driver not installed (pip install neo4j)"

    if AVAILABLE["neo4j"]:
        log("Neo4j: ✅ online (knowledge graph active)", "SUCCESS")
    else:
        log("Neo4j: ❌ unreachable (graph features off)", "WARN")
        if neo4j_started and last_err:
            log(f"   └─ last bolt error: {str(last_err)[:200]}", "WARN")
            for line in _tail_neo4j_log():
                log(f"   │ {line[:200]}", "WARN")
        log("   └─ VIOS continues without the graph: vectors + relational "
            "indexing are unaffected, GraphRAG entity queries are disabled.", "WARN")
    return AVAILABLE


# ═══════════════════════════════════════════════════════════
# POSTGRESQL
# ═══════════════════════════════════════════════════════════
def get_pg_conn():
    import psycopg2
    return psycopg2.connect(dbname=OMNI_PG_DB, user=OMNI_PG_USER,
                            password=OMNI_PG_PASSWORD, host=OMNI_PG_HOST,
                            connect_timeout=10)


# ── Graceful degradation: no-op stand-ins used when Postgres is unavailable ──
# Without these, a Postgres outage made every vision/oracle job raise on connect,
# burn all 3 retries and dead-letter — throwing away the Qdrant vectors and Neo4j
# graph writes that would have succeeded. Now the relational rows are skipped and
# the rest of the pipeline still lands.
class _NullCursor:
    def execute(self, *args, **kwargs):  return None
    def fetchone(self):                  return None
    def fetchall(self):                  return []
    def __enter__(self):                 return self
    def __exit__(self, *exc):            return False


class _NullConn:
    """Quacks like a psycopg2 connection; silently discards every write."""
    def cursor(self):          return _NullCursor()
    def commit(self):          return None
    def rollback(self):        return None
    def close(self):           return None
    def __enter__(self):       return self
    def __exit__(self, *exc):  return False


_pg_warned = False


def get_pg_conn_optional():
    """
    A live Postgres connection, or a no-op stand-in when PG is down.

    Callers keep their normal cursor/commit/close code path — writes just go
    nowhere when the backend is missing, so vector + graph indexing survives.
    """
    global _pg_warned
    if AVAILABLE["postgres"]:
        try:
            return get_pg_conn()
        except Exception as e:
            AVAILABLE["postgres"] = False
            log(f"PostgreSQL connection lost ({e}) — relational writes disabled", "WARN")
    if not _pg_warned:
        log("PostgreSQL unavailable — frames/chunks rows are being skipped. "
            "Vectors and graph still index; the Explorer table will look empty.", "WARN")
        _pg_warned = True
    return _NullConn()


def init_pg_schema():
    """Create the frames/chunks tables (idempotent)."""
    if not AVAILABLE["postgres"]:
        return False
    conn = None
    try:
        conn = get_pg_conn()
        with conn:
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
    finally:
        if conn:
            conn.close()


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
