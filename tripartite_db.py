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
# STATUS / OVERVIEW — resilient counts for the Database Explorer
# Every store is probed independently; a dead store returns
# {"status": "offline"} instead of raising.
# ═══════════════════════════════════════════════════════════
def qdrant_overview():
    try:
        client = get_qdrant()
        if client is None:
            return {"status": "offline", "reason": "qdrant-client unavailable"}
        collections = {}
        for name in QDRANT_COLLECTIONS:
            try:
                if client.collection_exists(name):
                    info = client.get_collection(name)
                    collections[name] = {
                        "points": int(getattr(info, "points_count", 0) or 0),
                        "dim": QDRANT_COLLECTIONS[name],
                    }
                else:
                    collections[name] = {"points": 0, "dim": QDRANT_COLLECTIONS[name], "missing": True}
            except Exception as e:
                collections[name] = {"error": str(e)[:120]}
        return {"status": "ok", "collections": collections}
    except Exception as e:
        return {"status": "offline", "reason": str(e)[:200]}


def postgres_overview():
    try:
        conn = get_pg_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM frames")
                frames = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM chunks")
                chunks = cur.fetchone()[0]
        conn.close()
        return {"status": "ok", "frames": frames, "chunks": chunks}
    except ImportError:
        return {"status": "offline", "reason": "psycopg2 not installed"}
    except Exception as e:
        return {"status": "offline", "reason": str(e)[:200]}


def neo4j_overview():
    driver = get_neo4j_driver()
    if driver is None:
        return {"status": "offline", "reason": "neo4j unreachable or not installed"}
    try:
        with driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            labels = {}
            for rec in session.run(
                    "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC LIMIT 10"):
                labels[rec["l"]] = rec["c"]
        return {"status": "ok", "nodes": nodes, "edges": edges, "labels": labels}
    except Exception as e:
        return {"status": "offline", "reason": str(e)[:200]}
    finally:
        try:
            driver.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# NEO4J GRAPH READS — power the interactive graph explorer
# All queries are read-only and LIMITed. Offline → {"status": "offline"}.
# ═══════════════════════════════════════════════════════════
def _node_dict(node):
    props = dict(node)
    labels = list(node.labels) if hasattr(node, "labels") else []
    ntype = labels[0] if labels else "Node"
    label = (props.get("name") or props.get("title") or props.get("uuid")
             or props.get("id") or ntype)
    return {"id": str(node.element_id), "label": str(label)[:80],
            "type": ntype, "props": {k: str(v)[:300] for k, v in props.items()}}


def fetch_graph(limit=200, q=None, label_filter=None):
    """Return {status, nodes, edges} for the interactive graph view."""
    driver = get_neo4j_driver()
    if driver is None:
        return {"status": "offline", "nodes": [], "edges": []}
    try:
        limit = max(1, min(int(limit), 1000))
        where, params = [], {"limit": limit}
        if q:
            where.append(
                "(toLower(coalesce(n.name,'')) CONTAINS toLower($q) "
                "OR toLower(coalesce(n.title,'')) CONTAINS toLower($q))")
            params["q"] = str(q)[:100]
        if label_filter:
            where.append(f"'{str(label_filter)[:40]}' IN labels(n)")
        where_clause = ("WHERE " + " AND ".join(where)) if where else ""

        nodes, edges, seen = [], [], set()
        with driver.session() as session:
            cypher = (f"MATCH (n) {where_clause} WITH n LIMIT $limit "
                      "OPTIONAL MATCH (n)-[r]-(m) "
                      "RETURN n, r, m LIMIT $limit")
            for rec in session.run(cypher, **params):
                for node in (rec["n"], rec["m"]):
                    if node is not None and node.element_id not in seen:
                        seen.add(node.element_id)
                        nodes.append(_node_dict(node))
                r = rec["r"]
                if r is not None:
                    edges.append({"source": str(r.start_node.element_id),
                                  "target": str(r.end_node.element_id),
                                  "type": r.type})
        return {"status": "ok", "nodes": nodes, "edges": edges}
    except Exception as e:
        return {"status": "offline", "reason": str(e)[:200], "nodes": [], "edges": []}
    finally:
        try:
            driver.close()
        except Exception:
            pass


def expand_node(node_id, limit=50):
    """Neighbors of one node for click-to-expand exploration."""
    driver = get_neo4j_driver()
    if driver is None:
        return {"status": "offline", "nodes": [], "edges": []}
    try:
        nodes, edges, seen = [], [], set()
        with driver.session() as session:
            cypher = ("MATCH (n)-[r]-(m) WHERE elementId(n) = $nid "
                      "RETURN n, r, m LIMIT $limit")
            for rec in session.run(cypher, nid=str(node_id), limit=max(1, min(int(limit), 200))):
                for node in (rec["n"], rec["m"]):
                    if node is not None and node.element_id not in seen:
                        seen.add(node.element_id)
                        nodes.append(_node_dict(node))
                r = rec["r"]
                if r is not None:
                    edges.append({"source": str(r.start_node.element_id),
                                  "target": str(r.end_node.element_id),
                                  "type": r.type})
        return {"status": "ok", "nodes": nodes, "edges": edges}
    except Exception as e:
        return {"status": "offline", "reason": str(e)[:200], "nodes": [], "edges": []}
    finally:
        try:
            driver.close()
        except Exception:
            pass


def get_node_detail(node_id):
    """Full properties of one node for the inspector panel."""
    driver = get_neo4j_driver()
    if driver is None:
        return {"status": "offline"}
    try:
        with driver.session() as session:
            rec = session.run(
                "MATCH (n) WHERE elementId(n) = $nid RETURN n", nid=str(node_id)).single()
            if not rec:
                return {"status": "not_found"}
            return {"status": "ok", "node": _node_dict(rec["n"])}
    except Exception as e:
        return {"status": "offline", "reason": str(e)[:200]}
    finally:
        try:
            driver.close()
        except Exception:
            pass


def graph_entity_search(q, limit=25):
    """Entity search: matching entities + the videos/chunks they connect to."""
    driver = get_neo4j_driver()
    if driver is None:
        return {"status": "offline", "results": []}
    try:
        results = []
        with driver.session() as session:
            cypher = (
                "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($q) "
                "OPTIONAL MATCH (e)<-[*1..2]-(c:Chunk)<-[*0..1]-(v:Video) "
                "RETURN e, collect(DISTINCT {chunk_id: c.id, msg_id: c.msg_id, "
                "start_t: c.start_t, video_uuid: v.uuid})[..10] AS ctx "
                "LIMIT $limit")
            for rec in session.run(cypher, q=str(q)[:100], limit=max(1, min(int(limit), 100))):
                ent = _node_dict(rec["e"])
                ctx = [c for c in (rec["ctx"] or []) if c and c.get("chunk_id")]
                results.append({"entity": ent, "contexts": ctx})
        return {"status": "ok", "results": results}
    except Exception as e:
        return {"status": "offline", "reason": str(e)[:200], "results": []}
    finally:
        try:
            driver.close()
        except Exception:
            pass


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
