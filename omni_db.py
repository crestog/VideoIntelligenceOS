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
import json
import os
import re
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

# Why a backend is not available, in the operator's terms. Written by
# ensure_services(), read by the admin panel. A boolean says the graph is off;
# this says which of the six things that could have caused it actually did.
DIAGNOSTICS = {"postgres": [], "neo4j": [], "qdrant": []}


def _diag(service: str, line: str, level="WARN"):
    DIAGNOSTICS.setdefault(service, []).append(line)
    log(f"   │ {line}", level)


def _run(cmd, timeout=60, env=None):
    """Run a command, never raise. Returns (returncode, combined output).

    -1 means the binary is not there, -2 means it hung. Both are answers, and
    both used to be swallowed by a bare `except: pass` that left the operator
    with "unreachable" and nothing else.
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env)
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    except FileNotFoundError:
        return -1, f"{cmd[0]}: not installed"
    except subprocess.TimeoutExpired:
        return -2, f"{' '.join(cmd[:3])}: timed out after {timeout}s"
    except OSError as e:
        return -3, f"{cmd[0]}: {e}"


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
def _pg_socket_dirs():
    """Directories that actually hold a Postgres unix socket, best first.

    The Debian package puts it in `/var/run/postgresql`, which is why that path
    was hard-coded here — but `/run/postgresql` is the same directory on a
    systemd image only when the compat symlink exists, a source build defaults
    to `/tmp`, and a cluster started with an explicit
    `unix_socket_directories` can be anywhere. Naming one path meant that on an
    image which chose another, the socket transport silently did not exist and
    the whole layer's fate rested on TCP and a password — which is the other
    half of this fault. Look for the socket instead of assuming where it is.
    """
    import glob
    found, seen = [], set()
    for base in (os.environ.get("PGHOST") or "", "/var/run/postgresql",
                 "/run/postgresql", "/tmp", "/var/run/postgres"):
        # PGHOST may name a TCP host rather than a directory; only a path can
        # hold a socket.
        if not base.startswith("/") or base in seen:
            continue
        seen.add(base)
        if glob.glob(os.path.join(base, ".s.PGSQL.*")):
            found.append(base)
    # Nothing listening yet — the caller may be racing a starting cluster, so
    # still offer the packaged default rather than no socket transport at all.
    return found or ["/var/run/postgresql"]


def _psql(sql: str, timeout=30):
    """Run one statement as the Postgres superuser, whatever this box allows.

    Three transports, because the one that works depends on the image:
    `sudo -u postgres` on a normal box, `su postgres -c` when sudo is absent
    (slim containers frequently ship without it), and a plain socket connection
    when we are already root and peer auth lets us straight in. Trying only the
    first is why the omni role was never created on some images, and a missing
    role reads exactly like a dead server: "unreachable".
    """
    quoted = sql.replace("'", "'\\''")
    cmds = [["sudo", "-u", "postgres", "psql", "-v", "ON_ERROR_STOP=0",
             "-c", sql],
            ["su", "postgres", "-c", f"psql -c '{quoted}'"]]
    cmds += [["psql", "-U", "postgres", "-h", d, "-c", sql]
             for d in _pg_socket_dirs()]
    for cmd in cmds:
        rc, out = _run(cmd, timeout=timeout)
        if rc == -1:
            continue                      # this transport does not exist here
        return rc, out
    return -1, "no way to reach psql as the postgres superuser"


def _pg_cluster_state():
    """(version, name, status) of the first cluster, or None if there is none.

    A container image that installed postgresql without a locale ends up with
    the binaries and the init script but no cluster at all — and then
    `service postgresql start` exits 0 having started nothing. That silent
    success is the single most confusing failure in this whole layer.
    """
    rc, out = _run(["pg_lsclusters", "--no-header"], timeout=20)
    if rc != 0:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            return parts[0], parts[1], parts[3]
    return None


def _start_postgres():
    """Start PostgreSQL and ensure the omni user/database exist.

    Returns True when a server is listening. Every failure names itself.
    """
    if not shutil.which("psql") and not os.path.isdir("/etc/postgresql"):
        _diag("postgres", "PostgreSQL is not installed on this machine "
                          "(no psql, no /etc/postgresql). setup.sh installs it "
                          "— check that the apt step ran and had network.")
        return False

    rc, out = _run(["service", "postgresql", "start"], timeout=90)
    if rc not in (0, -1):
        _diag("postgres", f"`service postgresql start` exited {rc}: {out[:200]}")

    cluster = _pg_cluster_state()
    if cluster is None:
        # No cluster: create one. Kaggle's image has the packages but the
        # postinst that would normally create `main` does not always run.
        rc, out = _run(["pg_createcluster", "--start", "16", "main"], timeout=180)
        if rc != 0:
            for ver in ("15", "14", "13", "17"):
                rc, out = _run(["pg_createcluster", "--start", ver, "main"],
                               timeout=180)
                if rc == 0:
                    break
        if rc != 0:
            _diag("postgres", f"no cluster exists and one could not be created: "
                              f"{out[:200]}")
        else:
            log("PostgreSQL cluster created", "INFO")
        cluster = _pg_cluster_state()

    if cluster and cluster[2] != "online":
        rc, out = _run(["pg_ctlcluster", cluster[0], cluster[1], "start"],
                       timeout=120)
        if rc != 0:
            _diag("postgres", f"cluster {cluster[0]}/{cluster[1]} would not "
                              f"start: {out[:200]}")

    # Idempotent role/db creation. "already exists" is the expected answer on
    # every run after the first, so its exit code is not a failure.
    #
    # But "already exists" is not the end of it. A cluster that survived from an
    # earlier session — a restored data directory, a second boot inside the same
    # container — already has the role, with whatever password it was created
    # with. CREATE fails, the failure is swallowed as expected, and every TCP
    # connection for the rest of the session answers `FATAL: password
    # authentication failed for user "omni"` while the log says the layer came
    # up fine. The password we are going to authenticate with is the one we must
    # set, so set it unconditionally: ALTER is idempotent, cheap, and the only
    # statement here whose success means the connection will work.
    rc, out = _psql(f"CREATE USER {OMNI_PG_USER} WITH PASSWORD "
                    f"'{OMNI_PG_PASSWORD}' SUPERUSER;")
    if rc == -1:
        _diag("postgres", "psql is unreachable as the postgres superuser — the "
                          "omni role cannot be created")
    elif rc != 0 and "already exists" not in out:
        _diag("postgres", f"could not create the {OMNI_PG_USER} role: {out[:200]}")

    rc, out = _psql(f"ALTER USER {OMNI_PG_USER} WITH PASSWORD "
                    f"'{OMNI_PG_PASSWORD}' SUPERUSER;")
    if rc not in (0, -1):
        _diag("postgres", f"the {OMNI_PG_USER} role exists but its password "
                          f"could not be reconciled, so TCP connections will "
                          f"fail authentication: {out[:200]}")

    rc, out = _psql(f"CREATE DATABASE {OMNI_PG_DB} OWNER {OMNI_PG_USER};")
    if rc not in (0, -1) and "already exists" not in out:
        _diag("postgres", f"could not create the {OMNI_PG_DB} database: "
                          f"{out[:200]}")
    return True


def _java_version(home: str) -> int:
    """Major version of the JRE at `home`, or 0 if it will not answer.

    Presence is not enough. Kaggle images ship Java 11 for Spark, and Neo4j 5
    refuses it — but only after the launcher has forked, so the failure lands
    in neo4j.log thirty seconds later and the start command itself exits 0.
    """
    rc, out = _run([os.path.join(home, "bin", "java"), "-version"], timeout=25)
    if rc != 0 and not out:
        return 0
    m = re.search(r'version "?(\d+)', out)
    if not m:
        return 0
    major = int(m.group(1))
    return 8 if major == 1 else major       # 1.8.0_x is Java 8


def _find_java():
    """
    Locate a JRE Neo4j will accept. Returns a JAVA_HOME path, or None.

    Neo4j 5.x needs Java 17+. The configured default is Kaggle's usual OpenJDK
    17 location, but that moves between image builds, so fall back to probing —
    and check the version of each candidate rather than taking the first
    `bin/java` found, because the first one is often the wrong one.
    """
    seen = []

    def usable(path):
        if not path or not os.path.isfile(os.path.join(path, "bin", "java")):
            return False
        v = _java_version(path)
        seen.append(f"{path} (Java {v or '?'})")
        return v >= 17

    if usable(JAVA_HOME):
        return JAVA_HOME

    for base in ("/usr/lib/jvm",):
        if not os.path.isdir(base):
            continue
        # Newest version first — Neo4j rejects anything below 17.
        for name in sorted(os.listdir(base), reverse=True):
            cand = os.path.join(base, name)
            if usable(cand):
                return cand

    java = shutil.which("java")
    if java:  # .../<home>/bin/java
        cand = os.path.dirname(os.path.dirname(os.path.realpath(java)))
        if usable(cand):
            return cand

    if seen:
        _diag("neo4j", "found Java, but none of it is 17 or newer: "
                       + "; ".join(seen[:4]))
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
        _diag("neo4j", f"not installed at {NEO4J_HOME} — setup.sh downloads the "
                       f"community tarball; that step failed or was skipped")
        return False

    java_home = _find_java()
    if not java_home:
        _diag("neo4j", "no Java 17+ runtime — run `apt-get install -y "
                       "openjdk-17-jre`, or set VIOS_NEO4J_HOME's JAVA_HOME")
        return False

    for sub in ("data", "logs", "transactions"):
        try:
            os.makedirs(os.path.join(NEO4J_DATA_DIR, sub), exist_ok=True)
        except OSError as e:
            _diag("neo4j", f"data dir {NEO4J_DATA_DIR} is not writable: {e}")
            return False

    # A leftover pid file from a container that was killed rather than shut
    # down makes the launcher refuse to start with "already running", and the
    # process it names died with the previous session.
    for pid_file in (os.path.join(NEO4J_DATA_DIR, "run", "neo4j.pid"),
                     os.path.join(NEO4J_HOME, "run", "neo4j.pid")):
        try:
            with open(pid_file, encoding="utf-8") as fh:
                pid = int(fh.read().strip())
            os.kill(pid, 0)
        except (OSError, ValueError):
            try:
                os.remove(pid_file)
                log("Removed a stale Neo4j pid file from a previous session",
                    "INFO")
            except OSError:
                pass

    if not _write_neo4j_conf():
        return False

    env = os.environ.copy()
    env["JAVA_HOME"] = java_home
    # NEO4J_CONF is read by the launcher script; without it a relocated
    # install can pick up a stale conf from a different prefix.
    env["NEO4J_CONF"] = os.path.join(NEO4J_HOME, "conf")

    rc, out = _run([neo4j_bin, "start"], timeout=180, env=env)
    if rc == 0:
        return True
    # "already running" is success wearing a failure's exit code.
    if "already running" in out.lower():
        log("Neo4j was already running", "INFO")
        return True
    _diag("neo4j", f"launcher exited {rc}: "
                   f"{out.replace(chr(10), ' · ')[:300] or 'no output'}")
    return False


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
    for k in DIAGNOSTICS:
        DIAGNOSTICS[k] = []

    try:
        import psycopg2                                  # noqa: F401,PLC0415
        pg_driver = True
    except ImportError:
        pg_driver = False
        _diag("postgres", "the psycopg2 driver is not installed "
                          "(pip install psycopg2-binary)")

    pg_present = _start_postgres() if pg_driver else False
    neo4j_started = _start_neo4j()

    # Postgres: quick probe with retries (service start is fast)
    last_pg = None
    if pg_present:
        for _ in range(10):
            try:
                get_pg_conn().close()
                AVAILABLE["postgres"] = True
                break
            except Exception as e:
                last_pg = e
                time.sleep(2)
    log(f"PostgreSQL: {'✅ online' if AVAILABLE['postgres'] else '❌ unreachable'}",
        "SUCCESS" if AVAILABLE["postgres"] else "WARN")
    if not AVAILABLE["postgres"]:
        if last_pg:
            # Not truncated to a couple of lines: `get_pg_conn` now reports what
            # every transport answered, and which of them failed how is the
            # whole diagnosis. "password authentication failed" and "no such
            # file or directory" call for opposite fixes.
            _diag("postgres", f"last connection error: {str(last_pg)[:600]}")
        log("   └─ VIOS continues without it: narratives and frame rows are "
            "skipped, vectors still index, and the Explorer table will look "
            "empty. Everything else is unaffected.", "WARN")

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
            _diag("neo4j", f"bolt never answered on {NEO4J_BOLT}: "
                           f"{str(last_err)[:180]}")
            for line in _tail_neo4j_log():
                _diag("neo4j", f"log: {line[:180]}")
        log("   └─ VIOS continues without the graph: vectors + relational "
            "indexing are unaffected, GraphRAG entity queries are disabled.", "WARN")

    _publish_report()
    return AVAILABLE


def _publish_report():
    """Put the report where the UI process can read it.

    The omni layer runs as its own process, so `AVAILABLE` in the FastAPI
    worker is always three falses — the admin panel was reporting the flags of
    a module that had never started a service. Redis already carries the log
    buffer across the same boundary; the report rides with it.
    """
    try:
        from queue_manager import get_redis            # noqa: PLC0415
        r = get_redis()
        r.set("VIOS_OMNI_SERVICES", json.dumps(service_report()), ex=3600)
    except Exception:
        pass                # a report that cannot be published is not an error


def service_report() -> dict:
    """What is up, what is down, and why — for the admin panel.

    The flags alone were never actionable. This is the same information the
    boot log carries, but readable after the log has scrolled past.
    """
    return {
        "available": dict(AVAILABLE),
        "diagnostics": {k: list(v) for k, v in DIAGNOSTICS.items()},
        "postgres": {"database": OMNI_PG_DB, "user": OMNI_PG_USER,
                     "host": OMNI_PG_HOST, "sockets": _pg_socket_dirs(),
                     "cluster": "/".join(_pg_cluster_state() or ()) or "none"},
        "neo4j": {"home": NEO4J_HOME, "bolt": NEO4J_BOLT,
                  "installed": os.path.exists(
                      os.path.join(NEO4J_HOME, "bin", "neo4j")),
                  "java": _find_java() or ""},
        "qdrant": {"path": QDRANT_PATH},
    }


# ═══════════════════════════════════════════════════════════
# POSTGRESQL
# ═══════════════════════════════════════════════════════════
def get_pg_conn():
    """A live connection, over TCP or the unix socket, whichever answers.

    `localhost` alone is not enough on a default Debian cluster: the packaged
    pg_hba.conf authenticates host connections with scram, and an image that
    never set a password for the role rejects them while the socket — peer or
    trust — lets the same role straight in. Falling back means one less thing
    that has to be configured correctly for the layer to come up.

    The exception raised on total failure carries every attempt, not just the
    last one. Before, the last rung was TCP, so a session whose role password
    was stale reported `password authentication failed` and nothing about the
    socket having been tried at all — and the operator had no way to tell a
    dead cluster from a wrong password from a socket in an unexpected place.
    """
    import psycopg2
    attempts, tried = [], []
    for kw in ([{"host": OMNI_PG_HOST, "password": OMNI_PG_PASSWORD}]
               + [{"host": d} for d in _pg_socket_dirs()]
               + [{"host": "127.0.0.1", "password": OMNI_PG_PASSWORD}]):
        try:
            return psycopg2.connect(dbname=OMNI_PG_DB, user=OMNI_PG_USER,
                                    connect_timeout=10, **kw)
        except Exception as e:
            tried.append(kw["host"])
            attempts.append(f"{kw['host']}: "
                            f"{str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__}")
    raise RuntimeError(
        f"no route to Postgres as {OMNI_PG_USER}@{OMNI_PG_DB} — tried "
        f"{len(tried)}: " + " | ".join(attempts))


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
                # Generation provenance — the Job-detail panel reads these to
                # show WHEN a narrative was produced and how long it took, so a
                # stale row from a broken run is visible instead of silent.
                for col, coltype in (("created_at", "DOUBLE PRECISION"),
                                     ("gen_ms", "INTEGER"),
                                     ("mode", "TEXT")):
                    cur.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {col} {coltype}")

                # chunk_id embeds start_t, so switching blitz(15s)→omni(5s)
                # inserts a SECOND set of rows for the same video instead of
                # replacing the first. The video then shows both generations
                # interleaved on the timeline — one of the ways the same
                # narrative appeared to repeat. Position is the real identity.
                #
                # Existing databases already hold those duplicates, and CREATE
                # UNIQUE INDEX would just fail on them, so collapse first:
                # keep exactly ONE row per (video, position).
                #
                # This must be a total ranking, not a pairwise comparison. An
                # earlier version deleted `a` only when a.description was no
                # longer than b's, so a pair where the older row had the longer
                # text matched in neither direction, both rows survived, the
                # unique index below failed, and schema init fell into the
                # except branch — disabling Postgres entirely on every boot.
                # row_number() over a deterministic order can only ever keep one.
                cur.execute("""
                    DELETE FROM chunks WHERE ctid IN (
                        SELECT ctid FROM (
                            SELECT ctid, row_number() OVER (
                                PARTITION BY video_uuid, start_t
                                ORDER BY COALESCE(created_at, 0) DESC,
                                         COALESCE(length(description), 0) DESC,
                                         ctid DESC) AS rn
                            FROM chunks) ranked
                        WHERE rn > 1)""")
                removed = cur.rowcount
                if removed and removed > 0:
                    log(f"Collapsed {removed} duplicate chunk rows "
                        f"(same video + start time)", "WARN")
                cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS
                               chunks_video_start_uniq ON chunks (video_uuid, start_t)""")
                # The timeline endpoint filters by video_uuid on every open.
                cur.execute("""CREATE INDEX IF NOT EXISTS chunks_video_idx
                               ON chunks (video_uuid)""")
                cur.execute("""CREATE INDEX IF NOT EXISTS frames_video_idx
                               ON frames (video_uuid)""")
                # Corpus-wide duplicate detection joins on the description text.
                # md5 keeps the index small; TEXT itself can exceed the 8 KB
                # btree row limit on a long narrative.
                cur.execute("""CREATE INDEX IF NOT EXISTS chunks_desc_md5_idx
                               ON chunks (md5(description))""")
            conn.commit()
        log("PostgreSQL schema ready (frames, chunks + timeline indexes)", "SUCCESS")
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
