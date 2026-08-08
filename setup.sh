#!/bin/bash
echo "⚙️ [SYSTEM] Installing OS Packages..."
apt-get update -yqq
apt-get install -yqq ffmpeg redis-server > /dev/null 2>&1
command -v ffmpeg >/dev/null || echo "❌ [SYSTEM] ffmpeg did not install — frame extraction and every media pass will fail."

# ── Omniscient layer services (skippable with VIOS_OMNI=0) ──
if [ "${VIOS_OMNI:-1}" != "0" ]; then
    echo "🔮 [OMNI] Installing PostgreSQL + OpenJDK 17 (Neo4j runtime)..."
    # Not silenced into /dev/null any more. When this step failed — a stale
    # apt index, no network on the mirror, a held package — the only symptom
    # was "PostgreSQL: unreachable" printed by the engine several minutes
    # later, with the actual error long discarded.
    if ! apt-get install -yqq openjdk-17-jre postgresql postgresql-contrib zstd 2>&1 | tail -5; then
        echo "⚠️ [OMNI] apt install reported a failure — see the lines above."
    fi

    command -v psql >/dev/null \
        || echo "⚠️ [OMNI] psql is still missing — PostgreSQL will be unreachable and narratives will not be stored."
    # Neo4j 5 refuses anything below Java 17, and Kaggle images ship Java 11
    # for Spark. Saying so here beats a JVM error in neo4j.log 30s into boot.
    if command -v java >/dev/null; then
        JV=$(java -version 2>&1 | head -1)
        echo "🔮 [OMNI] Java: $JV"
    else
        echo "⚠️ [OMNI] No java on PATH — Neo4j cannot start, the knowledge graph will be off."
    fi

    # A cluster is not created on every image. Without one, `service postgresql
    # start` exits 0 having started nothing at all.
    if command -v pg_lsclusters >/dev/null; then
        if [ -z "$(pg_lsclusters --no-header 2>/dev/null)" ]; then
            echo "🔮 [OMNI] No PostgreSQL cluster — creating one..."
            pg_createcluster --start 16 main 2>/dev/null \
                || pg_createcluster --start 15 main 2>/dev/null \
                || pg_createcluster --start 14 main 2>/dev/null \
                || echo "⚠️ [OMNI] Could not create a PostgreSQL cluster."
        fi
        service postgresql start > /dev/null 2>&1
        pg_lsclusters --no-header 2>/dev/null | sed 's/^/🔮 [OMNI] cluster: /'
    fi

    # Neo4j lives on scratch, not in the repo: the tarball is ~250 MB extracted
    # and /kaggle/working is the small output quota. VIOS_NEO4J_HOME tells
    # config.py where it went; symlink keeps the legacy in-repo path working.
    NEO4J_VER="neo4j-community-5.18.0"
    if [ -d /kaggle/temp ]; then
        NEO4J_PARENT="/kaggle/temp/vios_scratch"
    else
        NEO4J_PARENT="/tmp/vios_scratch"
    fi
    mkdir -p "$NEO4J_PARENT"

    if [ ! -d "$NEO4J_PARENT/$NEO4J_VER" ]; then
        echo "🔮 [OMNI] Downloading Neo4j Community 5.18.0 → $NEO4J_PARENT ..."
        # dist.neo4j.org is the direct artifact host; the artifact.php redirect
        # has been rate-limited before and a 302 to an error page extracts as a
        # 4 KB "tarball" that fails silently. Verify the size either way.
        wget -q -O /tmp/neo4j.tar.gz \
            "https://dist.neo4j.org/${NEO4J_VER}-unix.tar.gz" \
            || wget -q -O /tmp/neo4j.tar.gz \
               "https://neo4j.com/artifact.php?name=${NEO4J_VER}-unix.tar.gz"
        SZ=$(stat -c %s /tmp/neo4j.tar.gz 2>/dev/null || echo 0)
        if [ "$SZ" -lt 50000000 ]; then
            echo "⚠️ [OMNI] Neo4j download is only ${SZ} bytes — not the tarball. Knowledge graph disabled."
            rm -f /tmp/neo4j.tar.gz
        else
            tar -xf /tmp/neo4j.tar.gz -C "$NEO4J_PARENT" && rm -f /tmp/neo4j.tar.gz \
                && echo "🔮 [OMNI] Neo4j extracted to $NEO4J_PARENT/$NEO4J_VER" \
                || echo "⚠️ [OMNI] Neo4j tarball would not extract — knowledge graph disabled."
        fi
    fi
    # Legacy path compatibility: omni_db falls back to ./neo4j-community-5.18.0
    if [ -d "$NEO4J_PARENT/$NEO4J_VER" ] && [ ! -e "$NEO4J_VER" ]; then
        ln -s "$NEO4J_PARENT/$NEO4J_VER" "$NEO4J_VER" 2>/dev/null || true
    fi
fi

echo "📦 [SYSTEM] Installing Python Environment..."
pip install -q -r requirements.txt || echo "⚠️ [SYSTEM] pip reported a failure — some passes will be skipped for missing packages."

echo "🌍 [SYSTEM] Ensuring Network Tunnels..."
if [ ! -f cloudflared ]; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

echo "✅ [SYSTEM] Environment Provisioned."
