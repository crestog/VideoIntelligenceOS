#!/bin/bash
echo "⚙️ [SYSTEM] Installing OS Packages..."
apt-get update -yqq
apt-get install -yqq ffmpeg redis-server > /dev/null 2>&1

# ── Omniscient layer services (skippable with VIOS_OMNI=0) ──
if [ "${VIOS_OMNI:-1}" != "0" ]; then
    echo "🔮 [OMNI] Installing PostgreSQL + OpenJDK 17 (Neo4j runtime)..."
    apt-get install -yqq openjdk-17-jre postgresql postgresql-contrib zstd > /dev/null 2>&1

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
        wget -q -O /tmp/neo4j.tar.gz \
            "https://neo4j.com/artifact.php?name=${NEO4J_VER}-unix.tar.gz" \
            && tar -xf /tmp/neo4j.tar.gz -C "$NEO4J_PARENT" && rm -f /tmp/neo4j.tar.gz \
            || echo "⚠️ [OMNI] Neo4j download failed — knowledge graph will be disabled."
    fi
    # Legacy path compatibility: omni_db falls back to ./neo4j-community-5.18.0
    if [ -d "$NEO4J_PARENT/$NEO4J_VER" ] && [ ! -e "$NEO4J_VER" ]; then
        ln -s "$NEO4J_PARENT/$NEO4J_VER" "$NEO4J_VER" 2>/dev/null || true
    fi
fi

echo "📦 [SYSTEM] Installing Python Environment..."
pip install -q -r requirements.txt

echo "🌍 [SYSTEM] Ensuring Network Tunnels..."
if [ ! -f cloudflared ]; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

echo "✅ [SYSTEM] Environment Provisioned."
