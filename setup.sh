#!/bin/bash
echo "⚙️ [SYSTEM] Installing OS Packages..."
apt-get update -yqq
apt-get install -yqq ffmpeg redis-server > /dev/null 2>&1

# ── Omniscient layer services (skippable with VIOS_OMNI=0) ──
if [ "${VIOS_OMNI:-1}" != "0" ]; then
    echo "🔮 [OMNI] Installing PostgreSQL + OpenJDK 17 (Neo4j runtime)..."
    apt-get install -yqq openjdk-17-jre postgresql postgresql-contrib zstd > /dev/null 2>&1

    if [ ! -d neo4j-community-5.18.0 ]; then
        echo "🔮 [OMNI] Downloading Neo4j Community 5.18.0..."
        wget -q -O neo4j.tar.gz "https://neo4j.com/artifact.php?name=neo4j-community-5.18.0-unix.tar.gz" \
            && tar -xf neo4j.tar.gz && rm -f neo4j.tar.gz \
            || echo "⚠️ [OMNI] Neo4j download failed — knowledge graph will be disabled."
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
