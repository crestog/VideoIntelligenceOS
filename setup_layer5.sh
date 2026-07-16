#!/usr/bin/env bash
# ============================================================
# VIOS V19 Layer-5 Kaggle Bootstrap
# Idempotent — safe to re-run in any order.
#
# Usage (in a Kaggle notebook cell):
#   !bash setup_layer5.sh
#
# What this does:
#   1. Installs system packages (Java, PostgreSQL, Neo4j, Redis, zstd)
#   2. Installs all Python packages needed by Layer-5 workers
#   3. Starts PostgreSQL + Redis as background services
#   4. Creates the PostgreSQL user + database (omni/omnidb)
#   5. Downloads + configures Neo4j Community 5.18.0
#   6. Downloads cloudflared + pgweb binaries
# ============================================================

set -e

# ────────────────────────────────────────────────────────────
# 1. SYSTEM PACKAGES
# ────────────────────────────────────────────────────────────
echo "📦 [1/6] Installing system packages..."
apt-get update -y -q
apt-get install -y -q \
    openjdk-17-jre \
    postgresql \
    postgresql-contrib \
    redis-server \
    zstd \
    ffmpeg \
    unzip \
    wget \
    curl

echo "   ✅ System packages installed."

# ────────────────────────────────────────────────────────────
# 2. PYTHON PACKAGES
# ────────────────────────────────────────────────────────────
echo "🐍 [2/6] Installing Python packages..."
pip install -q \
    qdrant-client \
    psycopg2-binary \
    neo4j \
    pyrogram \
    tgcrypto \
    opencv-python \
    nest-asyncio \
    "transformers>=4.45.0" \
    "pillow<12.0.0" \
    "sentence-transformers==2.7.0" \
    easyocr \
    accelerate \
    scipy \
    huggingface_hub \
    torchvision \
    bitsandbytes \
    qwen-vl-utils \
    tiktoken \
    einops \
    redis \
    openai \
    pyngrok \
    flask \
    faster-whisper \
    fastapi \
    uvicorn \
    aiofiles \
    aiosqlite

echo "   ✅ Python packages installed."

# ────────────────────────────────────────────────────────────
# 3. START POSTGRESQL + REDIS
# ────────────────────────────────────────────────────────────
echo "🗄️  [3/6] Starting PostgreSQL and Redis..."
service postgresql start || true
service redis-server start || true

# Give services a moment to boot
sleep 1
echo "   ✅ Services started."

# ────────────────────────────────────────────────────────────
# 4. CREATE POSTGRESQL USER + DATABASE
# ────────────────────────────────────────────────────────────
echo "🔑 [4/6] Provisioning PostgreSQL database..."
sudo -u postgres psql -c "CREATE USER omni WITH PASSWORD 'omni';" 2>/dev/null || echo "   (user 'omni' already exists)"
sudo -u postgres psql -c "CREATE DATABASE omnidb OWNER omni;"     2>/dev/null || echo "   (database 'omnidb' already exists)"
echo "   ✅ PostgreSQL ready (user: omni, db: omnidb)."

# ────────────────────────────────────────────────────────────
# 5. DOWNLOAD + CONFIGURE NEO4J COMMUNITY 5.18.0
# ────────────────────────────────────────────────────────────
NEO4J_VERSION="5.18.0"
NEO4J_DIR="/kaggle/working/neo4j-community-${NEO4J_VERSION}"

echo "🕸️  [5/6] Setting up Neo4j Community ${NEO4J_VERSION}..."
if [ ! -d "${NEO4J_DIR}" ]; then
    echo "   Downloading Neo4j..."
    wget -q -O /tmp/neo4j.tar.gz \
        "https://neo4j.com/artifact.php?name=neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
    tar -xf /tmp/neo4j.tar.gz -C /kaggle/working/
    rm /tmp/neo4j.tar.gz
    echo "   Configuring Neo4j..."
    cat >> "${NEO4J_DIR}/conf/neo4j.conf" <<'EOF'
dbms.security.auth_enabled=false
server.bolt.listen_address=0.0.0.0:7687
server.bolt.advertised_address=localhost:7687
EOF
    echo "   ✅ Neo4j installed and configured."
else
    echo "   ✅ Neo4j already present at ${NEO4J_DIR}."
fi

# ────────────────────────────────────────────────────────────
# 6. OPTIONAL BINARIES (cloudflared, pgweb)
# ────────────────────────────────────────────────────────────
echo "🔧 [6/6] Fetching helper binaries..."

# cloudflared (for tunnel)
if [ ! -f "./cloudflared" ]; then
    wget -q -O ./cloudflared \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    chmod +x ./cloudflared
    echo "   ✅ cloudflared downloaded."
else
    echo "   ✅ cloudflared already present."
fi

# pgweb (PostgreSQL browser)
if [ ! -f "./pgweb" ]; then
    wget -q -O /tmp/pgweb.zip \
        "https://github.com/sosedoff/pgweb/releases/download/v0.14.1/pgweb_linux_amd64.zip"
    unzip -q -o /tmp/pgweb.zip -d /tmp/pgweb_unzip/
    mv /tmp/pgweb_unzip/pgweb_linux_amd64 ./pgweb
    chmod +x ./pgweb
    rm -rf /tmp/pgweb.zip /tmp/pgweb_unzip
    echo "   ✅ pgweb downloaded."
else
    echo "   ✅ pgweb already present."
fi

# ────────────────────────────────────────────────────────────
# 7. QDRANT SERVER BINARY (server mode — local/embedded mode only
#    supports ONE process; Oracle + Embed + search need concurrent access)
# ────────────────────────────────────────────────────────────
echo "🔷 [7/7] Fetching Qdrant server binary..."
(
  if [ ! -f "./qdrant" ]; then
    wget -q -O /tmp/qdrant.tar.gz "https://github.com/qdrant/qdrant/releases/download/v1.18.1/qdrant-x86_64-unknown-linux-gnu.tar.gz"
    tar -xzf /tmp/qdrant.tar.gz -C .
    chmod +x ./qdrant
    rm -f /tmp/qdrant.tar.gz
  fi
) || echo "   ⚠️ Qdrant binary download failed — vector search will stay in single-process embedded mode."
[ -f "./qdrant" ] && echo "   ✅ Qdrant server binary ready."

echo ""
echo "============================================================"
echo "  VIOS V19 Layer-5 Bootstrap COMPLETE"
echo "  Run:  python boot.py"
echo "============================================================"
