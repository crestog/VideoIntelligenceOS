#!/bin/bash
echo "⚙️ [SYSTEM] Installing OS Packages..."
apt-get update -yqq
apt-get install -yqq ffmpeg redis-server > /dev/null 2>&1

echo "📦 [SYSTEM] Installing Python Environment..."
pip install -q -r requirements.txt

echo "🌍 [SYSTEM] Ensuring Network Tunnels..."
if [ ! -f cloudflared ]; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

echo "✅ [SYSTEM] Environment Provisioned."
