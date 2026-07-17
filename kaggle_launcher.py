# ═══════════════════════════════════════════════════════════════════
# VIOS — ONE-CELL KAGGLE LAUNCHER  (paste this entire file into a
# single Kaggle notebook cell and run it; GPU T4 x2 recommended)
#
# What it does, in order:
#   1. Clones / fast-forwards feature/layer5-integration
#   2. Installs system deps (redis-server, ffmpeg) + Python packages,
#      then verifies every import actually works
#   3. Reads Kaggle Secrets → env vars, printing a checklist of
#      anything missing BEFORE the system starts
#   4. Downloads cloudflared (public UI tunnel) if not present
#   5. Idempotent pre-flight: re-running this cell kills the previous
#      run cleanly (no port/lock errors)
#   6. Boots the whole system via boot.py and prints a status table
#
# Required Kaggle Secrets (Add-ons → Secrets):
#   VIOS_API_ID, VIOS_API_HASH, VIOS_BOT_TOKEN, VIOS_CHANNEL_ID
# Optional:
#   VIOS_NVIDIA_API_KEY  (GraphRAG entity extraction)
# ═══════════════════════════════════════════════════════════════════

import os, subprocess, sys, time, shutil

REPO   = "https://github.com/crestog/VideoIntelligenceOS.git"
BRANCH = "feature/layer5-integration"
DIR    = "/kaggle/working/VideoIntelligenceOS"

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)

# ── 1. CLONE / UPDATE ────────────────────────────────────────────────
print("📦 [1/6] Fetching code…", flush=True)
if os.path.isdir(os.path.join(DIR, ".git")):
    r = sh(f"cd {DIR} && git fetch origin {BRANCH} && git checkout {BRANCH} && git reset --hard origin/{BRANCH}")
    print("   ✅ updated to", sh(f"cd {DIR} && git log --oneline -1").stdout.strip() or "(unknown)")
else:
    r = sh(f"git clone --depth 1 --branch {BRANCH} {REPO} {DIR}")
    if r.returncode != 0:
        print("   ❌ clone failed:\n", r.stderr[-800:]); raise SystemExit(1)
    print("   ✅ cloned", BRANCH)
os.chdir(DIR)

# ── 2. DEPENDENCIES (system + python) ────────────────────────────────
print("🔧 [2/6] Installing dependencies…", flush=True)
if not shutil.which("redis-server"):
    sh("apt-get install -y -qq redis-server > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq redis-server)")
if not shutil.which("ffmpeg"):
    sh("apt-get install -y -qq ffmpeg")
print("   redis-server:", "✅" if shutil.which("redis-server") else "❌ MISSING")
print("   ffmpeg      :", "✅" if shutil.which("ffmpeg") else "❌ MISSING")

sh(f"{sys.executable} -m pip install -q -r requirements.txt")
sh(f"{sys.executable} -m pip install -q qdrant-client neo4j sentence-transformers openai "
   f"bitsandbytes accelerate qwen-vl-utils psycopg2-binary")

_missing = []
for _m in ("redis", "fastapi", "uvicorn", "aiosqlite", "pyrogram", "transformers"):
    try:
        __import__(_m)
    except ImportError:
        _missing.append(_m)
print("   python pkgs :", "✅ all importable" if not _missing else f"❌ missing: {', '.join(_missing)}")
if _missing:
    raise SystemExit("Install failed — fix the packages above and re-run this cell.")

# ── 3. SECRETS → ENV (with checklist) ────────────────────────────────
print("🔐 [3/6] Loading Kaggle secrets…", flush=True)
SECRETS = [
    ("VIOS_API_ID",         True,  "Telegram sync"),
    ("VIOS_API_HASH",       True,  "Telegram sync"),
    ("VIOS_BOT_TOKEN",      True,  "Telegram sync"),
    ("VIOS_CHANNEL_ID",     True,  "Telegram sync"),
    ("VIOS_NVIDIA_API_KEY", False, "GraphRAG (optional)"),
]
try:
    from kaggle_secrets import UserSecretsClient
    _usc = UserSecretsClient()
    def _get(name):
        try: return _usc.get_secret(name)
        except Exception: return None
except Exception:
    def _get(name): return os.environ.get(name)

_missing_critical = False
for name, critical, purpose in SECRETS:
    val = _get(name)
    if val:
        os.environ[name] = str(val)
        print(f"   ✅ {name}")
    elif critical:
        _missing_critical = True
        print(f"   ❌ {name}  — MISSING (needed for {purpose}). Add it in Add-ons → Secrets.")
    else:
        print(f"   ⚠️ {name}  — not set ({purpose})")
if _missing_critical:
    print("   ⚠️ Boot will continue, but Telegram-dependent features WILL be disabled.")

# ── 4. CLOUDFLARED (public UI tunnel) ────────────────────────────────
print("🌐 [4/6] Preparing tunnel…", flush=True)
if not os.path.isfile("./cloudflared"):
    sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared")
print("   cloudflared :", "✅" if os.path.isfile("./cloudflared") else "❌ download failed (UI stays local at :8000)")

# ── 5. IDEMPOTENT PRE-FLIGHT (safe to re-run this cell) ─────────────
print("🧹 [5/6] Cleaning previous run…", flush=True)
for pat in ("boot.py", "ui_server.py", "model_manager.py", "oracle_worker.py",
            "vision_embed_worker.py", "frame_worker.py", "cloudflared", "redis-server", "qdrant"):
    sh(f"pkill -9 -f {pat} > /dev/null 2>&1 || true")
sh("rm -f /tmp/vios_session_active")
time.sleep(2)
print("   ✅ clean slate")

# ── 6. BOOT ──────────────────────────────────────────────────────────
print("🚀 [6/6] Booting VIOS — watch below for the public URL "
      "(look for 'WEB APP IS LIVE AT: https://…trycloudflare.com')\n", flush=True)

boot = subprocess.Popen([sys.executable, "-u", "boot.py"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

# Status table after warm-up, then stream logs forever
import threading
def _status_table():
    time.sleep(90)
    rows = []
    for label, pat in [("Redis", "redis-server"), ("UI server", "ui_server.py"),
                       ("Model manager", "model_manager.py"), ("Oracle", "oracle_worker.py"),
                       ("Embed worker", "vision_embed_worker.py"), ("Frame worker", "frame_worker.py"),
                       ("Tunnel", "cloudflared")]:
        alive = sh(f"pgrep -f {pat}").stdout.strip() != ""
        rows.append(f"   {'🟢' if alive else '🔴'} {label:<14} {'running' if alive else 'NOT running — check logs above'}")
    print("\n" + "═" * 52 + "\n📊 SERVICE STATUS (90s after boot)\n" + "\n".join(rows) + "\n" + "═" * 52 + "\n", flush=True)
threading.Thread(target=_status_table, daemon=True).start()

for line in iter(boot.stdout.readline, ""):
    print(line, end="", flush=True)
