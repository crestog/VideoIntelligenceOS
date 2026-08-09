"""
VIOS Boot Orchestrator — Master Control Plane

Phases:
  1. Pre-flight Sweep: Kill zombie processes from previous crashes
  2. Message Broker: Boot Redis in-memory (no AOF — Kaggle sessions are ephemeral,
     and a committed AOF file previously corrupted the boot; see .gitattributes)
  3. Session Init: Detect fresh session vs crash recovery
  4. Ignition: Launch all worker processes with auto-healing watchdog threads
"""

import os
import subprocess
import sys
import threading
import time


BOOT_MARKER = "/tmp/vios_session_active"


# ══════════════════════════════════════════════════════════
# PHASE 0: CREDENTIALS
# ══════════════════════════════════════════════════════════
# Done before anything else imports `config`, which reads the environment once
# at import time and keeps what it found.
#
# Kaggle Secrets are an API, not environment variables, and vios/creds.py is the
# only file here that knows how to call it. Everything else — the harvester, the
# upload bot, Atlas, every worker below — reads os.environ and has no fallback
# value, because a literal default once published a live bot token from this
# public repo. So a session with all four secrets stored correctly still booted
# with "Telegram disabled", and the advice printed further down was to export
# them by hand in the launch cell: the one place a credential should never be
# typed. Asking once here fixes every process at the same time, since Popen
# hands this environment to each of them.
print("🔑 [SYSTEM] Phase 0: Reading credentials...", flush=True)
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vios import creds as _creds

    _bridged = _creds.export_to_env()
    if _bridged:
        print(f"   ✅ Kaggle Secrets → environment: "
              f"{', '.join(sorted(_bridged.values()))}", flush=True)
    elif _creds.on_kaggle():
        print("   ℹ️ No Kaggle Secrets to add (already set, or none stored).",
              flush=True)
except Exception as _e:
    print(f"   ⚠️ Kaggle Secrets unavailable: {type(_e).__name__}: {_e}",
          flush=True)


def stream_logs(pipe, prefix, is_engine=False):
    """Stream subprocess output to console with prefix tagging."""
    for line in iter(pipe.readline, ''):
        if is_engine and ("Loading weights:" in line or "%|" in line):
            continue
        print(f"{prefix} {line}", end="", flush=True)


def run_with_watchdog(command, prefix, is_engine):
    """Ensures the process stays alive forever. Restarts it if it crashes."""
    while True:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        stream_logs(process.stdout, prefix, is_engine)
        process.wait()

        print(f"\n⚠️ [WATCHDOG] {prefix} crashed (exit={process.returncode}). Rebooting in 3s...", flush=True)
        time.sleep(3)


# ══════════════════════════════════════════════════════════
# PHASE 1: PRE-FLIGHT SWEEP
# ══════════════════════════════════════════════════════════
print("🧹 [SYSTEM] Phase 1: Sweeping Zombie Processes...", flush=True)
os.system("pkill -9 ffmpeg > /dev/null 2>&1 || true")
os.system("pkill -9 ffprobe > /dev/null 2>&1 || true")
os.system("pkill -9 cloudflared > /dev/null 2>&1 || true")
print("   ✅ Zombie sweep complete.", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 2: MESSAGE BROKER
# ══════════════════════════════════════════════════════════
print("🗄️ [SYSTEM] Phase 2: Booting Redis Message Broker...", flush=True)
os.system("redis-server --daemonize yes > /dev/null 2>&1")

# Verify Redis actually answers. The old code slept 0.5s and printed "online"
# unconditionally, so a failed start was invisible and every worker then died on
# ECONNREFUSED inside an endless watchdog reboot loop.
redis_ready = False
for _ in range(10):
    try:
        probe = subprocess.run(["redis-cli", "ping"], capture_output=True,
                               text=True, timeout=3)
        if probe.returncode == 0 and "PONG" in probe.stdout:
            redis_ready = True
            break
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    time.sleep(1)

if not redis_ready:
    print("   ❌ Redis did not answer PING after 10s. Aborting boot.", flush=True)
    print("      Diagnose with:  redis-server --daemonize no", flush=True)
    print("      Common causes:  port 6379 already in use · redis-server not", flush=True)
    print("                      installed (rerun setup.sh) · stale appendonly.aof", flush=True)
    sys.exit(1)

print("   ✅ Redis broker online (in-memory, no AOF).", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 2.5: STORAGE BUDGET
# ══════════════════════════════════════════════════════════
# Printed before any model downloads so a too-small scratch tier is visible
# up front, rather than surfacing 3 minutes later as "No space left on device"
# in the middle of a 16 GB Qwen shard.
print("💽 [SYSTEM] Phase 2.5: Storage Budget...", flush=True)
try:
    from config import disk_report, SCRATCH_DIR, MODEL_CACHE_DIR

    for label, path, free in disk_report():
        print(f"   {label:<24} {free:6.1f} GB free   {path}", flush=True)

    _scratch_free = next((f for lbl, _, f in disk_report() if lbl.startswith('SCRATCH')), 0.0)
    # The two model stacks together pull ~28 GB of weights.
    if _scratch_free < 30:
        print(f"   ⚠️ Scratch has {_scratch_free:.1f} GB — the full model set needs ~28 GB.", flush=True)
        print("      Expect some models to fail to load. Free space or set", flush=True)
        print("      VIOS_SCRATCH_DIR to a larger volume.", flush=True)
except Exception as e:
    print(f"   ⚠️ Storage report unavailable: {e}", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 3: SESSION INIT — Fresh Session vs Crash Recovery
# ══════════════════════════════════════════════════════════
is_fresh_session = not os.path.exists(BOOT_MARKER)

if is_fresh_session:
    print("🆕 [SYSTEM] Phase 3: Fresh Session Detected — initializing clean state...", flush=True)

    try:
        from queue_manager import get_redis
        r = get_redis()

        # Count stale data for logging
        stale_dedup = r.scard("PROCESSED_VIDEOS_SET")
        stale_priority = r.llen("QUEUE_VISION_PRIORITY")
        stale_default = r.llen("QUEUE_VISION_DEFAULT")
        stale_proc = r.llen("QUEUE_VISION_PROCESSING")
        stale_dlq = r.llen("QUEUE_VISION_DLQ")

        # Flush everything — Kaggle session is ephemeral, old Redis state is stale
        r.flushall()

        if stale_dedup + stale_priority + stale_default + stale_proc + stale_dlq > 0:
            print(f"   🗑️ Flushed stale state: {stale_dedup} dedup entries, "
                  f"{stale_priority + stale_default} queued jobs, "
                  f"{stale_proc} in-flight, {stale_dlq} dead-lettered", flush=True)
        else:
            print("   ✅ Redis is clean — no stale data.", flush=True)

        # 3a. Rebuild the dedup set from the database — Redis is ephemeral but
        #     the DB is the source of truth for what is already processed.
        try:
            from dedup_manager import rebuild_dedup_set
            n = rebuild_dedup_set()
            if n:
                print(f"   ✅ Dedup set rebuilt from DB: {n} videos marked processed.", flush=True)
        except Exception as e:
            print(f"   ⚠️ Dedup rebuild skipped: {e}", flush=True)

        # Write session marker so crash recovery works within this session
        with open(BOOT_MARKER, 'w') as f:
            f.write(str(time.time()))
        print("   ✅ Session initialized.", flush=True)

    except Exception as e:
        print(f"   ⚠️ Session init error: {e}", flush=True)

else:
    print("🔄 [SYSTEM] Phase 3: Crash Recovery — same session, checking orphaned jobs...", flush=True)
    try:
        from queue_manager import recover_processing_jobs, get_queue_metrics
        recovered = 0
        for _q in ("QUEUE_VISION", "QUEUE_ANALYZE", "QUEUE_OMNI_VISION", "QUEUE_OMNI_ORACLE"):
            recovered += recover_processing_jobs(_q)
        if recovered > 0:
            print(f"   🔄 Recovered {recovered} orphaned job(s) from PROCESSING → DEFAULT queue.", flush=True)
        else:
            print("   ✅ No orphaned jobs — clean state.", flush=True)

        # Print current queue snapshot
        try:
            metrics = get_queue_metrics()
            for q_name, q_data in metrics.items():
                if q_name.startswith("_"):
                    continue
                p = q_data.get("pending_total", 0)
                proc = q_data.get("processing", 0)
                done = q_data.get("total_completed", 0)
                dlq = q_data.get("dead_letter", 0)
                print(f"   📊 {q_name}: {p} pending | {proc} processing | {done} completed | {dlq} dead-lettered", flush=True)
        except:
            pass

    except Exception as e:
        print(f"   ⚠️ Recovery skipped: {e}", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 4: IGNITION
# ══════════════════════════════════════════════════════════
try:
    from config import OMNI_ENABLED
except Exception:
    OMNI_ENABLED = False

# ── Credential preflight ──────────────────────────────────────────────────
# Reported here, once, before the workers start. Telegram credentials are
# env-only (they used to be committed literals, which put a live bot token in
# a public repo), so a notebook that forgets to export the secrets would
# otherwise show up as two separate workers failing deep inside pyrogram.
try:
    from config import missing_telegram_secrets, NIM_API_KEY

    _absent = missing_telegram_secrets()
    if _absent:
        print("", flush=True)
        print(f"⚠️ [SECRETS] Telegram disabled — not set: {', '.join(_absent)}", flush=True)
        print("      The web UI, CV pipeline, dashboard and queues all still run.", flush=True)
        print("      To enable channel harvesting and the upload bot, add these", flush=True)
        print("      as Kaggle Secrets (Add-ons → Secrets) and restart. Any of", flush=True)
        print("      VIOS_BOT_TOKEN / VIOS_TELEGRAM_BOT_TOKEN / TELEGRAM_BOT_TOKEN", flush=True)
        print("      is accepted, and likewise for CHANNEL_ID, API_ID, API_HASH.", flush=True)
        print("      Phase 0 above picks them up on its own — nothing needs", flush=True)
        print("      exporting in the launch cell.", flush=True)
    else:
        print("🔑 [SECRETS] Telegram credentials present.", flush=True)
    if not NIM_API_KEY:
        print("⚠️ [SECRETS] VIOS_NIM_API_KEY not set — GraphRAG entity extraction "
              "and answer synthesis are skipped.", flush=True)
except Exception as e:
    print(f"⚠️ [SECRETS] Preflight unavailable: {e}", flush=True)

# ── Restore hint ───────────────────────────────────────────────────────────
# A fresh container has an empty database: scratch is wiped and PostgreSQL runs
# from the Debian default data directory on that same ephemeral disk, so last
# session's narratives are gone. The bundles in the channel are the only copy.
#
# Deliberately a hint and not an automatic restore. It would have to run here,
# before ignition, and at this point omni_engine has not started PostgreSQL
# yet — so the Postgres half of a bundle could not be loaded, and a restore
# that silently recovers the harvest DB while dropping the narratives is worse
# than none. Restore is a two-step action in the admin panel instead, where it
# runs against live services and states what it will overwrite first.
if is_fresh_session:
    try:
        import sqlite3
        from config import DB_PATH as _DBP
        _posts = 0
        if os.path.exists(_DBP):
            _c = sqlite3.connect(_DBP, timeout=10)
            try:
                _posts = _c.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            except sqlite3.Error:
                pass
            finally:
                _c.close()
        if _posts == 0:
            print("", flush=True)
            print("📦 [RESTORE] Empty database on a fresh container.", flush=True)
            print("      If a bundle was exported previously, open /admin →", flush=True)
            print("      Database Restore → Check for a Bundle to continue that", flush=True)
            print("      database instead of re-narrating every reel.", flush=True)
    except Exception:
        pass          # a hint that cannot be printed is not worth a warning

print("", flush=True)
print("=" * 60, flush=True)
print("🚀 IGNITING VIDEO INTELLIGENCE OS", flush=True)
print("=" * 60, flush=True)
print("   🤖 [ENGINE]    → model_manager.py  (7 SOTA GPU models)", flush=True)
print("   🖥️ [UI]        → ui_server.py      (FastAPI + Ghost Worker)", flush=True)
print("   🎞️ [CV-ENGINE] → frame_worker.py   (OpenCV frame extraction)", flush=True)
if OMNI_ENABLED:
    print("   🔮 [OMNI]      → omni_engine.py    (Tri-partite DB + GraphRAG + Bot)", flush=True)
else:
    # Said out loud, because the silent version cost a session. VIOS_OMNI=0
    # switches off Neo4j, Postgres, GraphRAG, the narrative passes and /omni,
    # and with no line here the boot log of a crippled stack was identical to a
    # complete one — the failure only surfaced later as empty pages.
    print("   🔮 [OMNI]      → DISABLED (VIOS_OMNI=0) — no Neo4j, no Postgres,",
          flush=True)
    print("                    no GraphRAG, no narratives, /omni is a notice.",
          flush=True)
    print("                    Unset VIOS_OMNI in the launch cell to restore it.",
          flush=True)
print("=" * 60, flush=True)
print("", flush=True)

# Launch workers via Watchdog Threads
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "model_manager.py"], "🤖 [ENGINE]", True), daemon=True).start()
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "ui_server.py"], "🖥️ [UI]", False), daemon=True).start()
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "frame_worker.py"], "🎞️ [CV-ENGINE]", False), daemon=True).start()
if OMNI_ENABLED:
    threading.Thread(target=run_with_watchdog, args=(["python", "-u", "omni_engine.py"], "🔮 [OMNI]", True), daemon=True).start()

try:
    # Keep main orchestrator alive indefinitely
    while True:
        time.sleep(100)
except KeyboardInterrupt:
    print("\n🛑 [SYSTEM] Manual Shutdown Initiated.")
