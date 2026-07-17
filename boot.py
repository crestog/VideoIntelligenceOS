"""
VIOS Boot Orchestrator — Master Control Plane

Phases:
  1. Pre-flight Sweep: Kill zombie processes from previous crashes
  2. Message Broker: Boot Redis in AOF persistence mode
  3. Session Init: Detect fresh session vs crash recovery
  4. Ignition: Launch all worker processes with auto-healing watchdog threads
"""

import os
import subprocess
import threading
import time


BOOT_MARKER = "/tmp/vios_session_active"


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
# PHASE 0: CONFIG VALIDATION + DEPENDENCY CHECK
# Fail with ONE clear message instead of a cryptic mid-run error.
# ══════════════════════════════════════════════════════════
print("🔎 [SYSTEM] Phase 0: Validating configuration & dependencies...", flush=True)
try:
    from config import validate_config
    _report = validate_config(strict=False)
    for _m in _report["missing_critical"]:
        print(f"   ❌ MISSING SECRET: {_m['name']}  (needed for: {_m['needed_for']})", flush=True)
    for _m in _report["missing_optional"]:
        print(f"   ⚠️ optional secret not set: {_m['name']}  ({_m['needed_for']})", flush=True)
    if _report["ok"]:
        print("   ✅ All critical secrets present.", flush=True)
    else:
        print("   ⚠️ Boot continues, but Telegram-dependent features WILL fail "
              "until the secrets above are set.", flush=True)
except Exception as _e:
    print(f"   ⚠️ Config validation skipped: {_e}", flush=True)

_MISSING_DEPS = []
for _mod in ("redis", "fastapi", "uvicorn", "aiosqlite", "pyrogram"):
    try:
        __import__(_mod)
    except ImportError:
        _MISSING_DEPS.append(_mod)
if _MISSING_DEPS:
    print(f"   ❌ MISSING PYTHON PACKAGES: {', '.join(_MISSING_DEPS)} — "
          f"install them before booting (pip install {' '.join(_MISSING_DEPS)})", flush=True)
else:
    print("   ✅ Core Python dependencies importable.", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 1: PRE-FLIGHT SWEEP
# ══════════════════════════════════════════════════════════
print("💾 [SYSTEM] Phase 0b: Checking disk space...", flush=True)
import shutil as _shutil
_free_gb = _shutil.disk_usage("/kaggle/working").free / (1024**3) if os.path.isdir("/kaggle/working") else 999
print(f"   {'✅' if _free_gb >= 25 else '⚠️'} {_free_gb:.1f}GB free "
      f"(Whisper+Qwen+SigLIP+CLIP need ~22GB combined — clear /kaggle/working/huggingface_cache if low)", flush=True)

print("🧹 [SYSTEM] Phase 1: Sweeping Zombie Processes...", flush=True)
os.system("pkill -9 ffmpeg > /dev/null 2>&1 || true")
os.system("pkill -9 ffprobe > /dev/null 2>&1 || true")
os.system("pkill -9 cloudflared > /dev/null 2>&1 || true")
print("   ✅ Zombie sweep complete.", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 2: MESSAGE BROKER
# ══════════════════════════════════════════════════════════
print("🗄️ [SYSTEM] Phase 2: Booting Redis Message Broker (AOF mode)...", flush=True)
os.makedirs("/kaggle/working/redis_data", exist_ok=True)
os.system("redis-server --daemonize yes --appendonly yes "
          "--dir /kaggle/working/redis_data "
          "--logfile /kaggle/working/redis_data/redis.log")

# Readiness check with retry — no race between Redis boot and workers
_redis_ok = False
for _attempt in range(15):
    try:
        from queue_manager import get_redis
        get_redis().ping()
        _redis_ok = True
        break
    except Exception:
        time.sleep(0.5 + _attempt * 0.2)
if _redis_ok:
    print("   ✅ Redis broker online (ping verified).", flush=True)
else:
    print("   ❌ Redis did not become ready — queue-dependent workers will retry on their own.", flush=True)
    _log = "/kaggle/working/redis_data/redis.log"
    if os.path.exists(_log):
        print("   📄 redis.log tail:", flush=True)
        os.system(f"tail -n 20 {_log}")
    else:
        print("   📄 No redis.log — redis-server likely failed to launch at all.", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 3: SESSION INIT — Fresh Session vs Crash Recovery
# ══════════════════════════════════════════════════════════
is_fresh_session = not os.path.exists(BOOT_MARKER)

if is_fresh_session:
    print("🆕 [SYSTEM] Phase 3: Fresh Session Detected — initializing clean state...", flush=True)

    # 3a. If the database is missing (factory-reset Kaggle), try restoring the
    #     latest snapshot from the Telegram channel BEFORE anything else runs.
    #     This is what makes "never reprocess" true across sessions.
    try:
        from config import DB_PATH
        db_missing = not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0
        if db_missing and os.environ.get("VIOS_AUTO_IMPORT", "1") != "0":
            print("   📥 lake.db missing — attempting snapshot import from Telegram...", flush=True)
            import asyncio
            from snapshot_manager import import_snapshot
            restored = asyncio.run(import_snapshot())
            print(f"   {'✅ Snapshot restored — library is instantly searchable.' if restored else '⚠️ No snapshot found — starting with a fresh database.'}", flush=True)
    except Exception as e:
        print(f"   ⚠️ Snapshot auto-import skipped: {e}", flush=True)

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

        # 3b. Rebuild the dedup set from the database — Redis is ephemeral but
        #     the DB is the source of truth for what is already processed.
        try:
            from snapshot_manager import rebuild_dedup_set
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
        for _q in ("QUEUE_VISION", "QUEUE_ANALYZE", "QUEUE_ORACLE", "QUEUE_VISION_EMBED"):
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
# PHASE 3c: AUTHORITATIVE SQLITE SCHEMA (idempotent, after any snapshot restore)
# ══════════════════════════════════════════════════════════
try:
    from db_schema import init_sqlite_schema
    if init_sqlite_schema():
        print("   ✅ SQLite schema ready (all tables + FTS + WAL).", flush=True)
except Exception as _e:
    print(f"   ⚠️ SQLite schema init failed: {_e}", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 3d: STALE-JOB REAPER — background thread that recovers
# in-flight jobs whose worker died mid-processing.
# ══════════════════════════════════════════════════════════
def _reaper_loop():
    from queue_manager import reap_stale_jobs
    while True:
        time.sleep(300)  # every 5 minutes
        try:
            result = reap_stale_jobs()
            total = sum(result.values())
            if total:
                print(f"♻️ [REAPER] Requeued {total} stale in-flight job(s): {result}", flush=True)
        except Exception as _e:
            print(f"⚠️ [REAPER] sweep failed: {_e}", flush=True)

threading.Thread(target=_reaper_loop, daemon=True).start()
print("   ✅ Stale-job reaper armed (5-minute sweeps).", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 4: IGNITION
# ══════════════════════════════════════════════════════════
print("", flush=True)
print("=" * 60, flush=True)
print("🚀 IGNITING VIDEO INTELLIGENCE OS", flush=True)
print("=" * 60, flush=True)
print("   🤖 [ENGINE]    → model_manager.py       (YOLO + Whisper GPU models)", flush=True)
print("   🧠 [ORACLE]    → oracle_worker.py        (Qwen2.5-VL + GraphRAG)", flush=True)
print("   📐 [EMBED]     → vision_embed_worker.py  (SigLIP/CLIP/Depth/RAFT)", flush=True)
print("   🖥️ [UI]        → ui_server.py            (FastAPI + Ghost Worker)", flush=True)
print("   🎞️ [CV-ENGINE] → frame_worker.py         (ffmpeg dual-tier extraction)", flush=True)
print("=" * 60, flush=True)
print("", flush=True)

# ── Optional: Start Neo4j knowledge-graph engine ──
print("🕸️ [SYSTEM] Phase 4a: Attempting Neo4j startup...", flush=True)
try:
    from tripartite_db import start_neo4j, ensure_neo4j_schema, get_neo4j_driver, init_postgres
    try:
        init_postgres()
        print("   ✅ PostgreSQL schema ready (frames/chunks tables).", flush=True)
    except Exception as _pg_e:
        print(f"   ⚠️ PostgreSQL schema init failed: {_pg_e}", flush=True)
    neo4j_ok = start_neo4j(timeout_sec=30)
    if neo4j_ok:
        _drv = get_neo4j_driver()
        ensure_neo4j_schema(_drv)
        if _drv:
            _drv.close()
        print("   ✅ Neo4j knowledge graph online.", flush=True)
    else:
        print("   ⚠️ Neo4j unavailable — graph features disabled (non-fatal).", flush=True)
except Exception as _e:
    print(f"   ⚠️ Neo4j startup skipped: {_e}", flush=True)

print("🔷 [SYSTEM] Phase 4b: Starting Qdrant vector database server...", flush=True)


def _qdrant_binary_ok(path):
    """True if `path --version` exits 0. Returns (ok, stderr_snippet)."""
    try:
        _r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
        return _r.returncode == 0, (_r.stderr or _r.stdout).strip()[:200]
    except Exception as _e:
        return False, str(_e)[:200]


def _fetch_qdrant_musl(dest_dir, version="v1.18.2"):
    """
    Download the musl (statically linked) Qdrant build — the gnu build needs
    GLIBC 2.38 while Kaggle's Ubuntu 22.04 ships 2.35. Self-heals stale
    binaries left behind by older setup scripts. Returns True on success.
    """
    url = (f"https://github.com/qdrant/qdrant/releases/download/{version}/"
           f"qdrant-x86_64-unknown-linux-musl.tar.gz")
    print(f"   📥 Fetching Qdrant {version} (musl/static)...", flush=True)
    rc = os.system(f"wget -q -O /tmp/qdrant.tar.gz '{url}' "
                   f"&& tar -xzf /tmp/qdrant.tar.gz -C '{dest_dir}' "
                   f"&& chmod +x '{dest_dir}/qdrant' && rm -f /tmp/qdrant.tar.gz")
    return rc == 0


try:
    _qdrant_bin = os.path.join(os.getcwd(), "qdrant")

    # ── 0. Self-heal: broken (glibc-incompatible) or missing binary → musl build ──
    _ok, _err = (_qdrant_binary_ok(_qdrant_bin) if os.path.isfile(_qdrant_bin) else (False, "not present"))
    if not _ok:
        print(f"   ⚠️ Qdrant binary unusable ({_err}) — re-downloading static build...", flush=True)
        try:
            os.remove(_qdrant_bin)
        except OSError:
            pass
        _fetch_qdrant_musl(os.getcwd())

    if os.path.isfile(_qdrant_bin):
        # ── 1. Verify the binary is actually executable ──
        _ver_result = subprocess.run(
            [_qdrant_bin, "--version"],
            capture_output=True, text=True, timeout=10
        )
        if _ver_result.returncode != 0:
            print(f"   ❌ Qdrant binary exists but --version failed (rc={_ver_result.returncode}):", flush=True)
            print(f"      stdout: {_ver_result.stdout.strip()[:200]}", flush=True)
            print(f"      stderr: {_ver_result.stderr.strip()[:200]}", flush=True)
            raise RuntimeError("Qdrant binary broken")

        print(f"   ℹ️  {_ver_result.stdout.strip()}", flush=True)

        # ── 2. Prepare storage + config ──
        _qdrant_storage = "/kaggle/working/qdrant_data"
        _qdrant_log = "/kaggle/working/qdrant_data/qdrant_server.log"
        os.makedirs(_qdrant_storage, exist_ok=True)

        # Write a minimal YAML config so Qdrant doesn't rely on env-vars alone
        _qdrant_config = os.path.join(os.getcwd(), "qdrant_config.yaml")
        with open(_qdrant_config, "w") as _cf:
            _cf.write(
                f"storage:\n"
                f"  storage_path: {_qdrant_storage}\n"
                f"  snapshots_path: {_qdrant_storage}/snapshots\n"
                f"service:\n"
                f"  host: 0.0.0.0\n"
                f"  grpc_port: 6334\n"
                f"  http_port: 6333\n"
                f"telemetry_disabled: true\n"
            )

        # ── 3. Kill any leftover Qdrant from a prior run ──
        subprocess.run("pkill -9 -f qdrant", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

        # ── 4. Launch with stderr captured to a log file ──
        _qdrant_log_fh = open(_qdrant_log, "w")
        _qdrant_proc = subprocess.Popen(
            [_qdrant_bin, "--config-path", _qdrant_config],
            stdout=_qdrant_log_fh,
            stderr=subprocess.STDOUT,
        )

        # ── 5. Wait for readiness with early-exit on process death ──
        import urllib.request
        _qdrant_ready = False
        for _tick in range(40):
            # Check if process died
            _rc = _qdrant_proc.poll()
            if _rc is not None:
                _qdrant_log_fh.close()
                print(f"   ❌ Qdrant process exited immediately (exit code {_rc}).", flush=True)
                # Print the last 25 lines of the log
                try:
                    with open(_qdrant_log) as _lf:
                        _lines = _lf.readlines()
                        print("   📄 Qdrant log (last 25 lines):", flush=True)
                        for _ll in _lines[-25:]:
                            print(f"      {_ll.rstrip()}", flush=True)
                except Exception:
                    pass
                break
            try:
                urllib.request.urlopen("http://localhost:6333/", timeout=1)
                _qdrant_ready = True
                break
            except Exception:
                time.sleep(1)

        if _qdrant_ready:
            print("   ✅ Qdrant server online (localhost:6333).", flush=True)
        elif _qdrant_proc.poll() is None:
            # Still running but not responding
            print("   ⚠️ Qdrant server started but did not respond in 40s (localhost:6333).", flush=True)
            print(f"   📄 Check log: {_qdrant_log}", flush=True)
            try:
                with open(_qdrant_log) as _lf:
                    _lines = _lf.readlines()
                    print("   📄 Qdrant log (last 15 lines):", flush=True)
                    for _ll in _lines[-15:]:
                        print(f"      {_ll.rstrip()}", flush=True)
            except Exception:
                pass
    else:
        print("   ⚠️ ./qdrant binary not found (setup_layer5.sh may need re-running) — vector writes will fail this session.", flush=True)
except Exception as _qe:
    print(f"   ⚠️ Qdrant server launch skipped: {_qe}", flush=True)

# Launch workers via Watchdog Threads
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "model_manager.py"],      "🤖 [ENGINE]",  True), daemon=True).start()
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "ui_server.py"],           "🖥️  [UI]",     False), daemon=True).start()
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "frame_worker.py"],        "🎞️  [CV]",     False), daemon=True).start()
time.sleep(20)  # stagger: let model_manager clear its HF Hub downloads before Oracle starts its own
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "oracle_worker.py"],      "🧠 [ORACLE]",  True), daemon=True).start()
time.sleep(20)  # stagger again before the 3rd concurrent downloader starts
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "vision_embed_worker.py"], "📐 [EMBED]",   True), daemon=True).start()

try:
    # Keep main orchestrator alive indefinitely
    while True:
        time.sleep(100)
except KeyboardInterrupt:
    print("\n🛑 [SYSTEM] Manual Shutdown Initiated.")
