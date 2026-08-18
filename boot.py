"""
VIOS Boot Orchestrator — Master Control Plane

Phases:
  1. Pre-flight Sweep: Kill zombie processes from previous crashes
  2. Message Broker: Boot Redis in-memory (no AOF — Kaggle sessions are ephemeral,
     and a committed AOF file previously corrupted the boot; see .gitattributes)
  3. Session Init: Detect fresh session vs crash recovery
  4. Ignition: Launch all worker processes with auto-healing watchdog threads
"""

import json
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


WATCHDOG_STATE = {}

# The watchdog lives in this process; /api/status is served by ui_server.py in
# another one. So the state is also written to a file both can see — a worker
# that is flapping has to be visible on the page, not only in a log nobody is
# reading at hour nine. One small JSON write per crash and per launch; nothing
# polls it here.
#
# The path is resolved on first use, not at import: `config` reads the
# environment once and keeps it, so nothing above may import it before Phase 0
# has bridged the Kaggle Secrets in.
_WATCHDOG_FILE = ""


def _watchdog_file() -> str:
    global _WATCHDOG_FILE
    if not _WATCHDOG_FILE:
        from config import LAKE_DIR
        _WATCHDOG_FILE = os.path.join(LAKE_DIR, "watchdog.json")
    return _WATCHDOG_FILE


def _publish_watchdog():
    """Best-effort: a failed write must never take a worker down with it."""
    try:
        path = _watchdog_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"at": time.time(), "workers": WATCHDOG_STATE}, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def run_with_watchdog(command, prefix, is_engine):
    """Keep a worker alive, with a backoff so a broken one cannot drown the log.

    The three-second restart was fine for a worker that crashes once. For one
    that cannot start at all — a missing weight, a syntax error, a port already
    bound — it produced twenty tracebacks a minute forever, and the errors worth
    reading scrolled past between them. So the delay grows, capped, and resets
    once the process has stayed up long enough to have actually run.

    The counters are published in WATCHDOG_STATE, and through it to
    watchdog.json, so /api/status can show that a worker is flapping rather than
    working.
    """
    delay, crashes = 3, 0
    while True:
        started = time.time()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        WATCHDOG_STATE[prefix] = {"pid": process.pid, "crashes": crashes,
                                  "since": started, "delay": delay}
        _publish_watchdog()
        stream_logs(process.stdout, prefix, is_engine)
        process.wait()
        lived = time.time() - started

        if lived >= 120:
            # It ran long enough to have done work. Whatever killed it is not
            # a startup failure, so do not punish the next attempt.
            delay, crashes = 3, 0
        else:
            crashes += 1
            delay = min(delay * 2, 120)
        WATCHDOG_STATE[prefix] = {"pid": 0, "crashes": crashes,
                                  "since": started, "delay": delay,
                                  "exit": process.returncode,
                                  "lived": round(lived, 1)}
        _publish_watchdog()
        note = f" — {crashes} in a row" if crashes > 1 else ""
        print(f"\n⚠️ [WATCHDOG] {prefix} crashed (exit={process.returncode}, "
              f"up {lived:.0f}s{note}). Rebooting in {delay}s...", flush=True)
        time.sleep(delay)


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
        from queue_manager import recover_processing_jobs, get_queue_metrics, V2_QUEUES
        recovered = 0
        # The v1 lanes plus the v2 processing plane's four. A v2 GPU worker that
        # dies holding a job — the correct response to an unrecoverable CUDA
        # fault — leaves that video's cohort slice in PROCESSING, and without
        # recovery it is never re-offered. The row's lease expiry would free it
        # eventually; recovering the hint means the next worker takes it now.
        for _q in ("QUEUE_VISION", "QUEUE_ANALYZE", "QUEUE_OMNI_VISION",
                   "QUEUE_OMNI_ORACLE", *V2_QUEUES):
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

# ── Who owns the GPU ──────────────────────────────────────────────────────
# Two planes in this repository load models, and until now both did, onto the
# same card, with nothing between them. `model_manager.py` and the omniscient
# vision worker keep SigLIP, CLIP, DINOv2, Whisper large-v3, RAFT, YOLO and
# EasyOCR warm on cuda:0 — about 5.5 GiB that is never released — while the v2
# processing engine bin-packs its cohorts against whatever `resources.probe()`
# saw free at the moment it looked. The result on a real run was neither plane
# working: v2 filled the card, then v1 could not allocate 34 MiB and its vision
# jobs went RETRIED → DEAD in a loop.
#
# This is not a Kaggle problem to fight. It is two copies of the same models
# competing, because the v1 plane's seven models are a strict subset of what the
# v2 registry already runs at higher coverage. So one plane owns the card, and
# it is the one whose output the product is built on.
#
#   VIOS_GPU_OWNER=v2  (default) — the processing engine gets the whole card.
#                                  model_manager and omni_engine do not start.
#                                  The Postgres/Neo4j/Redis services and every
#                                  CPU worker are untouched.
#   VIOS_GPU_OWNER=v1            — the old behaviour, both planes, for when the
#                                  omniscient dashboard is what you are working
#                                  on and the v2 sweep is stopped.
#   VIOS_GPU_OWNER=both          — explicitly opt back into the contention.
GPU_OWNER = (os.environ.get("VIOS_GPU_OWNER", "") or "v2").strip().lower()
if GPU_OWNER not in ("v1", "v2", "both"):
    GPU_OWNER = "v2"
V1_GPU = GPU_OWNER in ("v1", "both")
V2_GPU = GPU_OWNER in ("v2", "both")
os.environ["VIOS_GPU_OWNER"] = GPU_OWNER

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

# ── Restore notice ─────────────────────────────────────────────────────────
# A fresh container has an empty database: scratch is wiped and PostgreSQL runs
# from the Debian default data directory on that same ephemeral disk, so last
# session's narratives are gone. The bundles in the channel are the only copy.
#
# The restore still cannot happen *here*. At this point omni_engine has not
# started PostgreSQL, so the Postgres half of a bundle could not be loaded, and
# a restore that silently recovers the harvest DB while dropping the narratives
# is worse than none. That reasoning did not change — the action moved instead.
# `vios/process/routes.py: autostart` now runs it from the web process, where
# the services are live: it waits for Postgres, restores the harvest database if
# and only if it is empty, replays every evidence shard in the channel, and then
# reconciles the coverage table so nothing already done is processed again.
# This block says so, and says how to turn it off.
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
            print("📦 [RESTORE] Empty database on a fresh container —", flush=True)
            print("      automatic restore armed. Once the services are up it", flush=True)
            print("      pulls the last bundle, replays every evidence shard,", flush=True)
            print("      and marks what is already done so it is not redone.", flush=True)
            print("      Watch it at /process. VIOS_PROCESS_AUTOSTART=0 turns", flush=True)
            print("      it off; /admin → Database Restore is the manual path.", flush=True)
    except Exception:
        pass          # a notice that cannot be printed is not worth a warning

print("", flush=True)
print("=" * 60, flush=True)
print("🚀 IGNITING VIDEO INTELLIGENCE OS", flush=True)
print("=" * 60, flush=True)
if V1_GPU:
    print("   🤖 [ENGINE]    → model_manager.py  (7 SOTA GPU models)", flush=True)
print("   🖥️ [UI]        → ui_server.py      (FastAPI + Ghost Worker)", flush=True)
print("   🎞️ [CV-ENGINE] → frame_worker.py   (OpenCV frame extraction)", flush=True)
if OMNI_ENABLED and V1_GPU:
    print("   🔮 [OMNI]      → omni_engine.py    (Tri-partite DB + GraphRAG + Bot)", flush=True)
elif OMNI_ENABLED:
    # Started either way now. The God-Mode Explorer is a browser over Postgres,
    # Neo4j and Qdrant — none of which wants a card — and the footer links /omni
    # from every page in the system. Not starting this process meant that link
    # answered "still booting…" for the whole session about something that was
    # never going to boot.
    print("   🔮 [OMNI]      → omni_engine.py    (Explorer only — DBs + GraphRAG,",
          flush=True)
    print("                    no models, no workers, no bot)", flush=True)
elif not OMNI_ENABLED:
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
print("-" * 60, flush=True)
print(f"   🎛️ GPU owner   → {GPU_OWNER}", flush=True)
if not V1_GPU:
    # Named, with the reason and the way back. A worker that silently does not
    # start is indistinguishable from one that started and did nothing.
    print("      model_manager.py is held back so the processing engine has",
          flush=True)
    print("      the whole card — the two planes were OOM-ing each other over",
          flush=True)
    print("      the same seven models. omni_engine.py still runs, serving the",
          flush=True)
    print("      Explorer with its models and workers off, because /omni is",
          flush=True)
    print("      linked from every page and needs no GPU to answer.", flush=True)
    print("      VIOS_GPU_OWNER=both restores the old behaviour.", flush=True)
print("=" * 60, flush=True)
print("", flush=True)

# Launch workers via Watchdog Threads
if V1_GPU:
    threading.Thread(target=run_with_watchdog, args=(["python", "-u", "model_manager.py"], "🤖 [ENGINE]", True), daemon=True).start()
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "ui_server.py"], "🖥️ [UI]", False), daemon=True).start()
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "frame_worker.py"], "🎞️ [CV-ENGINE]", False), daemon=True).start()
if OMNI_ENABLED and V1_GPU:
    threading.Thread(target=run_with_watchdog, args=(["python", "-u", "omni_engine.py"], "🔮 [OMNI]", True), daemon=True).start()
elif OMNI_ENABLED:
    # `is_engine=False`: the watchdog's engine path exists for a process whose
    # crashes are CUDA-context losses. In explorer mode this one holds no context
    # to lose, so it restarts like any other service.
    threading.Thread(target=run_with_watchdog, args=(["python", "-u", "omni_engine.py"], "🔮 [OMNI]", False), daemon=True).start()

try:
    # Keep main orchestrator alive indefinitely
    while True:
        time.sleep(100)
except KeyboardInterrupt:
    print("\n🛑 [SYSTEM] Manual Shutdown Initiated.")
