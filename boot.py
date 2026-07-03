"""
VIOS Boot Orchestrator — Master Control Plane

Phases:
  1. Pre-flight Sweep: Kill zombie processes from previous crashes
  2. Message Broker: Boot Redis in AOF persistence mode
  3. Crash Recovery: Requeue orphaned jobs stuck in PROCESSING set
  4. Ignition: Launch all worker processes with auto-healing watchdog threads
"""

import os
import sys
import subprocess
import threading
import time


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
print("🗄️ [SYSTEM] Phase 2: Booting Redis Message Broker (AOF mode)...", flush=True)
os.system("redis-server --daemonize yes --appendonly yes")
print("   ✅ Redis broker online.", flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 3: CRASH RECOVERY
# ══════════════════════════════════════════════════════════
print("🔄 [SYSTEM] Phase 3: Crash Recovery — checking for orphaned jobs...", flush=True)
try:
    from queue_manager import recover_processing_jobs, get_queue_metrics
    recovered = recover_processing_jobs("QUEUE_VISION")
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
print("", flush=True)
print("=" * 60, flush=True)
print("🚀 IGNITING VIDEO INTELLIGENCE OS", flush=True)
print("=" * 60, flush=True)
print("   🤖 [ENGINE]    → model_manager.py  (7 SOTA GPU models)", flush=True)
print("   🖥️ [UI]        → ui_server.py      (FastAPI + Ghost Worker)", flush=True)
print("   🎞️ [CV-ENGINE] → frame_worker.py   (OpenCV frame extraction)", flush=True)
print("=" * 60, flush=True)
print("", flush=True)

# Launch workers via Watchdog Threads
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "model_manager.py"], "🤖 [ENGINE]", True), daemon=True).start()
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "ui_server.py"], "🖥️ [UI]", False), daemon=True).start()
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "frame_worker.py"], "🎞️ [CV-ENGINE]", False), daemon=True).start()

try:
    # Keep main orchestrator alive indefinitely
    while True:
        time.sleep(100)
except KeyboardInterrupt:
    print("\n🛑 [SYSTEM] Manual Shutdown Initiated.")
