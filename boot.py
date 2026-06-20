import os
import sys
import subprocess
import threading
import time

def stream_logs(pipe, prefix, is_engine=False):
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
        
        # If it reaches here, the process crashed.
        print(f"\n⚠️ [WATCHDOG] {prefix} process crashed. Rebooting in 3 seconds...", flush=True)
        time.sleep(3)

print("🧹 [SYSTEM] Sweeping Zombie Processes...", flush=True)
os.system("pkill -9 ffmpeg > /dev/null 2>&1 || true")
os.system("pkill -9 ffprobe > /dev/null 2>&1 || true")
os.system("pkill -9 cloudflared > /dev/null 2>&1 || true")

print("🗄️ [SYSTEM] Booting Message Broker...", flush=True)
os.system("redis-server --daemonize yes --appendonly yes")

print("\n🚀 IGNITING VIDEO INTELLIGENCE OS...\n" + "="*50, flush=True)

# Launch workers via Watchdog Threads
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "model_manager.py"], "🤖 [ENGINE]", True), daemon=True).start()
threading.Thread(target=run_with_watchdog, args=(["python", "-u", "ui_server.py"], "🖥️ [UI]", False), daemon=True).start()

try:
    # Keep main orchestrator alive indefinitely
    while True:
        time.sleep(100)
except KeyboardInterrupt:
    print("\n🛑 [SYSTEM] Manual Shutdown Initiated.")
