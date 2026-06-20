import os
import sys
import subprocess
import threading

def stream_logs(pipe, prefix, is_engine=False):
    """Reads output from a background process and prints it live with zero latency."""
    for line in iter(pipe.readline, ''):
        # Professional Logging Filter
        if is_engine and ("Loading weights:" in line or "%|" in line):
            continue
            
        # Force immediate screen render
        print(f"{prefix} {line}", end="", flush=True)

print("🗄️ [SYSTEM] Booting Message Broker...", flush=True)
os.system("service redis-server start > /dev/null 2>&1")

print("\n🚀 IGNITING VIDEO INTELLIGENCE OS...\n" + "="*50, flush=True)

p_model = subprocess.Popen(
    ["python", "-u", "model_manager.py"], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True,
    bufsize=1 # Line buffered
)
threading.Thread(target=stream_logs, args=(p_model.stdout, "🤖 [ENGINE]", True), daemon=True).start()

p_ui = subprocess.Popen(
    ["python", "-u", "ui_server.py"], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True,
    bufsize=1 # Line buffered
)
threading.Thread(target=stream_logs, args=(p_ui.stdout, "🖥️ [UI]", False), daemon=True).start()

try:
    p_ui.wait()
except KeyboardInterrupt:
    print("\n🛑 [SYSTEM] Shutting down OS...", flush=True)
    p_ui.terminate()
    p_model.terminate()
