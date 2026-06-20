import os
import sys
import subprocess
import threading

def stream_logs(pipe, prefix, is_engine=False):
    """Reads output from a background process and prints it live to the console, with filtering."""
    for line in iter(pipe.readline, ''):
        # Professional Logging Filter: Suppress Hugging Face tqdm progress bar spam
        if is_engine and ("Loading weights:" in line or "%|" in line):
            continue
            
        sys.stdout.write(f"{prefix} {line}")
        sys.stdout.flush()

print("🗄️ [SYSTEM] Booting Message Broker...")
os.system("service redis-server start > /dev/null 2>&1")

print("\n🚀 IGNITING VIDEO INTELLIGENCE OS...\n" + "="*50)

# 1. Launch the Autonomous Model Manager
p_model = subprocess.Popen(
    ["python", "-u", "model_manager.py"], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True
)
# Attach thread with filtering enabled (is_engine=True)
threading.Thread(target=stream_logs, args=(p_model.stdout, "🤖 [ENGINE]", True), daemon=True).start()

# 2. Launch the UI Server
p_ui = subprocess.Popen(
    ["python", "-u", "ui_server.py"], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True
)
# Attach thread without filtering
threading.Thread(target=stream_logs, args=(p_ui.stdout, "🖥️ [UI]", False), daemon=True).start()

try:
    p_ui.wait()
except KeyboardInterrupt:
    print("\n🛑 [SYSTEM] Shutting down OS...")
    p_ui.terminate()
    p_model.terminate()
