import os, sys, subprocess, threading

def stream_logs(pipe, prefix):
    for line in iter(pipe.readline, ''):
        sys.stdout.write(f"{prefix} {line}")
        sys.stdout.flush()

print("🗄️ [SYSTEM] Booting Message Broker...")
os.system("service redis-server start > /dev/null 2>&1")

print("\n🚀 IGNITING VIDEO INTELLIGENCE OS...\n" + "="*50)

p_model = subprocess.Popen(["python", "-u", "model_manager.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
threading.Thread(target=stream_logs, args=(p_model.stdout, "🤖 [ENGINE]"), daemon=True).start()

p_ui = subprocess.Popen(["python", "-u", "ui_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
threading.Thread(target=stream_logs, args=(p_ui.stdout, "🖥️ [UI]"), daemon=True).start()

try:
    p_ui.wait()
except KeyboardInterrupt:
    print("\n🛑 [SYSTEM] Shutting down OS...")
    p_ui.terminate()
    p_model.terminate()
