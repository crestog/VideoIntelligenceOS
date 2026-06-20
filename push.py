import os
import sys
import subprocess
from kaggle_secrets import UserSecretsClient

def run_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)

def sync_to_github(commit_message):
    print("🚀 Initiating VIOS Git Sync...")
    try:
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("github_token")
    except:
        print("❌ ERROR: Secrets not loaded.")
        sys.exit(1)
        
    repo_url = f"https://crestog:{token}@github.com/crestog/VideoIntelligenceOS.git"
    os.chdir("/kaggle/working/vios_system")
    
    if not os.path.exists(".git"):
        run_cmd("git init")
        run_cmd("git checkout -b main")
        run_cmd(f"git remote add origin {repo_url}")
    else:
        run_cmd(f"git remote set-url origin {repo_url}")

    run_cmd("git add .")
    run_cmd('git config user.email "sahudevansh482@gmail.com"')
    run_cmd('git config user.name "Devansh Sahu"')
    run_cmd(f'git commit -m "{commit_message}"')
    run_cmd("git push -u origin main")
    print("✅ Sync Complete.")

if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) if len(sys.argv) >= 2 else "chore: ui integration update"
    sync_to_github(msg)
