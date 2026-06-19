import os
import sys
import subprocess
from kaggle_secrets import UserSecretsClient

def run_cmd(cmd, hide_output=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ERROR executing command\n{result.stderr.strip()}")
    elif not hide_output and result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    return result.returncode == 0

def sync_to_github(commit_message):
    print(f"🚀 Initiating VIOS Git Sync...")
    try:
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("github_token")
        user = "crestog"
        email = "sahudevansh482@gmail.com"
    except Exception:
        print("❌ ERROR: Kaggle Secrets not found.")
        sys.exit(1)

    repo_url = f"https://{user}:{token}@github.com/crestog/VideoIntelligenceOS.git"
    os.chdir("/kaggle/working/vios_system")

    # Initialize repo if it doesn't exist
    if not os.path.exists(".git"):
        run_cmd("git init", hide_output=True)
        run_cmd("git checkout -b main", hide_output=True)
        run_cmd(f"git remote add origin {repo_url}", hide_output=True)
    else:
        run_cmd(f"git remote set-url origin {repo_url}", hide_output=True)

    run_cmd("git add .", hide_output=True)
    
    # Check if there are actually changes to commit
    status = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if not status.stdout.strip():
        print("✅ No new changes to commit.")
        return

    print(f"📝 Committing: '{commit_message}'")
    run_cmd(f'git config user.email "{email}"', hide_output=True)
    run_cmd(f'git config user.name "Devansh Sahu"', hide_output=True)
    run_cmd(f'git commit -m "{commit_message}"', hide_output=True)
    
    print("☁️ Pushing to crestog/VideoIntelligenceOS...")
    run_cmd("git push -u origin main", hide_output=True)
    print("✅ Sync Complete.")

if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) if len(sys.argv) >= 2 else "chore: repository update"
    sync_to_github(msg)
