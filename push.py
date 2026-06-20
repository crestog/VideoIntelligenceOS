import os, sys, subprocess
from kaggle_secrets import UserSecretsClient
def run(cmd): subprocess.run(cmd, shell=True)
if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "chore: update"
    os.chdir("/kaggle/working/vios_system")
    run("git rm -r --cached archive/ *.db cloudflared 2>/dev/null || true") # Clean repo
    run("git add .")
    run('git config user.email "sahudevansh482@gmail.com"')
    run('git config user.name "Devansh Sahu"')
    run(f'git commit -m "{msg}"')
    run("git push origin main")
    print("✅ Pushed to GitHub.")
