import os
import subprocess
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
token = user_secrets.get_secret("github_token")
user = user_secrets.get_secret("github_user")
email = user_secrets.get_secret("github_email")

repo_url = f"https://{user}:{token}@github.com/crestog/VideoIntelligenceOS.git"
os.chdir("/kaggle/working/vios_system")

commands = [
    "git init",
    f"git config user.name '{user}'",
    f"git config user.email '{email}'",
    "git checkout -b main",
    "git add .",
    "git commit -m 'chore: wipe remote repository clean'",
    f"git push --force {repo_url} main"
]

print("🚀 Executing native Git force-push...")
for cmd in commands:
    safe_cmd = cmd.replace(token, "*****")
    print(f"> {safe_cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.stdout: print(res.stdout.strip())
    if res.stderr: print(res.stderr.strip())

print("✅ GitHub repository successfully wiped and reset.")
