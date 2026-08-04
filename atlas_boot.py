"""
Atlas — one-command boot.

Run this and nothing else. It brings up the whole site: scans the Telegram
channel for database bundles, imports every one it finds, builds the moment
index, loads the encoder and serves the interface on a public URL.

    python atlas_boot.py

Credentials come from the environment, never from this file:

    VIOS_API_ID / TELEGRAM_API_ID        MTProto app id     (large parts)
    VIOS_API_HASH / TELEGRAM_API_HASH    MTProto app hash
    VIOS_BOT_TOKEN / TELEGRAM_BOT_TOKEN  bot token          (required)
    ATLAS_CHANNEL_ID                     optional; defaults to the VIOS channel

On Kaggle the port is not reachable from outside, so this asks for a tunnel and
prints the URL. Without one it still serves on localhost, which is useful when
Atlas runs on a machine you can reach directly.

Nothing here is destructive. Importing a bundle twice changes nothing, so this
is safe to re-run after every harvest.
"""

import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ══════════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════
def check_deps() -> list:
    """Report what is missing rather than dying on the first import.

    Atlas degrades honestly: without sentence-transformers it still searches
    lexically, without pyrogram it still imports small bundles over the Bot
    API. Only fastapi, uvicorn and requests are truly required, so those are
    the only ones that stop the boot.
    """
    hard, soft = [], []
    for mod, why, required in (
        ("fastapi", "the web server", True),
        ("uvicorn", "the web server", True),
        ("requests", "the Telegram Bot API", True),
        ("numpy", "vector search", False),
        ("pyrogram", "downloading parts larger than 20 MB", False),
        ("sentence_transformers", "semantic search", False),
    ):
        try:
            __import__(mod)
        except ImportError:
            (hard if required else soft).append(f"{mod} ({why})")
    if hard:
        print("Missing required packages:")
        for m in hard:
            print("  -", m)
        print("\n  pip install fastapi uvicorn requests\n")
        sys.exit(1)
    return soft


# ══════════════════════════════════════════════════════════════════════════
# TUNNEL
# ══════════════════════════════════════════════════════════════════════════
def open_tunnel(port: int) -> str:
    """Expose the port publicly. Tries pyngrok, then cloudflared.

    Kaggle blocks inbound connections, so without one of these the site is only
    reachable from inside the notebook. Returns "" when neither is available —
    the server still starts, because a failed tunnel is not a reason to have no
    Atlas.
    """
    try:
        from pyngrok import conf, ngrok
        token = (os.environ.get("NGROK_AUTH_TOKEN")
                 or os.environ.get("NGROK_TOKEN") or "")
        if token:
            conf.get_default().auth_token = token
        url = str(ngrok.connect(port, "http").public_url)
        if url.startswith("http://"):
            url = "https://" + url[7:]
        return url
    except Exception as e:
        print(f"  ngrok unavailable ({type(e).__name__}: {e})")

    import re
    import shutil
    import subprocess
    exe = shutil.which("cloudflared")
    if exe:
        try:
            proc = subprocess.Popen(
                [exe, "tunnel", "--url", f"http://127.0.0.1:{port}",
                 "--no-autoupdate"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            found = []

            def read():
                for line in proc.stdout:
                    m = re.search(r"https://[-\w.]+\.trycloudflare\.com", line)
                    if m and not found:
                        found.append(m.group(0))

            threading.Thread(target=read, daemon=True).start()
            for _ in range(60):
                if found:
                    return found[0]
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
        except Exception as e:
            print(f"  cloudflared failed ({type(e).__name__}: {e})")
    return ""


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Serve Atlas.")
    ap.add_argument("--port", type=int, default=0, help="listen port")
    ap.add_argument("--no-tunnel", action="store_true",
                    help="serve on localhost only")
    ap.add_argument("--no-scan", action="store_true",
                    help="serve what is already imported, do not touch the channel")
    args = ap.parse_args()

    # Read by server._boot(). Set before the import so it is in place by the
    # time anything reads config.
    if args.no_scan:
        os.environ["ATLAS_NO_SCAN"] = "1"

    soft = check_deps()

    from atlas import config, server

    port = args.port or config.PORT

    print("=" * 68)
    print("  ATLAS")
    print("=" * 68)
    print(f"  database   {config.DB_PATH}")
    print(f"  cache      {config.CACHE_DIR}")
    print(f"  channel    {config.CHANNEL_ID}")

    missing = config.missing_secrets()
    if missing:
        print(f"  credentials  MISSING: {', '.join(missing)}")
        print("               Atlas will serve what is already imported but")
        print("               cannot reach the channel. Set them and re-run.")
    else:
        print("  credentials  present"
              + ("" if config.API_ID and config.API_HASH
                 else "  (bot token only — parts over 20 MB will be skipped)"))
    for s in soft:
        print(f"  degraded   no {s}")

    url = ""
    if not args.no_tunnel:
        print("\n  opening a public tunnel…")
        url = open_tunnel(port)

    print("-" * 68)
    if url:
        print(f"  OPEN THIS →  {url}")
    else:
        print(f"  local only →  http://127.0.0.1:{port}")
        if not args.no_tunnel:
            print("  (no tunnel — `pip install pyngrok` and set NGROK_AUTH_TOKEN")
            print("   to get a public URL from Kaggle)")
    print("-" * 68)
    print("  The page works immediately. Scanning, importing and indexing")
    print("  report their progress in the status pill at the top right.")
    print("=" * 68, flush=True)

    try:
        server.serve(port=port)
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
