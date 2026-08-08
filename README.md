# 🧠 Video Intelligence OS (Project Omega)

**Lead Architect:** Devansh Sahu  
**Phase:** Core Infrastructure & Ingestion Complete  

Video Intelligence OS (VIOS) is a decoupled, multi-modal cloud operating system designed to autonomously harvest, standardize, and analyze massive volumes of video data to build a "Civilizational Index." 

Built specifically for ephemeral, resource-constrained cloud hardware (like Kaggle), VIOS utilizes an enterprise-grade microservice architecture to ensure zero-latency UI interactions, mathematical fault tolerance, and maximum hardware utilization.

---

## 🏗️ Core Architecture Overview

The system is divided into three completely independent execution layers, unified by a persistent Redis message broker.

### 1. The Watchdog Orchestrator (`boot.py`)
The master control plane. It never performs heavy computation; its only job is to keep the system alive.
* **Pre-flight Sweep:** Automatically purges zombie FFmpeg/Cloudflare processes from previous crashes to protect system RAM.
* **Redis Initialization:** Boots Redis in-memory (Kaggle sessions are ephemeral).
* **Auto-Healing Threads:** Launches the UI Server and Model Engine in isolated threads. If either worker crashes, the Watchdog catches the failure and resurrects the process in 3 seconds.
* **Stream Filtering:** Captures standard output from all workers, aggressively filters out raw library noise (like HuggingFace progress bars), and renders a clean, unified, zero-latency console for the developer.

### 2. The Ingestion Node (`ui_server.py`)
The high-speed data harvester and user interface, built on `FastAPI` and `Pyrogram`.
* **The Ghost Worker:** Autonomously scans the configured Telegram channel (`global_index_v`), fetching missing videos backwards through time to maintain a perfect local copy.
* **Metadata Extraction:** Uses Regex heuristics to parse structured captions (Creator, Category, Likes) and writes them to the database.
* **Format Standardization:** Routes all downloaded media through a dynamic FFmpeg pipeline. If a video is web-compatible, it triggers an `Instant Remux` (`-c copy`). If not, it falls back to a lossless `CRF 17` conversion. 
* **The Command Deck UI:** A responsive HTML/JS web interface featuring an infinite-scroll FTS5 search index, live video playback, keyboard macros, and live cloud-status monitoring.

### 3. The Multi-Modal Engine (`model_manager.py`)
The AI compute layer. It warms up 7 state-of-the-art neural networks and anchors them into the GPU. 
* **Strict VRAM Stacking:** Models are loaded in a mathematically determined sequence (heaviest to lightest) to eliminate VRAM fragmentation and prevent CUDA OOM crashes.
* **The 7 Foundational Models:**
  1. **YOLO11x:** Object detection and spatial awareness.
  2. **Whisper-Large-v3:** High-fidelity audio transcription.
  3. **DINOv2-Large:** Deep visual feature extraction.
  4. **SigLIP2:** High-performance text-image semantic embedding.
  5. **CLIP:** Secondary semantic vectorization.
  6. **RAFT-Large:** Optical flow and motion physics tracking.
  7. **EasyOCR:** GPU-accelerated text-in-video extraction.

---

## 🛡️ Enterprise Engineering Standards Implemented

To survive the hostile environment of cloud scraping and ephemeral servers, this OS features multiple layers of advanced fault protection:

### Network & Scraping Resilience
* **Exponential Backoff:** The Pyrogram client actively listens for Telegram `FloodWait` exceptions and `Broken Pipe` socket crashes, automatically pacing its downloads and sleeping to avoid permanent server bans.

### Compute & Hardware Optimization
* **FFmpeg Semaphores:** Video conversion is bottlenecked by a global `asyncio.Semaphore(2)`, mathematically guaranteeing that the OS will never spawn more than two concurrent FFmpeg processes, protecting the CPU from exhaustion.
* **Disk Space Watchdog:** The worker continuously monitors the `videos/` directory. If total system storage drops below 5GB, the OS pauses all scraping to prevent catastrophic SQLite corruption.

### Latency & Data Flow
* **Asynchronous Database Pipeline:** The SQLite database utilizes `aiosqlite` and `WAL` (Write-Ahead Logging) mode, allowing the Ghost Worker to perform rapid, continuous inserts without freezing the FastAPI web loop.
* **Idempotent Queues:** Before pushing a downloaded video to the Redis AI processing queue, the OS checks a high-speed Redis Set (`SADD`). This acts as an absolute gatekeeper, ensuring the GPUs never waste time processing duplicate media.
* **Cursor Pagination:** The UI explore tab utilizes Offset Pagination (`LIMIT 30 OFFSET X`), reducing API payload sizes by 97% and providing instantaneous front-end rendering.

---

## 💾 Durability: how the database survives a session

Every tier this OS runs on is disposable. `/kaggle/temp` is wiped between
sessions, and PostgreSQL starts from the Debian default data directory on that
same ephemeral disk — so the Omniscient store, and with it every Qwen narrative
the GPU produced, does not survive on its own. The `/kaggle/working` OUTPUT tier
(~19.5 GB) persists while the notebook exists, but goes when the notebook is
deleted. **The Telegram channel is the only storage in this system that outlives
the machine**, so it doubles as the backup target.

| Tier | Path | Lifetime | Holds |
|---|---|---|---|
| Scratch | `/kaggle/temp/vios_scratch` | the session | models, videos, frames, Qdrant, Neo4j |
| Output | `/kaggle/working/Insta-Vault` | the notebook | `lake.db`, bot session, bundles |
| Channel | Telegram | indefinite | reels, and the database bundles |

### Export → bundle → channel (`db_export.py`)

A bundle is `index.sqlite.zst` (a `VACUUM INTO` snapshot of the harvest DB) plus
`omnidb.sql.zst` (`pg_dump` of frames, chunks and narratives), zstd-compressed
and split into ≤18 MB parts. Qdrant vectors and the Neo4j graph are deliberately
omitted: both are derived from the frames and narratives, and rebuilding beats
replicating gigabytes.

The invariant that makes it safe: **a bundle exists if and only if its manifest
message is posted.** Parts upload first and are inert on their own, so a run that
dies halfway leaves unreferenced parts rather than a corrupt bundle that restore
might believe. The manifest carries every part's `message_id`, `file_id` and
SHA-256, so the channel is self-describing given only a bot token — no external
metadata store.

**Why 18 MB parts, and why HTTP instead of pyrogram.** Transfers go through the
Bot API (`tg_transport.py`), not MTProto. MTProto allows 2 GB files, but its
`send_document` has no timeout anywhere in its path: a stalled upload blocks
forever while the job still reports `running`, which is exactly how an export
came to sit at "72%" indefinitely. The Bot API gives a deadline on every request,
bounded retries, and byte-level progress — at the cost of a 50 MB upload cap and,
the binding one, a **20 MB cap on `getFile`**. A part that cannot be downloaded
is not a backup, so the part size is set by the download limit, not the upload
one. Bots also cannot look up an arbitrary message id, hence the `file_id` in the
manifest.

Progress is real: bytes sent against bytes total, a measured rate, an ETA, and a
stall counter that starts at the last byte rather than the last part. Past 45 s
of no movement the panel says so. **Cancel** stops between parts and keeps the
finished ones, so the next run resumes — the manifest is still posted only at the
end, so a resumed bundle is as valid as an uninterrupted one. **Test Connection**
answers in one second whether the token can reach the channel at all, which
otherwise took a whole failed export to discover.

### Restore → channel → database (`db_restore.py`)

Two steps, on purpose. **Check** reads only the manifest and reports how the
bundle's row counts compare to the local ones; **Restore** overwrites, and is
only reachable once that comparison exists. Restoring a 40-post bundle over a
900-post database is data loss, and the admin panel says so in those terms
before the confirm rather than after. A pre-restore snapshot lands on scratch, so
a regretted restore is recoverable for the rest of the session.

Two implementation details worth knowing before changing either file:

* **The fallback scan cannot use `get_chat_history`.** `messages.getHistory` is a
  user-only MTProto method and returns `BOT_METHOD_INVALID` for a bot account —
  the same asymmetry the harvester's scanner works around. Restore probes the
  newest id the way the harvester does and walks backwards with `getMessages`,
  which bots *are* allowed to call. The pinned manifest is the fast path; this
  runs when the bot lacks the rights to pin.
* **SQLite is replaced through the backup API, not by moving the file.** Every
  module opens `DB_PATH` per call and workers are mid-query at any moment; a file
  swap strands those handles on a deleted inode and orphans the WAL sidecars.
  `backup()` replaces pages inside the destination's own locking, so open
  connections either wait or see the new content — never a mixture.

Restore is **not** run automatically at boot. It would have to happen before
ignition, and `omni_engine` has not started PostgreSQL by then — a restore that
recovers the harvest DB while silently dropping the narratives is worse than
none. `boot.py` prints a hint when a fresh container has an empty database, and
the work happens in `/admin` against live services.

**What a restore will do, before it does it.** The plan lists every target with
its verdict — *replaced*, *rebuilt from what was replaced*, or *left alone* — so
Qdrant and Neo4j being absent from the bundle is stated rather than discovered.
After an apply, the outcome panel reports what actually loaded, the row counts
before and after, whether they match the manifest, where the pre-restore snapshot
went, and what to do next.

---

## ⏸️ One pause, and one reset

### Global pause (`system_control.py`)

Pause used to be three unrelated half-measures: `CV_PAUSED` was only read in the
queue-empty branch, so pressing Pause under load did nothing; `OMNI_PAUSED`
covered only the Omniscient engine; and the harvester had no flag at all, so
downloads kept filling the disk the pause was usually pressed to protect.

There is now one `VIOS_PAUSED` key, ORed with a per-component key so a single
stage can be stopped without stopping the rest. Every consumer checks it **before
`claim_job`**, not after — including inside the harvester's inner download loop,
which can run for hours without returning to the outer one. Components heartbeat
each cycle (`VIOS_HB:<component>`, 90 s TTL), so the panel draws *requested* and
*observed* state separately. That distinction matters: a dead worker also stops
claiming jobs, and used to be indistinguishable from an obedient one.

### Factory reset (`/admin` → Factory Reset)

Nine targets, each **measured in real bytes before the click** — on a Kaggle box
the only question is how much this frees. The re-derivable ones (media, vectors,
graph, queues, exports, session files) default on; PostgreSQL, `lake.db` and the
28 GB model cache are opt-in per click. The confirmation phrase `DELETE
EVERYTHING` is validated inside `start_reset`, not in the route, so it cannot be
skipped by calling the module directly.

Order is load-bearing:

* **Neo4j is stopped before its store is deleted.** A live store manager rewrites
  the files it was told to forget, and the restart then finds a half-deleted
  database it refuses to open.
* **Redis' `flushall` takes the pause flags with it,** so they are re-applied
  immediately after — the reset is not over.
* **Workers are killed, not reasoned with.** `boot.py`'s watchdog restarts them
  in 3 s against a machine that no longer holds what they had open. `ui_server`
  is deliberately *not* killed: it is the process running the reset.
* **That exception is why `lake_schema.py` exists.** `lake.db`'s schema was only
  ever created by `ui_server`'s `__main__`, so deleting the file left the
  harvester's next INSERT hitting `no such table: posts` until the container was
  rebooted. The reset re-applies the schema the moment the file is gone — the
  database comes back *empty*, not *missing*. Every other target self-heals when
  the watchdog restarts its owner.

**The Telegram channel is never touched.** Reels and uploaded bundles stay
exactly where they are; the panel says so on screen next to the button, because
a reset with no bundle anywhere is permanent and that belongs before the click.

---

## 🚀 The Kaggle Launchpad (CI/CD)

The repository relies on Infrastructure as Code (IaC) via `setup.sh` and `requirements.txt`. The entire OS can be bootstrapped from a factory-reset Kaggle instance using this single cell:

```python
import os
from kaggle_secrets import UserSecretsClient

os.chdir("/kaggle/working")
WORKSPACE = "/kaggle/working/vios_system"

secrets = UserSecretsClient()
repo = f"https://crestog:{secrets.get_secret('github_token')}@[github.com/crestog/VideoIntelligenceOS.git](https://github.com/crestog/VideoIntelligenceOS.git)"
os.system(f"git clone {repo} {WORKSPACE} 2>/dev/null || git -C {WORKSPACE} pull {repo} main > /dev/null 2>&1")

try:
    os.environ["HF_TOKEN"] = secrets.get_secret("hf_token")
except: 
    pass

os.chdir(WORKSPACE)
print("⚙️ [SYSTEM] Provisioning Environment...")
os.system("bash setup.sh > setup_logs.txt 2>&1") 
os.system("python boot.py")
```

### Launching the Omniscient branch (`feature/omniscient-unified`)

The unified build adds the tri-partite database (PostgreSQL + Qdrant + Neo4j), the
Qwen2.5-VL Oracle, GraphRAG, and the God-Mode Explorer tab. It needs a longer
`setup.sh` run (Java 17, PostgreSQL, Neo4j) and two extra secrets. Use this cell:

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

BRANCH    = "feature/omniscient-unified"
WORKSPACE = "/kaggle/working/vios_system"
secrets   = UserSecretsClient()

# ── Credentials (all injected as env vars; nothing is hardcoded in the repo) ──
repo = f"https://crestog:{secrets.get_secret('github_token')}@github.com/crestog/VideoIntelligenceOS.git"

for env_key, secret_name in [
    ("HF_TOKEN",             "hf_token"),
    ("VIOS_NIM_API_KEY",     "nim_api_key"),
    # Telegram credentials are env-only — config.py has no fallback.
    ("VIOS_API_ID",          "TELEGRAM_API_ID"),
    ("VIOS_API_HASH",        "TELEGRAM_API_HASH"),
    ("VIOS_BOT_TOKEN",       "TELEGRAM_BOT_TOKEN"),
    # The channel id is an address rather than a key, but it is also env-only:
    # a default would point every clone of this public repo at one archive.
    ("VIOS_CHANNEL_ID",      "TELEGRAM_CHANNEL_ID"),
]:
    try:
        os.environ[env_key] = secrets.get_secret(secret_name)
    except Exception:
        print(f"⚠️  Secret '{secret_name}' not set — continuing without it.")

os.environ["VIOS_OMNI"] = "1"   # set to "0" to boot the classic VIOS stack only

# ── Clone or fast-forward the branch ──
os.chdir("/kaggle/working")
if os.path.isdir(f"{WORKSPACE}/.git"):
    subprocess.run(["git", "-C", WORKSPACE, "fetch", repo, BRANCH], check=True)
    subprocess.run(["git", "-C", WORKSPACE, "checkout", "-B", BRANCH, "FETCH_HEAD"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, repo, WORKSPACE], check=True)

os.chdir(WORKSPACE)
print(f"✅ [GIT] On {BRANCH} @ "
      f"{subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()}")

# ── Provision (Java 17 + PostgreSQL + Neo4j + pip). Takes ~8-12 min cold. ──
print("⚙️  [SYSTEM] Provisioning — tail setup_logs.txt if this stalls...")
os.system("bash setup.sh > setup_logs.txt 2>&1")

# ── Ignite: Redis → UI Server → Model Engine → Omniscient Engine ──
os.system("python boot.py")
```

**What to expect in the console:** the watchdog boots Redis, then four auto-healing
workers. The Omniscient engine reports each backend as it comes up
(`✅ PostgreSQL`, `✅ Qdrant`, `✅ Neo4j`) and degrades gracefully if one fails —
a missing backend disables only the features that depend on it. Once cloudflared
prints the tunnel URL, the **Omniscient** tab in the Command Deck serves the
God-Mode Explorer, and the Telegram bot accepts video uploads on the priority lane.

**Required Kaggle Secrets:** `github_token` (mandatory), `TELEGRAM_API_ID` /
`TELEGRAM_API_HASH` / `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHANNEL_ID` (needed for the
upload bot and channel harvesting — without them both stay disabled and everything
else runs normally), `hf_token` (optional — all models are currently public),
`nim_api_key` (optional — without it, GraphRAG extraction and answer synthesis are
skipped and raw visual output is returned).

> **Credentials are env-only.** `config.py` carries no fallback values. An earlier
> revision hardcoded a working bot token and API hash as defaults, which published
> them in this repository's history — treat any credential from before this commit
> as compromised and rotate it. `boot.py` prints which secrets are absent during
> preflight, so a forgotten export shows up as one clear line rather than two
> workers failing deep inside pyrogram.

