# Running VIOS

One server, one URL, two engines. This is the whole operating manual.

---

## 1. The Kaggle cell

Make a new notebook. **Settings → Accelerator → GPU T4 x2**, and **Internet → On**.
Both matter: no internet means no Telegram and no model weights, and the
processing plane plans its cohorts against the VRAM it actually finds.

```python
!git clone -b atlas https://github.com/crestog/VideoIntelligenceOS.git
%cd VideoIntelligenceOS
!bash setup.sh          # apt packages, python deps, cloudflared. ~6-10 min, once per session.
!python boot.py         # starts Redis, the workers, and the web server
```

`setup.sh` installs Neo4j and Postgres for the older omniscient layer, which is
about half the install time. If you only want the v2 planes (capture and
process), skip them:

```python
!VIOS_OMNI=0 bash setup.sh
```

Within a minute of `boot.py`, the log prints:

```
============================================================
🌍 WEB APP IS LIVE AT: https://something-random.trycloudflare.com
============================================================
```

Open that. **Leave the Kaggle cell running** — closing it kills the server.
The URL changes every session; that is a property of free cloudflare tunnels,
not a bug.

---

## 2. Yes, it is all one website

One FastAPI app on port 8000, one tunnel, one URL. Everything is a path on it:

| Path | What it is |
|---|---|
| `/` | Feed, Explore, V17 Workspace, Omniscient, Admin — the v1 tabs |
| `/capture` | **v2** — Instagram → Telegram. Downloads reels, uploads them to your channel. |
| `/process` | **v2** — the rotation engine. Turns those reels into the evidence database. |
| `/v17`, `/admin`, `/omni` | the older standalone pages |

Capture and Process are separate *pages* rather than in-page tabs on purpose:
each is a long-running engine that polls its own status, and making them tabs
would keep both running in the background of every other tab. They are linked
from the top bar on `/`, and from each other's sidebar.

---

## 3. The order to run things

The two engines are a pipeline. The second one has nothing to do until the
first one has put reels in the channel.

**First — `/capture`.** Open Setup, paste your bot token, channel id, API id
and API hash. Upload your Instagram export (the raw ZIP, or the `.md` list).
Run the readiness check. Press Start.

It downloads roughly **one reel every two minutes** and does not go faster.
That pace is the entire anti-detection strategy — a residential-looking request
rate, jittered, with long idle gaps. Five thousand reels is about a week of
wall-clock. You are meant to start it and walk away; it survives being stopped
and restarted, and it never re-downloads a reel it has already uploaded, even
months later. That memory is `capture_ledger.db`.

**Then — `/process`.** Same four credentials (this tab keeps its own copy;
the engines can run in different processes). Optionally a Hugging Face token
for the diarisation pass. Pick your passes. Readiness check. Start.

You can run capture on one account and process on another at the same time.
They only meet through the Telegram channel.

---

## 4. How the multiple Kaggle accounts work

This is simpler than it sounds, because **the accounts never talk to each
other.** There is no coordinator, no shared lock, no server in the middle.

Each video has a key. The engine hashes that key into a partition number. On
each account you set two fields in `/process` → Setup:

- **workers in total** — how many accounts you are using. Same value everywhere.
- **this one is number** — 0 on the first account, 1 on the second, 2 on the
  third, and so on. Different on every account.

An account only processes videos where `hash(key) % total == mine`. Ten
accounts with `total = 10` and `mine = 0..9` cover the library exactly once,
with no overlap and no gaps, and no account needs to know the others exist.

**The results merge through the channel.** Every account periodically exports
its new findings as a compressed shard and uploads it to the same Telegram
channel. Any session that has *replay evidence shards on start* ticked pulls
in every shard every other account has published. So after a few rounds, each
account's local database holds everyone's work.

Two consequences worth knowing:

- **The order does not matter.** Shards are append-only and carry the id range
  they cover, so replaying them twice is harmless and replaying them out of
  order is harmless.
- **You can change the count later.** Going from 3 accounts to 10 re-slices the
  work; already-finished videos stay finished, because coverage is recorded per
  (video, pass), not per account.
- **The slices are even, not perfectly equal.** Keys hash into 64 buckets, which
  are then dealt out to your accounts. With 10 accounts, four of them get 7
  buckets and six get 6 — about a 17% spread. Nothing overlaps and nothing is
  missed; some accounts just finish a little earlier. Counts that divide 64
  (2, 4, 8, 16) come out exactly equal.

If you only want to run one account for now, leave it at **total = 1, mine = 0**.
Everything works; it just takes ten times longer.

---

## 5. Testing it — what to actually check

You cannot meaningfully test this on a laptop. It wants two GPUs, ffmpeg, and
a live Telegram channel. Here is the fastest path to knowing whether it works,
in order, each step cheap and each one proving something real.

**Step 1 — does the server come up?**
Open the tunnel URL. If `/` renders, FastAPI, the templates and the tunnel are
all fine.

**Step 2 — do the v2 pages exist?**
Click **Capture** and **Process** in the top bar. If either shows a plain
"unavailable" or 404, the router did not import — check the boot log for
`capture tab unavailable:` or `process tab unavailable:` and read the reason.
The most likely one is a missing package.

**Step 3 — does Telegram answer?**
`/capture` → Setup → paste credentials → **Run readiness check**. The check
sends a message to your channel and deletes it. If it comes back green, your
bot token, channel id and permissions are all correct — that is a stronger
proof than any "connected" indicator, because posting is what the engine
actually needs to do.

**Step 4 — does one reel work end to end?**
In `/capture`, set the limit to **1** and start. In two minutes you should see
one reel land in your Telegram channel with its JSON metadata. If that works,
five thousand will work; it is the same loop.

**Step 5 — does a pass produce claims?**
`/process` → Setup → readiness check → Start. Watch the Run page. Within a
couple of minutes the **claims** counter should move off zero and the rotation
strip should show cohort 1 lit up. Then go to **Database**, open the video, and
read what the passes actually wrote. That is the real test: not that it ran,
but that what it wrote is worth having.

**Step 6 — does it survive being stopped?**
Stop the engine. Start it again. It should pick up where it left off, not at
the beginning. Then let it run.

If you want to shortcut steps 1–3: the readiness check on either tab is
designed to be the whole test. It checks ffmpeg, every Python package each
selected pass imports, free VRAM, free disk, Telegram write access, and whether
there is any work to do — and it tells you which failures are advisory and
which actually block the start.

---

## 6. Things that will surprise you

- **The session is 12 hours.** Kaggle stops you there. Both engines are built
  around it: they checkpoint continuously, and restarting resumes.
- **`/kaggle/working` is small.** Weights, proxies and scratch go to
  `/kaggle/temp`, which is wiped between sessions and is not the output quota.
  Nothing important lives on the notebook's disk — the channel is the storage.
- **A missing package costs you a pass, not the session.** The engine plans
  around what is actually installed and records the skipped pass with a reason.
- **A pass that will not fit in VRAM is dropped from the plan**, with the reason
  shown on the Passes page. It is not an error.
- **Credentials are never written to disk.** They live in the engine's memory
  for the life of the process. Restarting the notebook means pasting them again.
  That is deliberate.
