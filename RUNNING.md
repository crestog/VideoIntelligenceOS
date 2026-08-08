# Running VIOS

One server, one URL, two engines. This is the whole operating manual.

---

## 1. The Kaggle cell

Make a new notebook. **Settings → Accelerator → GPU T4 x2**, and **Internet → On**.
Both matter: no internet means no Telegram and no model weights, and the
processing plane plans its cohorts against the VRAM it actually finds.

**Before the first run, store your credentials once.** Add-ons → Secrets, one
row per value, using these exact names:

| Secret name | Where it comes from |
|---|---|
| `VIOS_BOT_TOKEN` | @BotFather |
| `VIOS_CHANNEL_ID` | your channel, the `-100…` form |
| `VIOS_API_ID` | my.telegram.org |
| `VIOS_API_HASH` | my.telegram.org |
| `VIOS_HF_TOKEN` | Hugging Face — optional, only the diarisation pass needs it |

Secrets are attached to your Kaggle account, not to a notebook or a session, so
this is genuinely once: every future session of every future notebook has them
until you delete the row. **Do not paste any of these into the cell.** `boot.py`
reads them itself, in Phase 0, and hands them to every process it starts — the
harvester, the upload bot, the workers and Atlas all see them without anything
being typed into a notebook that may end up shared or public.

Then the whole thing is one cell:

```python
!git clone -b atlas https://github.com/crestog/VideoIntelligenceOS.git
%cd VideoIntelligenceOS
!bash setup.sh          # apt packages, python deps, cloudflared. ~6-10 min, once per session.
!python boot.py         # starts Redis, the workers, and the web server
```

The first lines of the boot log tell you whether the secrets arrived:

```
🔑 [SYSTEM] Phase 0: Reading credentials...
   ✅ Kaggle Secrets → environment: VIOS_API_HASH, VIOS_API_ID, VIOS_BOT_TOKEN, VIOS_CHANNEL_ID
```

Names only — a value is never printed. If a row is missing you get
`⚠️ [SECRETS] Telegram disabled — not set: …` further down, naming exactly which
ones, and everything that does not need Telegram still runs.

`setup.sh` installs Neo4j and Postgres for the older omniscient layer, which is
about half the install time. If you only want the v2 planes (capture, process
and Atlas), skip them:

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
| `/atlas` | **v2** — the reader. Search the evidence, find the moment, play it. |
| `/v17`, `/admin`, `/omni` | the older standalone pages |
| `/docs` | every endpoint this server exposes, with a form to call it |

Every one of those pages ends with the same footer listing all the others, so
none of them is a dead end and none has to be remembered. The list lives in
`sitemap.py` and is served, not pasted, which is why adding a page adds it
everywhere at once.

Capture and Process are separate *pages* rather than in-page tabs on purpose:
each is a long-running engine that polls its own status, and making them tabs
would keep both running in the background of every other tab. They are linked
from the top bar on `/`, and from each other's sidebar.

---

## 3. The order to run things

The two engines are a pipeline. The second one has nothing to do until the
first one has put reels in the channel.

**First — `/capture`.** Open Setup. If your Kaggle Secrets are in place the four
credential fields are already satisfied and the page says where each one came
from — you only paste something here to override it for one session. Upload your
Instagram export (the raw ZIP, or the `.md` list). Run the readiness check.
Press Start.

On the **safe** profile it downloads roughly **one reel every two minutes** and
does not go faster. That pace is the entire anti-detection strategy — a
residential-looking request rate, jittered, with long idle gaps and breaks. Five
thousand reels is about eight days of wall clock. You are meant to start it and
walk away; it survives being stopped and restarted, and it never re-downloads a
reel it has already uploaded, even months later. That memory is
`capture_ledger.db`.

On the **fast** profile the floor drops from 25 s to 2 s, breaks and quiet hours
switch off, and the same run takes hours instead of days. It will get the
account blocked — the request rate sits near the band where 403s start, and that
is the accepted trade, not a side effect. Use it with a throwaway login. Nothing
is lost when the ban lands: every reel already captured is in the channel and in
the ledger, so the cost is one login and never any data. You can switch profile
mid-run; the pacer re-aims immediately.

**Then — `/process`.** Same credentials, same story — stored secrets are already
there. Optionally a Hugging Face token for the diarisation pass. Pick your
passes. Readiness check. Start.

**Then — `/atlas`.** It reads what the other two wrote. See section 3b.

You can run capture on one account and process on another at the same time.
They only meet through the Telegram channel.

---

## 3b. Atlas, and what it is reading

Atlas is a reader. It has no harvester, no worker and no queue, it never writes
to the channel, and running it twice is the same as running it once. It boots
with the rest of the stack and lives at `/atlas`.

What it reads is **two different things from the same channel**, which is worth
knowing because they arrive from different places and behave differently:

- **Bundles** — the harvester's snapshots of the capture ledger. A manifest
  naming the parts of a compressed SQLite file, pinned. Each one is a complete
  picture, so a later bundle overwrites the rows it shares.
- **Evidence shards** — `vios-evidence-*.jsonl.gz`, one per batch of claims,
  written by every `/process` account you are running. Nothing is pinned, there
  is no manifest, and they are *additive*: a shard is never a snapshot, it is
  the new findings since the last one.

Atlas is told the shape of neither. It infers each table's columns, their types
and their key from the rows themselves, which is what lets a pass you add
upstream become searchable here with no code change on this side. Embeddings and
thumbnails are dropped on the way in — they are the largest thing in the file and
the only part Atlas cannot use, because the vectors it searches with are its own.

Press **Scan** on the Sources tab to pull everything in the channel, or leave it:
it scans on boot. The tab lists every bundle and shard it imported and what each
one contained, which is where you look when a video is in the channel but not in
the search results.

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

**Step 7 — can you find the moment again?**
`/atlas` → Sources → **Scan**. It should list the shards `/process` published,
each with what it held. Then go to Search and type a sentence that describes
something you know is in one of those reels — not a keyword, a sentence. A
result is a *video* with a ribbon under it: the reel's duration as a strip, with
the matching passages lit at their real timestamps. Click one of the lit blocks.
The player should open at that second.

That is the whole system proving itself end to end: Instagram → Telegram →
GPU passes → evidence shard → channel → Atlas → the exact second. If the ribbon
is there but empty, the claims arrived without timestamps; if the search finds
nothing, check the Sources tab first — an empty scan and a bad index look
identical from the search box.

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
- **Credentials are never written to disk by anything here.** Kaggle Secrets
  holds them, Kaggle hands them to the process, and `boot.py` puts them in the
  environment for the session. Nothing lands in the repo, the notebook or the
  output quota, and no page will print one back to you — the Setup tab reports
  where each value came from, never what it is. Revoking is deleting the row.
