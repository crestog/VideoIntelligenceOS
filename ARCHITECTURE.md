# VIOS V18 — System Architecture

**Goal:** every second of every video findable in natural language (moment search),
plus full knowledge extraction (what each video means, teaches, and how it works) —
on 2×T4 / 15 GB RAM / 57 GB disk Kaggle sessions, with Telegram as permanent storage.

---

## 1. The One Rule That Shapes Everything

> **Videos are immutable and live in Telegram forever, keyed by message ID.
> The database is the only mutable artifact — so the database itself is
> versioned INTO the same Telegram channel.**

Process once → snapshot the DB to Telegram → any future session imports the
snapshot and is instantly searchable. Nothing is ever reprocessed.

```
┌─────────────────────── TELEGRAM CHANNEL ───────────────────────┐
│  video #1 … video #N          (immutable, msg_id = primary key)│
│  #VIOS_SNAPSHOT part 1..k     (compressed DB, ≤1.9 GB/part)    │
│  #VIOS_SNAPSHOT_MANIFEST      (pointer msg, PINNED)            │
└────────────────────────────────────────────────────────────────┘
          ▲ export (snapshot_manager.py)        │ import on boot
          │                                     ▼
┌──────────────────────── KAGGLE SESSION ────────────────────────┐
│                          lake.db (SQLite WAL)                  │
│  posts / videos / frame_notes / transcripts / moments_search   │
└────────────────────────────────────────────────────────────────┘
```

## 2. Process Topology (boot.py watchdog)

| Process | Hardware | Role |
|---|---|---|
| `ui_server.py` | CPU | FastAPI + Ghost Worker (Telegram download, ffmpeg remux), cloudflared tunnel |
| `frame_worker.py` | CPU (all cores) | ffmpeg dual-tier frame extraction |
| `model_manager.py` | **GPU 0 + GPU 1** | model warm-up, then QUEUE_ANALYZE consumer |

GPU split: **GPU 0** = vision (YOLO11x, DINOv2, SigLIP2, CLIP, RAFT, EasyOCR),
**GPU 1** = Whisper-Large-v3 (audio/reasoning lane). Both run simultaneously —
audio transcription of video *N* overlaps frame analysis of video *N*.

## 3. Queue Dataflow (Redis, AOF)

```
Ghost Worker ──▶ QUEUE_VISION ──▶ frame_worker ──▶ QUEUE_ANALYZE ──▶ model_manager
   (download)     (2-lane:           (ffmpeg,          (GPU jobs)       (Whisper+YOLO+OCR
                   priority/default)  2 tiers)                           → DB rows)
```

* **Atomic claims (v3):** `BRPOPLPUSH queue → PROCESSING` — one Redis op, so a
  crash can never lose a job (v2 used two ops with a fatal window between them).
* Retry ×3 → dead-letter queue, replayable from the admin panel.
* Boot recovery re-queues orphaned PROCESSING jobs for **both** queues.
* Dedup: `PROCESSED_VIDEOS_SET` is **rebuilt from the DB** at every boot —
  Redis is ephemeral, the DB is truth. This is what makes "never reprocess" real.

## 4. Frame Storage: Two Tiers, One Decode

`frame_worker` runs **one ffmpeg pass** with a split filter:

| Tier | Path | Size/frame | Use |
|---|---|---|---|
| FULL | `frames_<id>/frame_%05d_ts_%.3fs.jpg` | ~80–200 KB | zoom/inspection, AI models |
| PREVIEW | `frames_<id>/.preview/frame_%05d.jpg` (320 px) | ~5–10 KB | instant workstation loading |

ffmpeg decodes once and encodes both tiers on all cores — 4–8× faster than the
old per-frame OpenCV `imwrite` loop, and the preview tier is what makes the
2-second workstation guarantee possible.

## 5. Workstation Loading — ≤2 s to Interactive at Any Length

```
t=0ms    GET /api/workspace/{id}         frame index + exact fps (from DB)
t≈150ms  GET /api/frame/{id}/0?tier=preview   FIRST PAINT (one round trip)
t≈300ms  GET /api/batch?step=N&tier=preview   LADDER: ~120 frames spanning the
                                              whole clip ≈ 1 MB → scrub ANYWHERE
t≈1-2s   fill preview tier, playhead-nearest chunks first (6 parallel streams)
bg       full-res tier streams in and silently replaces previews
```

* Batch wire format is self-describing: `[u32 idx][u32 size][jpeg]…` so sparse
  batches carry their positions.
* Scrubbing to an unloaded frame draws the **nearest loaded** frame — never a
  black screen; the exact frame swaps in when its chunk lands.
* Backend keeps an mtime-invalidated **frame-list cache** (the old code re-globbed
  the whole folder on every batch request).
* Immutable cache headers — a revisited video loads from browser cache at RAM speed.

## 6. Kaggle ↔ Browser Network Path

* cloudflared tunnel is HTTP/2: **one TCP connection, 6 multiplexed streams**
  (raising parallel fetches costs nothing; head-of-line blocking is per-stream).
* JPEG payloads are pre-compressed — no gzip middleware on the binary path.
* Preview tier cuts transported bytes ~15× for the interactive phase.
* WebSocket `/ws` pushes status + queue metrics; extraction progress is polled
  at 2.5 s only while a workspace is open.

## 7. The Intelligence Layer (Output 1 + 2)

**Stage A — mechanical extraction (implemented):**
`transcripts` (Whisper segments with start/end), `frame_notes` (YOLO objects +
OCR text + generated description per sampled frame @2/s), `moments_search`
(FTS5 over speech + visual content, keyed by `(msg_id, ts_sec)`).
→ Moment search v1: FTS query → (video, timestamp) → workstation opens at that exact frame.

**Stage B — semantic index (models already warm):**
SigLIP2/CLIP embeddings per sampled frame + per transcript segment, stored as
BLOBs in the DB (fits the snapshot cycle unchanged). Natural-language query →
embed → cosine top-k → exact millisecond. DINOv2 for visual-similarity
("videos that look like this"), RAFT for motion/pacing metrics.

**Stage C — knowledge extraction:**
An LLM pass over each video's bundle (transcript + notes + OCR) producing the
structured analysis JSON: topic, intent, audience, hook type/analysis,
retention mechanics, pattern interrupts (timestamped), ending type, CTA text,
narrative structure, emotional arc, persuasion mechanisms, editing style.
Stored per-video in an `analysis` table; each field FTS-indexed.
New models later = new tables + re-export snapshot. The cycle never changes.

## 8. Snapshot Cycle (snapshot_manager.py)

**Export:** `VACUUM INTO` (transactional, WAL-safe, no downtime) → tar with
`.thumbnails/` → zstd → split ≤1.9 GB → upload parts → upload + **pin** manifest.
**Import:** read pinned manifest (O(1); fallback: scan) → download parts →
sha256-verify → restore → **rebuild dedup set from `videos` table**.
**Boot:** if `lake.db` is missing, auto-import runs before workers ignite
(`VIOS_AUTO_IMPORT=0` to disable). UI buttons + status live in the Database tab.

## 9. Database Tab (v17_ui.html)

Overview cards (posts / harvested / extracted / frames / notes / transcript
segments / DB size) · read-only browser over **every** table with search +
pagination · snapshot Export/Import buttons with live status.

## 10. File Map

| File | Role |
|---|---|
| `boot.py` | watchdog, Redis boot, snapshot auto-import, dedup rebuild |
| `ui_server.py` | FastAPI shell, Ghost Worker, WebSocket, tunnel |
| `frame_worker.py` | ffmpeg dual-tier extraction → QUEUE_ANALYZE |
| `model_manager.py` | GPU warm-up + analysis worker (transcripts, frame notes, FTS) |
| `snapshot_manager.py` | Telegram DB export/import + dedup rebuild (CLI + API) |
| `v17_backend.py` | workspace/frame/batch/notes/db/snapshot endpoints |
| `v17_ui.html` | Library · Workstation · Database (three views, one file) |
| `queue_manager.py` | atomic reliable queues, DLQ, metrics |
| `config.py` | single source of truth (paths, tiers, creds, queues) |
