# VIOS — Master Plan

### Research, architecture, gaps, and the road from "a Kaggle notebook that works" to "a system I'd bet a decade on"

**Status:** brainstorming document. No code changes were made to produce it.
**Date:** 2 August 2026
**Scope:** everything — vision, storage permanence, retrieval, knowledge organisation, UI/UX, infrastructure, repository structure.

---

## Table of contents

| # | Section |
|---|---|
| 0 | [How to read this document](#0-how-to-read-this-document) |
| 1 | [What you are actually building](#1-what-you-are-actually-building) |
| 2 | [The correction: Stage B, and why 3.2 GB is warm](#2-the-correction-stage-b-and-why-32-gb-is-warm) |
| 3 | [State of the union — an honest audit](#3-state-of-the-union--an-honest-audit) |
| 4 | [The research](#4-the-research--what-other-people-and-companies-actually-do) |
| 5 | [Gaps, bad logic, and could-be-betters](#5-gaps-bad-logic-and-could-be-betters) |
| 6 | [The target architecture](#6-the-target-architecture) |
| 7 | [Database permanence: making Kaggle disposable](#7-database-permanence-making-kaggle-disposable) |
| 8 | [Search that actually earns "100% accurate"](#8-search-that-actually-earns-100-accurate) |
| 9 | [Instant video playback](#9-instant-video-playback) |
| 10 | [The knowledge layer and the roadmap engine](#10-the-knowledge-layer-and-the-roadmap-engine) |
| 11 | [Infrastructure: enterprise-grade, single user, free](#11-infrastructure-enterprise-grade-single-user-free) |
| 12 | [The website — UI, UX, and user flow](#12-the-website--ui-ux-and-user-flow) |
| 13 | [Feature catalogue](#13-feature-catalogue) |
| 14 | [Out-of-the-box ideas from other domains](#14-out-of-the-box-ideas-from-other-domains) |
| 15 | [Repository reorganisation](#15-repository-reorganisation) |
| 16 | [Phased roadmap](#16-phased-roadmap) |
| 17 | [Risks, failure modes, and what I'd worry about](#17-risks-failure-modes-and-what-id-worry-about) |
| 18 | [Open questions for you](#18-open-questions-for-you) |
| 19 | [Sources](#19-sources) |

---

## 0. How to read this document

This is long on purpose. You asked for everything in one file, so it's here — but it's built so you can attack it in pieces.

- **Sections 1–3** are about *understanding*: what you're building, and where it actually stands today. Read these first; they set the vocabulary for everything after.
- **Section 4** is the research dump. Every claim has a source in §19. If you only read one part, read §4.1 (video RAG), §4.2 (hybrid retrieval), and §4.6 (free infra) — those three change the most decisions.
- **Sections 5–10** are the design. §5 is the critical one: it's the list of things that are wrong or weak right now, ranked.
- **Sections 11–15** are the build-out: infra, UI, features, repo shape.
- **Sections 16–18** are what to do about it.

Two ground rules I've applied throughout:

1. **Free means free forever, not free-for-12-months.** Anything with a trial clock or a credit card requirement is called out explicitly.
2. **One user is a superpower, not a limitation.** Almost every hard problem in "how do big companies do this" comes from multi-tenancy, concurrency, and hot-shard skew. You have none of those. That means you can use techniques that would be insane at scale — full-corpus rescoring, cross-encoder reranking on every query, keeping the whole index in RAM — and they'll be *faster* than what YouTube does, for your corpus.

---

## 1. What you are actually building

Let me restate your goal in my own words, because getting this precise changes every architectural decision downstream.

> **A personal, permanent, searchable memory of everything in a video collection — where "searchable" means finding the exact second, "understandable" means answering questions about the content, and "organised" means the extracted knowledge composes into learning paths.**

Broken into the four outputs you named:

### Output 1 — Retrieval
"Whenever I need to find any video from my collection, my tool should return 100% accurate search results."

This is **moment retrieval**, not video retrieval. The unit of the answer is `(video_id, timestamp_range)`, not `video_id`. That distinction is the single most important framing decision in the whole system, and the current codebase already half-gets it (`moments_search` is keyed by `(msg_id, ts_sec)`) and half-doesn't (the Omniscient search path returns chunks, and the CV path returns FTS rows, and they don't speak to each other).

### Output 2 — Comprehension
"Knowledge extraction and answering questions from videos."

This is **video RAG / video QA**. The answer is a synthesised statement plus *citations that are clickable timestamps*. The citation requirement is not cosmetic — it's the only way you'll ever trust the answers, and it's what makes Output 1 and Output 2 the same system rather than two systems.

### Output 3 — Organisation
"Organise that entire extracted knowledge so I can find anything, but more detailed — if I want to learn something, it gives me a roadmap based on what people shared in all the videos. Basically Google + roadmap.sh + courses, combined, from video knowledge."

This is the ambitious one and it's the one nobody has built. It's a **personal knowledge graph with prerequisite ordering**. §10 is entirely about this, and §4.4 explains why the academic literature has a hole exactly the shape of your project.

### Output 4 — Composability
"Integrate the database into any UI I desire."

This is an **API-first contract**. It means the database is not "the thing the website reads" — it's a product with a stable schema and a query interface, and the website is one of N clients. That reframes the whole storage question: the DB has to be portable, versioned, and independently loadable.

### The constraints that shape everything

| Constraint | Consequence |
|---|---|
| Compute is Kaggle (2×T4, 15 GB RAM, ~57 GB disk, ~12 h sessions) | Everything is **ephemeral by default**. Processing is a batch job you run when you feel like it, not a service. |
| Storage must be free and permanent | Telegram is the object store. Everything else is a cache or an index. |
| One user | No auth complexity, no rate-limit engineering, no multi-tenancy. But also: no CDN budget, no always-on GPU. |
| Free forever | Rules out Fly.io, Railway, most managed Postgres. Rules *in* Cloudflare, Oracle Always Free, GitHub, HF Spaces. |
| The website must feel enterprise-grade | The bar is Frame.io / Linear / Notion, not "a Flask dashboard with tables". |

### The mental model I'd hold

Think of the system as **four planes**, not four processes:

```
  CAPTURE PLANE   Telegram channel → the immutable video corpus. Grows forever.
                  ↓
  COMPUTE PLANE   Kaggle sessions → burn GPU, emit artifacts. Disposable, restartable,
                  idempotent. Nothing here is ever the source of truth.
                  ↓
  ARTIFACT PLANE  The index bundle: SQLite + vectors + graph + sprites + proxies.
                  Versioned, content-addressed, stored back in Telegram.
                  ↓
  SERVE PLANE     Cloudflare Worker + R2 + browser. Reads artifacts. Never writes.
                  Always on, always free, always fast.
```

The current system has the capture and compute planes. It has an artifact plane that dies with the session. It has no serve plane at all — the "website" is the compute plane's debug UI exposed through a tunnel. **That's the fundamental architectural gap, and most of this document is about closing it.**

---

## 2. The correction: Stage B, and why 3.2 GB is warm

You corrected me and you were right. I said `model_manager.py` warms 3.2 GB of models it never uses (SigLIP so400m, CLIP-L, DINOv2-large, RAFT-L) and framed that as "remove them." That framing was wrong.

The accurate finding is:

> **`ARCHITECTURE.md` §7 specifies three stages. Stage A (mechanical extraction) is implemented. Stage C (LLM knowledge extraction) exists in the Omniscient half. Stage B — the semantic frame/segment index — was specified, the models were warmed for it, and the code that consumes them was never written.**

Concretely, in `model_manager.py`, seven models get loaded:

```python
for model in ["yolo", "whisper", "dinov2", "siglip", "clip", "raft", "easyocr"]:
```

and exactly three are ever read back out of `WARM_MODELS`:

```
183:  whisper = WARM_MODELS.get('whisper_model')   # analyze_audio
219:  yolo    = WARM_MODELS.get('yolo_model')      # analyze_frames
220:  ocr     = WARM_MODELS.get('ocr_reader')      # analyze_frames
```

So the models are correct and necessary — they're the right models for exactly what you described (embedding every sampled frame and every chunk so that natural-language queries land on a millisecond). What's missing is the ~150 lines that:

1. batch the extracted frames through SigLIP + CLIP,
2. batch the transcript segments through a text encoder,
3. write the vectors somewhere durable,
4. and expose a cosine top-k endpoint.

Three further notes that matter for the plan:

- **The Omniscient half already does this**, on GPU 0, into embedded Qdrant. So you have *two* half-built semantic indexes: one with warm models and no code, one with code and a database that dies with the session. §6 merges them.
- **DINOv2 and RAFT serve different jobs than SigLIP/CLIP.** DINOv2 is self-supervised and gives you *visual similarity* ("more videos that look like this frame") without any text alignment — it's much better at "same scene/style/person" than CLIP is. RAFT gives you motion magnitude, which is the cheapest possible proxy for pacing, cuts, and "is this a talking head or an action sequence". Both are legitimately valuable and neither is replaceable by SigLIP.
- **The duplication is real and worth fixing** — `model_manager.py` and `omni_models.py` each load their own SigLIP so400m and CLIP-L, on the same GPU, in separate processes. That's ~1.6 GB of VRAM burned on identical weights, and it contributes directly to the GPU-1 OOM you saw. §5 covers this.

**Verdict:** don't delete anything. Build Stage B, and make one process own each model.

---

## 3. State of the union — an honest audit

### What works, and works well

| Thing | Assessment |
|---|---|
| **The immutability rule** (`ARCHITECTURE.md` §1) | This is the best decision in the whole project. "Videos live in Telegram forever, keyed by message ID; the DB is a rebuildable index" is exactly the right invariant, and it's what makes everything in §7 possible. |
| **`queue_manager.py` v3** | Genuinely good. `BRPOPLPUSH` atomic claims, DLQ, poison-pill quarantine via `max_recoveries`, crash recovery on boot. This is production-shaped code. The FIFO orientation (LPUSH producer / RPOP consumer) is correct and the retry-to-back-of-queue behaviour in `fail_job` is a real (if accidental) design win — a job that OOM'd retries after memory pressure has passed. |
| **Two-tier frame extraction** | One ffmpeg pass with a split filter emitting full-res + 320 px preview is the right call, and it's the reason the workstation can be interactive in ~2 s. |
| **The batch wire format** | `[u32 idx][u32 size][jpeg]…` self-describing sparse batches — this is a small thing done properly. |
| **`config.py` as single source of truth** | Every entrypoint imports it; scratch/output disk split is explicit; the `PYTORCH_CUDA_ALLOC_CONF` prohibition is documented *with the reason*. Good. |
| **The dtype capability gate** | `_gpu_dtype()` picking fp16 on Turing because T4s have no bf16 tensor cores and bitsandbytes selects its 4-bit kernel path off compute dtype — that's a real, non-obvious bug class, correctly handled. |
| **Fatal-error policy for lost CUDA contexts** | Treating illegal-memory-access as fatal and `os._exit(1)` so the watchdog rebuilds the context, while excluding OOM from that path, is the right distinction. Most people get this wrong. |

### What's broken or missing, at a glance

| # | Issue | Severity |
|---|---|---|
| G1 | **No permanence.** The DB dies with the session. | Existential |
| G2 | **No serve plane.** The website is the compute plane's debug UI behind a tunnel. | Existential |
| G3 | **Two disconnected intelligence pipelines** writing to two disconnected stores. | Critical |
| G4 | **Stage B unbuilt** — no semantic frame/segment index in the CV half. | Critical |
| G5 | **GPU-1 co-tenancy OOM** — Whisper + Qwen on one 15 GB card across two processes. | High |
| G6 | **Duplicate model loads** (SigLIP, CLIP, RAFT, EasyOCR loaded twice). | High |
| G7 | **No evaluation harness.** "100% accurate" is unmeasured. | High |
| G8 | **No reranking, no fusion.** Retrieval is single-stage. | High |
| G9 | **No video proxies or sprites** — playback of a search result means downloading the original. | High |
| G10 | **Repository is flat** — 37 tracked files at root, including dead ones. | Medium |
| G11 | **Neo4j + Postgres + Qdrant + Redis + SQLite** — five datastores for one user. | Medium |
| G12 | **No schema versioning or migrations.** | Medium |
| G13 | **Dead code**: `admin_backend.py`, `admin_ui.html`, `main_ui.html`, `local_preview.py`, `push.py`, `test_harness.py`, `public/placeholder-*`. | Low |

Sections 5 and 6 go through these properly.

### The live defect, precisely

From your last Kaggle log:

```
[22:58:15] ❌ ANALYZE #1218 FAILED → RETRIED | CUDA failed with error out of memory
[22:58:18] ❌ ANALYZE #1216 FAILED → RETRIED | parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal
```

Both are CTranslate2 (faster-whisper's backend) — `CUDA failed with error …` is CTranslate2's own error format, and `parallel_for failed:` is the Thrust layer beneath it. The mechanism:

- `model_manager.py` puts Whisper-large-v3 fp16 on `cuda:1` (~3.1 GB of weights).
- `omni_models.py` puts Qwen2.5-VL-7B NF4 on `cuda:1` (~5.5 GB of weights).
- That's ~8.6 GB of static weights on a 15 GB T4, leaving ~6.4 GB to be shared between Whisper's encoder activation buffers and Qwen's video-generation spike.
- **They are in separate processes.** Each has its own PyTorch/CTranslate2 caching allocator. Freed blocks in one are *not* visible to the other. So the effective headroom is worse than the arithmetic suggests — each allocator hoards.

It's transient (job `#1214` succeeded three seconds later), and the retry-to-back-of-queue behaviour means it self-heals. But it's pure waste, and the fix in §6 (one GPU-owning process) removes it structurally rather than tuning around it.

I'm confident about the OOM. I'm less certain why the second failure surfaces specifically as `invalid device ordinal` rather than another OOM — my best read is a CTranslate2 context in a degraded state after an allocation failure, but I haven't proven that.

Also outstanding: `⚠️ GraphRAG NIM extraction skipped: NIM API not configured` — `VIOS_NIM_API_KEY` is unset, so the graph half of the knowledge layer is silently doing nothing. §10 proposes replacing that dependency entirely.

---

## 4. The research — what other people and companies actually do

Everything in this section is sourced. Full links in §19.

### 4.1 Video RAG: the canonical pipeline

The industry has converged on a remarkably consistent shape for video retrieval, and it's worth naming because it tells you exactly which parts of your system are load-bearing.

```
ingest → segment → embed (multimodal, timestamped) → vector store with metadata
                                                          ↓
query → embed → ANN candidate retrieval (stage 1) → rerank / temporal fusion (stage 2)
                                                          ↓
                          top-k CLIPS (not full videos) → video-language model → answer
```

The single most repeated lesson: **pass clips, not videos, to the expensive model.** One reported benchmark went from 72 s to 20 s of processing with equal accuracy purely by narrowing what reached the VLM. Your Omniscient path already does the right thing here (chunk hit → frame scoring → peak detection → clip to Qwen); your CV path does not.

The second lesson is about **segmentation granularity**. Fixed-window chunking (your 5 s / 15 s modes) is the pragmatic default, but the literature consistently finds that *semantic* segmentation — cutting at scene/topic boundaries — improves retrieval, because a chunk that straddles a topic change embeds to the average of two things and is close to neither. Cheap proxies for scene boundaries: ffmpeg's `select='gt(scene,0.4)'`, or the RAFT motion signal you're already computing.

The third: **metadata is not optional.** Every production video RAG system stores rich metadata alongside vectors (timestamps, speaker, source, modality, confidence) because filtering before ANN is what makes retrieval fast *and* precise. Pure vector search with no filters is a demo, not a system.

### 4.2 Hybrid search: BM25 + dense + Reciprocal Rank Fusion

This is, by a wide margin, the highest-leverage finding in the research for your "100% accurate" goal.

**The finding:** hybrid retrieval — combining a lexical scorer (BM25) with a dense vector scorer, and fusing them — is repeatedly reported as the single most impactful improvement teams make after getting a baseline RAG pipeline working.

**Why it works:** the two methods fail in *opposite* directions.
- BM25 nails exact tokens: proper nouns, product names, error codes, version numbers, acronyms. It has no idea that "car" and "automobile" are related.
- Dense vectors nail paraphrase and concept, and are terrible at rare literal tokens they've never seen — a model that never saw `Qwen2.5-VL` in training embeds it as noise.

On a document-QA benchmark (T2-RAGBench), BM25 beat `text-embedding-3-large` on every metric except Recall@20. That's not "BM25 is better" — it's "you cannot predict which one wins for a given query, so run both."

**Reciprocal Rank Fusion** is how you combine them:

```
score(d) = Σ  1 / (k + rank_i(d))        over each retriever i,  k = 60
```

The crucial property: **RRF operates on ranks, not scores.** BM25 scores are unbounded and corpus-dependent; cosine similarity is bounded in [-1, 1]. There is no principled way to normalise one into the other. Ranks sidestep the problem entirely, are immune to score outliers, and work across sharded indexes. `k=60` is the standard constant from the original paper and is widely used unchanged.

**Two-stage retrieval is the other half.** Retrieve ~100 candidates cheaply via RRF, then rerank with a cross-encoder to get the final ~10. A cross-encoder scores `(query, document)` jointly rather than comparing two independent embeddings, so it's dramatically more accurate — and dramatically slower, which is exactly why it only ever runs on 100 items, never on the corpus.

**For you, with one user, this is a gift.** A cross-encoder over 100 candidates is ~100 ms on CPU with a small model. You can afford to run it on *every single query*. Google can't.

**Evaluation thresholds** that the RAG community treats as "good": faithfulness > 0.85, answer relevancy > 0.8, context precision > 0.75, context recall > 0.8. Tooling: RAGAS, DeepEval, Promptfoo. §5/G7 argues you need this before you can claim any accuracy number.

### 4.3 Late interaction: ColPali, ColQwen, and the video variants

The frontier technique, and directly relevant to your frame-embedding design.

**Standard approach:** one vector per image/document. Query → one vector → cosine. Fast, lossy — everything in the image is averaged into 768 or 1152 numbers.

**Late interaction (ColBERT-style):** keep *many* vectors per item. ColPali runs a SigLIP-So400m vision encoder over a page image, producing **1024 patch embeddings** (a 32×32 grid over 448×448 px), each projected to 128 dimensions. The query is also multi-vector. Scoring is **MaxSim**: for each query token vector, take its maximum similarity against all document patch vectors, then sum.

```
score(q, d) = Σ_i  max_j  ⟨ q_i , d_j ⟩
```

**Why it matters:** the accuracy edge over single-vector is roughly **1–2% on broad topical queries but 10–20% on entity-rich factoid queries** — i.e. exactly the queries where you'd say "the search is broken." "The video where the guy draws the three-circle diagram on a whiteboard" is an entity-rich factoid query.

**The cost:** storage. Roughly 1B vectors per 1M pages. For video this is worse — a 2 fps sample of a 10-minute video is 1,200 frames × 1,024 patches = 1.2M vectors *for one video*.

**The relevant variants:** ColQwen2 (Qwen2-VL backbone, Apache 2.0), Video-ColBERT (CVPR 2025), CLaMR, ColQwen-Omni.

**My read for VIOS:** do not adopt full late interaction as the primary index — the storage cost is incompatible with "ship the DB through Telegram." But there's a hybrid that's very attractive: **single-vector SigLIP for the first-stage recall, late-interaction MaxSim as the reranker over the top ~50 frames.** You compute the patch embeddings on demand at query time from the cached frames (or store them only for the frames that survived first-stage retrieval in past queries). That gives you the 10–20% factoid accuracy win at ~0 storage cost, paid for in query latency you can afford.

### 4.4 Knowledge organisation and learning-path generation

This is where your Output 3 lives, and the research says something encouraging.

**The core primitive** is the **prerequisite relation**: `p → q` means "you must understand p before q." A concept graph over these relations is a **DAG**, and **a topological sort of that DAG is a learning path.** That's the whole trick. Everything else is how you get the edges.

**Two families of approaches:**
1. *Extraction from content* — read the corpus, infer prerequisites from how concepts are introduced and referenced.
2. *Inference within an existing graph* — start from a curated concept graph and predict missing edges.

**The most relevant recent work** is the KnowLP / EDU-Graph-RAG line, which runs in two stages:
- **Knowledge Structure Graph Generation** — chunk the text, use tuned LLM prompts to extract both *prerequisite* and *similarity* relations between concepts.
- **Learning Path Generation** — a reinforcement-learning "P-agent" (PPO) sequences the graph into a path, optimising for coverage and ordering.

**A crucial asymmetry** the literature flags: a **false-positive prerequisite is more harmful than an omitted one.** A spurious edge injects a hard ordering constraint that forces the learner through irrelevant material and can create cycles that break the topological sort. A missing edge just means the path is slightly less careful. **Design your extractor to be conservative — high precision, accept low recall.**

**Personal Knowledge Graphs** (Balog & Kenter, 2019) is the other relevant thread. The key design guidance: a PKG should *link to* public knowledge graphs (Wikidata, ORKG) rather than try to absorb them. Your graph stores "Devansh's corpus mentions concept X" and holds a Wikidata QID; it does not try to redefine what X is.

**And here is the gap I want you to notice:** the prerequisite / learning-path literature almost universally assumes a **curated pedagogical corpus** — textbooks, MOOCs, structured courses with existing chapter ordering. The PKG literature is about *entities and facts*, and ignores sequencing entirely. **Nobody has published prerequisite ordering over a noisy personal video corpus.** That's exactly your Output 3. It's genuinely novel, which means (a) it's exciting, and (b) there's no recipe to copy, so it needs to be built incrementally with a fallback that degrades gracefully.

### 4.5 Telegram as permanent, free, unlimited storage

Your instinct here is sound and there's a mature ecosystem proving it.

**The reference implementation** is **teldrive** (Go). It presents a Telegram channel as a filesystem, and the key technique is **chunked upload**: files larger than the per-file limit (2 GB free / 4 GB Premium) are split, each part uploaded as its own message, and a metadata database records the part ordering. On download, parts are fetched and reassembled. That's how "unlimited free storage" actually works in practice.

Important operational details:
- teldrive needs an **external Postgres** for its metadata (people commonly use Supabase's free tier). That metadata DB is the single point of failure — if you lose it, you have a channel full of anonymous binary blobs.
- **rclone has a `teldrive` backend**, default chunk size 512 MB. That means you get `rclone sync`, `rclone mount`, encryption via `rclone crypt`, and checksums — for free, without writing an uploader.
- WebDAV mounting is supported, so the channel can literally appear as a drive.

**For streaming rather than downloading**, the technique is different and more interesting:
- The **Bot API `getFile` endpoint caps at 20 MB** unless you run a local Bot API server. For video this is a non-starter.
- The real approach is **MTProto `upload.getFile`** with `offset` and `limit`, wrapped in an HTTP server that returns **`206 Partial Content`**. That gives you byte-range seeking — the browser's `<video>` element can then seek into a file that lives in Telegram, without ever downloading it whole.
- **Hard protocol constraints:** the offset must be **divisible by 4096**, and the limit must be a **power of two**. Align your range math to 512 KB or 1 MB chunk boundaries.
- **`tgcrypto` is essential** — MTProto uses AES-IGE, and the pure-Python implementation is the throughput bottleneck. With the native extension it isn't.
- **Multiple bot tokens spread `FLOOD_WAIT`.** TG-FileStreamBot round-robins across up to 50 bot tokens for exactly this reason.

Working reference projects: TG-FileStreamBot (Go, gotgproto, multi-token round-robin), stremio-telegram-debrid (Pyrogram + tgcrypto + Uvicorn, HTTP 206 seeking), Telegram-Stremio, webbridgebot.

**Risk to be clear-eyed about:** this is against the spirit (and arguably the letter) of Telegram's ToS for a large enough corpus, and channels have been removed. **Mitigation: the artifact bundle must be reproducible from the video corpus, and the video corpus should have at least one other copy.** Never let Telegram be a single point of failure for something you can't rebuild.

### 4.6 The free infrastructure landscape, August 2026

The landscape has moved significantly and mostly in a bad direction. Here's where it actually stands.

#### Cloudflare — the backbone of the plan

| Product | Free tier | Notes |
|---|---|---|
| **R2** | 10 GB storage, 1M Class A ops/mo, 10M Class B ops/mo | **Zero egress. Forever.** This is the killer feature — S3 would bill you for every byte served. |
| **Workers** | 100k requests/day | **10 ms CPU per invocation**, 50 subrequests/invocation, 3 MB gzipped bundle |
| **D1** (SQLite) | 5 GB | 5M rows read/day, 100k rows written/day |
| **Vectorize** | 5M stored vectors | 30M queried vector-dimensions/month |
| **KV** | 1 GB | 100k reads/day, **1k writes/day** |
| **Durable Objects** | 100k requests/day | SQLite backend only on free |
| **Queues** | 10k ops/day | |
| **Workers AI** | 10,000 neurons/day | Includes embedding models |
| **Pages** | Unlimited bandwidth | |

Two warnings that matter enormously:
- **The free CDN's ToS prohibits hosting video**, and there's a 512 MB per-file cache limit. **Serve video from R2, not through the CDN cache.** R2 has its own egress-free path.
- **Durable Object writes have no spending cap.** There's a well-known anecdote of a runaway loop generating a $36,000 bill. If you use DOs, put a hard iteration guard in the code and set an account-level budget alert on day one.

The **10 ms CPU limit** is the real design constraint. It means the Worker must be a *router and a range-proxy*, not a compute engine. No embedding, no reranking, no SQL scans inside the Worker. This pushes work to (a) the browser via WASM, or (b) precomputed artifacts.

#### Oracle Cloud Always Free — halved, but still the best free VM

**Critical update: on 15 June 2026 the Ampere A1 allocation was cut in half — from 4 OCPU / 24 GB RAM to 2 OCPU / 12 GB RAM.** There was no public announcement; enforcement has been inconsistent, and instances over the new limit may be shut down. The 200 GB block storage and the 2× AMD micro instances are unchanged.

2 OCPU / 12 GB / 200 GB is still, by a distance, the most capable genuinely-free always-on machine available. Caveat: you must choose a Home Region that has A1 capacity, and **the home region cannot be changed later**.

#### Everything else, briefly

| Platform | Verdict |
|---|---|
| **Fly.io** | ❌ Free tier gone. New accounts get a trial of 2 VM-hours or 7 days, whichever ends first. ~$2/mo for a 256 MB always-on machine. |
| **Railway** | ❌ No free tier since 2023; removed prepaid credit in early 2026, now requires a post-paid card. $5 one-time trial credit. |
| **Render** | ⚠️ Real free tier, but web services **spin down after 15 min idle** with a 30–50 s cold wake. Static sites don't sleep. |
| **Hugging Face Spaces** | ✅ Free CPU Basic: **2 vCPU, 16 GB RAM**, sleeps after **48 hours** idle — far more forgiving than Render. ZeroGPU (H200) is quota-based. Community GPU grants exist for qualifying open projects. |
| **Deno Deploy** | ✅ Genuinely free, hard-capped (no overage billing): ~1M req/mo, 50 ms CPU/request, 1 GiB KV. JS/TS only. |
| **Val Town** | ⚠️ As of May 2026, **new vals on the free plan must be public**. Disqualifying if you have secrets. JS/TS only. |
| **GCP** | ✅ The only Big Four provider with a permanent free VM (e2-micro). |
| **Self-hosted PaaS** | Coolify (most polished, biggest template library) > Dokploy (simplest) > CapRover (Docker-native) > Dokku. Free at the licence level; the cost is you patching the platform. |

**The takeaway:** the only genuinely-free always-on compute in 2026 is a self-hosted VM (Oracle A1 or GCP e2-micro), and the only genuinely-free always-on *serving* is Cloudflare's edge. HF Spaces' 48-hour sleep window makes it a viable third leg for anything that needs Python and 16 GB of RAM.

### 4.7 Querying a database over HTTP without a server

This is the most "outside the box" finding and it's directly applicable.

**`sql.js-httpvfs`** (phiresky) implements a **SQLite virtual filesystem over HTTP Range requests.** You host a plain `.sqlite` file on any static host — GitHub Pages, R2, S3 — and the browser runs real SQL against it, fetching only the pages it needs. It uses three virtual "read heads" with prefetch ramping to amortise latency.

**The absolutely critical constraint:** *queries must hit an index.* A query that causes a table SCAN downloads the entire table. With indexes, a lookup in a multi-GB database is a handful of KB. Without them, it's the whole file. The author is candid that it's demo-grade — there's no cache eviction, so a long session accumulates pages in memory.

**`sqlite-wasm-http`** is the modern successor. It needs COOP/COEP headers for `SharedArrayBuffer`, which Cloudflare Pages can set trivially.

**`sqlite3vfshttp`** (Go) does the same server-side and works against presigned S3/R2 URLs.

**DuckDB-WASM + Parquet** is the analytical counterpart: excellent for scans and aggregations over columnar data, but it has **no persistent indexes**, so it's poor at the selective point-lookups that dominate search. The right split is: **SQLite for retrieval, DuckDB/Parquet for analytics dashboards.**

**Why this matters so much for you:** it means the "serve plane" can be *entirely static*. R2 stores an index bundle; the browser queries it directly over range requests; the Worker only signs URLs and proxies video. No always-on server, no database bill, no cold starts, and it scales to a corpus far larger than any free managed DB would allow.

### 4.8 Video UI/UX: the patterns worth stealing

I researched how the best video tools handle exactly your problem — navigating and searching long media.

**The dominant idea: text is the navigation surface.**

- **Descript** made this mainstream for editing: the transcript *is* the timeline. Its interface has a Time Ruler (timecodes + markers for jumps), a Layer Lane (per-track), a "Wordbar" linking script words to timeline sync, and a Script Track. Crucially, it keeps **both** modalities — text for structure, timeline for precision. Users can highlight transcript words to cut/move/delete, *or* use a blade tool at exact timecodes.
- **Reduct** is closer to your use case: it's built for searching, tagging, and pulling insights across a large footage library, not for assembling a cut. Its standout capability — **highlight key quotes across dozens of different videos and automatically stitch them into a highlight reel** — is essentially your "search returns moments" feature with a compile step on top.
- The enabling primitive for both is **word-level timecodes**. Whisper gives you these (`word_timestamps=True`); once you have them, every word in the UI is a frame-accurate seek target.

**Frame.io's contribution: timecode as the anchor for everything.**
- A comment posts as both a card in a list *and* a bubble on the scrubber.
- **Bidirectional linkage** — click the comment, seek the video; seek the video, highlight the comment.
- Typing in the comment box **auto-pauses**; sending resumes. Tiny detail, enormous friction reduction.
- Comments can optionally carry *no* timestamp — an escape hatch for general notes.
- Infrastructurally: Frame.io **transcodes to multiple resolutions for fast scrub and playback** while keeping the original for download. Proxy generation is a *prerequisite* for responsive scrubbing, not an optimisation.

**Scrubbing patterns:**
- Netflix: large hit targets, thumbnail previews while scrubbing, subtle controls.
- YouTube: chapters, captions, speed, keyboard shortcuts, transcript panel — all coexisting.
- The counterintuitive principle from the UX literature: **the best playback UI is nearly invisible**, and predictability beats discoverability — the user should never wonder how to bring controls back.
- A known flaw worth designing around: conventional scrubbers are a **rigid undifferentiated scale** — the same bar length whether the video has 60 frames or 60,000 — so a dense passage gets compressed into a few pixels. The proposed remedy in the patent literature is a **content-proportionate timeline** where sections are sized by scene density. You have the RAFT motion signal and scene-cut data to do this.

**Thumbnail sprites — the standard, exactly.**

The format is a WebVTT file whose cues map time ranges to rectangles in a sprite image, using media-fragment `#xywh` syntax:

```
WEBVTT

00:00.000 --> 00:05.000
storyboard.jpg#xywh=0,0,128,72

00:05.000 --> 00:10.000
storyboard.jpg#xywh=128,0,128,72
```

Generation is one ffmpeg command — one frame every 10 s, scaled to 160 px, tiled 10×10:

```bash
ffmpeg -i input.mp4 -vf "fps=1/10,scale=160:-1,tile=10x10" -frames:v 1 storyboard.jpg
```

Best practices from the research:
- **Always sprite, never individual files** — hundreds of HTTP requests during a scrub is the failure mode.
- **Shard long content across multiple sheets** (one sheet per few hundred thumbs); WebVTT handles this because each cue names its own image.
- **Scale the interval to duration** — ~2 s for short clips, much wider for long ones. The seek bar has finite pixels; too many frames is confusing and forces an oversized mosaic download.
- **128 px wide thumbs**, height from aspect ratio, is a common player requirement.
- **The silent bug is cue/extraction misalignment** — if your VTT intervals don't exactly match the ffmpeg extraction interval, previews drift further out of sync the longer the video runs.
- The VTT is **out-of-band** — not referenced from the HLS manifest, passed separately to the player.

**Player reality check:** **hls.js will not do thumbnails for you.** It's a transport layer that feeds a standard `<video>` element; thumbnail seeking has been an open "help wanted" request since 2020. You handle it in your own UI: fetch the VTT, parse cues, and on scrub-bar `mousemove` set `background-image` + `background-position` from the `#xywh` values. Video.js has plugins (`videojs-sprite-thumbnails`, and a VTT-driven `player.thumbnails({src})`).

**Mobile gotcha:** hover doesn't exist on touch. You need an explicit long-press or drag-preview mode.

**Media asset management search patterns** worth noting: natural-language search over the library, **timestamped auto-tagging** (object detection results become scrubber-level navigation markers), and comment-on-timecode inside the asset manager itself.

**One honest tension from the research:** transcription and AI features are now table stakes, so differentiation comes from the **navigation model**, not the underlying ASR. That's good news for you — your differentiation is the cross-video moment graph, which nobody consumer-facing has.

### 4.9 How to structure a large Python monorepo

Since you want the repo to look like a serious project, here's what the current best practice actually is.

**uv workspaces** are the current answer, and the proof point is substantial: **Apache Airflow migrated to uv workspaces — 120+ distributions, 700+ dependencies, 3,600+ contributors** (Jarek Potiuk, FOSDEM 2026). `uv sync` resolves 900+ packages in seconds.

The structure:
- **`apps/`** (deployable things) and **`packages/`** (shared libraries). That's the whole layout decision.
- **A "virtual root"**: delete the `[project]` table from the root `pyproject.toml` entirely; keep only

  ```toml
  [tool.uv.workspace]
  members = ["packages/*", "apps/*"]

  [tool.uv.sources]
  core = { workspace = true }
  ```

  This makes the root a coordinator, not a package.
- **One `uv.lock` for the entire workspace.** Every app resolves against the same dependency set, which eliminates the "works in service A, breaks in service B" class of bug.
- `uv init --lib` generates the src layout.
- **Strict dependency direction: apps depend on packages, never the reverse, and sibling apps never import each other.** This is the rule that keeps a monorepo from degenerating into a tarball.

**Known weak spot:** uv cannot *build* workspaces into distributable artifacts. Workarounds are Una, or just building per-app Dockerfiles.

**Tooling:** `prek` is a monorepo-aware pre-commit replacement.

---

## 5. Gaps, bad logic, and could-be-betters

Ranked by how much they hurt. Each has a finding, an explanation, and a recommendation.

---

### G1 — Nothing survives the session *(Existential)*

**The problem.** `lake.db`, the Qdrant collection, the Postgres tables, the Neo4j graph, the extracted frames, the thumbnails — all of it lives on Kaggle scratch or the 20 GB output quota and vanishes when the session ends. Every session either starts from zero or re-does work that was already done.

**Why it's the top item.** Everything else in this document is an optimisation. This one determines whether the project is a system or a demo. You cannot build Output 3 (a knowledge graph that accretes over years) on storage that resets every 12 hours.

**Recommendation.** Make the **artifact bundle** a first-class, versioned, content-addressed object that is uploaded to Telegram at the end of every session and downloaded at the start of the next. Full design in §7.

---

### G2 — There is no serve plane *(Existential)*

**The problem.** The "website" is `ui_server.py` running inside the Kaggle session, exposed via a cloudflared tunnel. When the session ends, the website ends. The URL changes every time. You cannot bookmark it, share it, or use it from your phone while the GPU box is off.

**Why it's structural, not cosmetic.** The compute plane and the serve plane have opposite requirements. Compute wants a beefy ephemeral GPU box. Serving wants a tiny always-on edge process. Fusing them means you get the worst of both: a website that's only up when you're burning GPU hours.

**Recommendation.** Split them completely. Compute emits artifacts; serve reads artifacts. The serve plane is Cloudflare Pages + a Worker + R2, and it is up 100% of the time whether or not Kaggle is running. §11.

---

### G3 — Two intelligence pipelines that don't know about each other *(Critical)*

**The problem.** You have:

| | CV pipeline | Omniscient pipeline |
|---|---|---|
| Process | `model_manager.py` | `omni_engine.py` |
| Frames | ffmpeg dual-tier → `frame_notes` | own sampling → PG `frames` |
| Text | Whisper → `transcripts` | Qwen narratives → PG `chunks` |
| Search index | SQLite FTS5 `moments_search` | Qdrant vectors + Neo4j |
| Query entrypoint | web UI search box | Telegram bot |

They process **the same videos, twice**, into **two schemas**, queried through **two interfaces**, with **no join key between them**. A search in the web UI cannot see Qwen's narrative. A search via the bot cannot see the FTS index or the YOLO objects.

**This is the merge artifact.** It made sense as "two things that each worked, now running side by side." It doesn't make sense as an architecture.

**Why it's expensive.** Beyond the wasted GPU: it means neither index is complete, so neither can be accurate, so you can never reach the "100%" bar with either one.

**Recommendation.** One ingest pipeline, one canonical schema, one query API, N presentation surfaces (web, bot, CLI, whatever). §6.

---

### G4 — Stage B was specified and never built *(Critical)*

Covered in §2. The models are warm and correct; the ~150 lines that consume them don't exist. Until they do, natural-language search in the CV half is FTS-only — it can find "neural network" if someone *said* "neural network," and it cannot find "the part where he draws the loss curve going down."

**Recommendation.** Build it — but build it *once*, in the unified pipeline, not twice.

---

### G5 — GPU 1 co-tenancy across two processes *(High)*

Covered in §3. ~8.6 GB of static weights on a 15 GB card, split across two processes with independent caching allocators that cannot share freed blocks.

**Recommendation.** One process owns all GPU work. Whisper and Qwen never need to run simultaneously for the *same* video — transcription happens at ingest, narration happens at chunk level, and they can be sequential stages in one worker with an explicit unload between them if headroom is tight. Alternatively: Whisper on GPU 0 (which has more headroom after dedup, per G6) and Qwen alone on GPU 1.

---

### G6 — Every shared model is loaded twice *(High)*

**The problem.** SigLIP so400m, CLIP-L, RAFT-L, and EasyOCR are each loaded by *both* `model_manager.py` and `omni_models.py`, both onto `cuda:0`. That's roughly 1.6–2 GB of VRAM spent storing byte-identical weights, plus double the load time at boot.

**Recommendation.** Falls out of G3/G5 automatically once one process owns the GPU. If you ever genuinely need two GPU processes, the tool is a small model-server (Triton, Ray Serve, or a hand-rolled ZMQ shim) — but for one user, one process is simpler and strictly better.

---

### G7 — "100% accurate" is currently unmeasured *(High)*

**The problem.** There's no eval set, no metric, and no regression harness. So "improve search" is unfalsifiable — every change feels like it might have helped.

**Why it's higher priority than it looks.** You're about to make a lot of retrieval changes (hybrid, RRF, reranking, late interaction). Without a baseline, you'll have no idea which of them helped, and you'll carry the ones that hurt.

**Recommendation.** Build a **golden set** of 100–200 queries with hand-labelled correct `(video, timestamp±5s)` answers, drawn from your own real usage. Track Recall@1, Recall@5, MRR, and nDCG@10. Run it on every retrieval change. This is maybe two evenings of work and it's the difference between engineering and vibes. Community thresholds for the generation side: faithfulness > 0.85, answer relevancy > 0.8, context precision > 0.75, context recall > 0.8.

---

### G8 — Retrieval is single-stage, with no fusion and no reranking *(High)*

**The problem.** The CV path is FTS-only. The Omniscient path is: BGE chunk hit → SigLIP+CLIP frame scoring → peak detection. That's a reasonable cascade but it's still fundamentally one dense retriever with a visual rescoring step. There's no lexical retriever, no rank fusion, and no cross-encoder.

Per §4.2 this is the single most impactful improvement available, and you already have both halves — an FTS5 index (lexical) and vector indexes (dense) — sitting in separate systems, unfused.

**Recommendation.** RRF over {BM25/FTS5, BGE-text, SigLIP-visual, CLIP-visual} → top-100 → cross-encoder rerank → top-10. §8.

---

### G9 — Playing a search result means fetching the original video *(High)*

**The problem.** There are no proxies and no sprites. A search returns `(video, t=347s)`; opening it requires the original file. On Kaggle it's on local disk (fine); from the serve plane it means pulling a possibly-hundreds-of-MB file from Telegram before the first frame renders.

Per §4.8: **proxy transcoding is a prerequisite for responsive scrubbing, not an optimisation.** Frame.io does it, Netflix does it, everyone does it.

**Recommendation.** During ingest, emit three artifacts per video, all cheap: a ~480 p H.264 proxy, an HLS segment set, and a sprite sheet + WebVTT. §9.

---

### G10 — The repository is flat *(Medium)*

37 tracked files at the root, mixing entrypoints, libraries, HTML, shell scripts, and dead code. It reads like a notebook that grew, which is exactly what it is. §15 has a full proposed layout.

---

### G11 — Five datastores for one user *(Medium)*

Redis + SQLite + Postgres + Qdrant + Neo4j. Each one is a process to start, a failure mode to handle, a schema to version, and a thing that can't be shipped through Telegram easily.

**Honest assessment of each:**

| Store | Verdict |
|---|---|
| **Redis** | Keep. It's the queue and it's ephemeral by design. Correct tool. |
| **SQLite** | Keep, and **promote it to the canonical store.** It's a single file, it has FTS5, it has fast vector extensions now (`sqlite-vec`), and it's the *only* one of the five that can be shipped through Telegram and queried from a browser over HTTP range requests (§4.7). That last property is decisive. |
| **Postgres** | Drop. Nothing it does here needs a server. Its tables (`frames`, `chunks`) map onto SQLite directly. |
| **Qdrant** (embedded) | Drop for the artifact plane; the embedded single-process mode is also what forced the dashboard-in-engine + proxy contortion. Replace with `sqlite-vec` (vectors live in the same file as everything else) or a plain NumPy `.npy` memory-mapped matrix. For ~100k vectors, brute-force cosine in NumPy is *sub-millisecond* and beats an ANN index on accuracy. You are not at a scale where ANN is necessary. |
| **Neo4j** | Drop as a runtime dependency. The graph is small and read-mostly; store it as edge tables in SQLite and load into NetworkX (or a JS graph lib in the browser) when you need traversal. Neo4j needs a JVM, a heap config, and a startup dance for something you could hold in 20 MB of RAM. |

**Recommendation.** Collapse to **SQLite (canonical, shippable) + Redis (ephemeral queue) + flat files (frames, sprites, proxies)**. This one change makes G1, G2, and G10 all dramatically easier.

---

### G12 — No schema versioning *(Medium)*

If the DB is going to survive across sessions and be downloaded from Telegram months later, a schema change becomes a migration problem. Right now there's no version marker and no migration path.

**Recommendation.** A `schema_version` table, forward-only numbered migrations, and a `bundle_manifest.json` inside every artifact bundle recording schema version, model versions, and the ingest code commit hash. That manifest is also what lets you answer "which videos need reprocessing because I upgraded the embedding model?"

---

### G13 — Dead code *(Low)*

`admin_backend.py` (25 KB), `admin_ui.html` (27 KB), `main_ui.html` (34 KB), `local_preview.py`, `push.py`, `test_harness.py`, and the `public/placeholder-*` assets appear to be superseded or unused. That's ~90 KB of HTML alone that a reader has to determine is irrelevant.

**Recommendation.** Delete in the reorganisation. Git remembers.

---

### Design observations that aren't quite bugs

**O1 — Fixed-window chunking is leaving accuracy on the table.** 5 s / 15 s windows are simple but they cut through the middle of ideas. Scene-boundary segmentation (ffmpeg `select='gt(scene,0.4)'`, or thresholding your RAFT motion signal) would produce chunks that embed more cleanly. Worth an A/B once G7 exists.

**O2 — The Whisper generator is consumed inside a list comprehension.** In `analyze_audio`, transcription actually executes at the comprehension, not at the `.transcribe()` call. That's correct behaviour but it means timing instrumentation around `.transcribe()` measures nothing, and an exception surfaces at a confusing line. Worth a comment.

**O3 — `model_manager.py` has no CUDA-context-loss guard.** The Omniscient loops correctly treat a lost context as fatal and `os._exit(1)`. The Phase-2 analysis loop just does `fail_job` + `clear_ram()` + `sleep(2)` — so if it ever *does* lose its context, it will fail every subsequent job in a tight loop rather than restarting. The Omniscient policy should be applied here too.

**O4 — GraphRAG silently no-ops without a NIM key.** A core feature of the knowledge layer degrades to nothing with only a `WARN` line. Either make it a hard startup check, or (better, §10) replace the external LLM dependency with the local Qwen you already have loaded.

**O5 — The dedup-rebuild-from-DB pattern is excellent and should be generalised.** "Redis is ephemeral, the DB is truth, rebuild the set on boot" is the right instinct. Apply the same thinking one level up: "Kaggle is ephemeral, Telegram is truth, rebuild the artifact bundle on boot."

**O6 — Priority routing asymmetry is intentional and correct**, for the record: browsed-category videos jump the CV queue (`ui_server.py:389`), harvest jobs ride the default lane (`ui_server.py:403`), bot uploads always pre-empt (`omni_engine.py:740`). That's a coherent policy; don't let a refactor flatten it.

---

## 6. The target architecture

Here's the shape I'd build toward. It is deliberately *smaller* than what exists now in terms of moving parts, and much larger in capability.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CAPTURE                                                                    │
│  Telegram channel — videos, immutable, msg_id = primary key                 │
│  Telegram channel #2 (private) — artifact bundles, versioned                │
└─────────────────────────────────────────────────────────────────────────────┘
                    │ pull videos                          ▲ push artifacts
                    ▼                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPUTE  (Kaggle session — disposable, idempotent, restartable)            │
│                                                                             │
│   boot ─▶ restore bundle from Telegram ─▶ rebuild Redis dedup set           │
│                                                                             │
│   harvester ──▶ QUEUE_INGEST ──▶ ┌─ media worker (CPU, all cores) ─────┐    │
│                                  │  ffmpeg: proxy + HLS + sprites +    │    │
│                                  │  frames (2 tiers) + audio           │    │
│                                  └──────────────┬──────────────────────┘    │
│                                                 ▼                           │
│                                  ┌─ GPU worker (ONE process, both GPUs) ─┐  │
│                                  │  Whisper → word-level transcript      │  │
│                                  │  YOLO + OCR → frame notes             │  │
│                                  │  SigLIP + CLIP → frame vectors        │  │
│                                  │  DINOv2 → visual-similarity vectors   │  │
│                                  │  RAFT → motion/pacing curve           │  │
│                                  │  BGE → chunk text vectors             │  │
│                                  │  Qwen2.5-VL → chunk narrative + Q&A   │  │
│                                  │  Qwen (text) → concepts + prereqs     │  │
│                                  └──────────────┬────────────────────────┘  │
│                                                 ▼                           │
│                          ONE SQLite file (WAL) + flat media dirs            │
│                                                 │                           │
│   shutdown ─▶ seal bundle ─▶ chunk ─▶ upload to Telegram ─▶ write manifest  │
└─────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼  (sync, once per session)
┌─────────────────────────────────────────────────────────────────────────────┐
│  ARTIFACT  (Cloudflare R2 — 10 GB free, zero egress)                        │
│    index.sqlite          ← queried from the browser over HTTP Range         │
│    vectors/*.f16.bin     ← memory-mappable matrices                         │
│    sprites/{id}.jpg+vtt  ← scrub previews                                   │
│    proxies/{id}/*.m3u8   ← HLS segments for instant playback                │
│    manifest.json         ← schema + model versions, bundle hash             │
└─────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SERVE  (always on, free, no cold start)                                    │
│    Cloudflare Pages  → the app (static SPA)                                 │
│    Cloudflare Worker → R2 signing, Telegram range-proxy, tiny API           │
│    Browser           → sql.js-httpvfs over index.sqlite                     │
│                        + WASM embedder for query vectors                    │
│                        + in-browser reranking                               │
│    Optional: Oracle A1 box for heavy/always-on work (§11)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The five architectural decisions this encodes

**1. One canonical store: SQLite.** Everything — posts, videos, transcripts (word-level), frame notes, chunks, narratives, vectors (via `sqlite-vec` or sidecar `.bin`), concepts, prerequisite edges, and the FTS5 index — lives in one file. That file is the product. It ships through Telegram, it lands in R2, and the browser queries it directly. Postgres, Qdrant, and Neo4j all go away.

**2. One GPU process.** Every model loads once. Whisper and Qwen are sequenced, not co-resident under contention. G5 and G6 dissolve.

**3. Compute is stateless.** A Kaggle session's entire job is: pull bundle → process new videos → push bundle. If it crashes at any point, the next session picks up from the last sealed bundle. Nothing is lost because nothing was ever only-here.

**4. The serve plane never writes.** It's a read-only view over an immutable artifact. That's what makes it free (no database), fast (no cold start, edge-cached), and simple (no auth, no consistency).

**5. The API is the contract, not the UI.** Your Output 4 ("integrate the database into any UI I desire") is satisfied by construction: the artifact bundle *is* the API. Any client that can do HTTP range requests and run SQLite can be a full client. That includes a CLI, an Obsidian plugin, a Raycast extension, a phone app, or an MCP server for Claude.

### What each layer explicitly does *not* do

- **Compute does not serve.** No tunnel, no public URL, no user-facing latency requirement.
- **Serve does not compute.** No embedding models on the Worker (the 10 ms CPU limit forbids it anyway); query embedding happens in the browser via WASM or is precomputed.
- **Telegram does not index.** It's dumb object storage. If Telegram is unavailable, the R2 copy still works; if R2 is wiped, Telegram rebuilds it.

---

## 7. Database permanence: making Kaggle disposable

Your instinct — "upload the database to Telegram, then redownload it in such a way that it feels like we never moved it" — is exactly right, and there's a well-trodden path (§4.5). Here's how I'd design it.

### The artifact bundle

At the end of a session, everything worth keeping is sealed into a **bundle**:

```
bundle-v{schema}-{seq}-{sha256[:12]}/
├── manifest.json          # schema version, model versions, code commit,
│                          # video count, byte sizes, per-part checksums
├── index.sqlite.zst       # the canonical DB, zstd-compressed
├── vectors/
│   ├── siglip.f16.bin     # [N × 1152] float16, row i ↔ frame_id in DB
│   ├── clip.f16.bin       # [N × 768]
│   ├── dinov2.f16.bin     # [N × 1024]
│   └── bge.f16.bin        # [M × 1024]  chunk text vectors
├── sprites/{msg_id}.jpg + {msg_id}.vtt
└── proxies/{msg_id}/index.m3u8 + *.ts
```

**Why vectors are sidecar `.bin` files rather than SQLite BLOBs:** a `[N × 1152]` float16 matrix is memory-mappable and directly consumable by NumPy with zero parsing. Storing them as per-row BLOBs means N round trips and N deserialisations. For range-request serving it also means the browser can fetch *exactly* the byte range for a vector without pulling database pages. Store the mapping (`frame_id → row_index`) in SQLite.

**Why float16:** halves the size at negligible accuracy cost for cosine similarity. 100k frames × 1152 dims × 2 bytes = 230 MB for SigLIP. Quantise to int8 later if size becomes the binding constraint (typically <1% recall loss with per-vector scale factors).

### Upload strategy

Follow the teldrive model (§4.5):

1. **Chunk** each file into parts below the per-file limit (2 GB free, 4 GB Premium). Use a fixed part size — 512 MB is rclone's default and a sane choice.
2. **Upload each part** as its own message to a *private* channel dedicated to artifacts (separate from the video channel — different retention risk, different access pattern).
3. **Record the mapping** — `part_index → message_id`, plus a SHA-256 per part — in the manifest.
4. **Upload the manifest last**, as a small text message. It's the commit point: a bundle exists if and only if its manifest is posted.
5. **Pin the latest manifest** in the channel so restore is a single API call to find it.

**Do not write your own uploader if you can avoid it.** `rclone` with the `teldrive` backend gives you chunking, retry, checksums, and `rclone crypt` for encryption. Wrapping rclone is dramatically less code than reimplementing MTProto chunked upload correctly.

**The metadata bootstrap problem.** teldrive stores its part mapping in an external Postgres, which is a single point of failure. Avoid that: put the mapping in the *manifest message itself*, in the channel. The channel is then fully self-describing — given only the channel and a bot token, you can reconstruct everything. Belt and braces: keep a copy of the manifest in a GitHub repo (it's a few KB of JSON) and in R2.

### Restore strategy — "as if we never moved it"

This is the part your phrasing gets exactly right, and it has a precise technical meaning: **restore must be transparent to the code that uses the DB.** No code path should know whether it's a fresh session or the hundredth.

```
boot:
  1. read pinned manifest from artifact channel
  2. compare bundle hash to local marker file on scratch
  3. if different or absent:
       download parts in parallel (N workers) → reassemble → verify SHA-256
       decompress index.sqlite.zst → SCRATCH/index.sqlite
       memory-map vectors/*.bin
  4. rebuild the Redis dedup set from index.sqlite   ← already exists, unchanged
  5. every module opens SCRATCH/index.sqlite exactly as it does today
```

Step 5 is the "feels like we never moved it" property, and it's satisfied by making the path in `config.py` the only thing that ever changes.

### Three refinements that matter a lot

**A. Incremental bundles.** Re-uploading a 5 GB bundle every session is wasteful and slow. Instead:
- Ship a **base bundle** plus **deltas**. A delta contains only rows added since the base plus the new vector rows appended to the matrices (append-only matrices make this trivial — new frames are always new rows at the end).
- Every N sessions, or when deltas exceed ~30% of base size, **compact** into a new base.
- This is exactly how LSM trees and Git packfiles work, and it's the right shape here.

**B. Content addressing.** Name every part by its content hash. Re-uploading an unchanged part becomes a no-op, deduplication is free, and corruption is detectable. This is Git's core insight and it applies cleanly.

**C. Two-tier restore.** You don't always need everything.
- **Cold start** (new Kaggle session, need to process more videos): pull the whole bundle.
- **Query only** (the website): pull *nothing*. Query `index.sqlite` in R2 over HTTP range requests (§4.7). The browser downloads a few hundred KB of index pages to answer a query against a multi-GB database.

That second mode is the one that makes the website feel instant, and it only works because SQLite-over-range-requests exists.

### Size budget, roughly

| Artifact | Per 1,000 videos (~5 min avg) | Notes |
|---|---|---|
| `index.sqlite` (compressed) | ~1.5–3 GB | Transcripts + notes + narratives + FTS. FTS roughly doubles text size. |
| SigLIP vectors | ~1.4 GB | 2 fps sampling → ~600k frames × 1152 × 2 B |
| CLIP vectors | ~0.9 GB | |
| DINOv2 vectors | ~1.2 GB | Consider sampling at 0.5 fps — visual similarity doesn't need 2 fps |
| BGE chunk vectors | ~50 MB | Chunks are ~100× rarer than frames |
| Sprites | ~200 MB | ~200 KB/video |
| HLS proxies @480p | ~50–100 GB | **The big one** |

**The proxies don't fit in R2's 10 GB free tier** and they're the reason the serve plane needs the Telegram range-proxy (§9). Everything else fits comfortably: index + vectors + sprites ≈ 5–7 GB for 1,000 videos, which is inside R2 free.

**Storage-reduction levers if you need them:** drop DINOv2 sampling to 0.5 fps; int8-quantise the vector matrices (4× reduction); store CLIP only as a reranking signal computed on demand rather than a full index.

---

## 8. Search that actually earns "100% accurate"

"100% accurate" isn't achievable as a literal number, but the *feeling* you're describing — "I always find it, first try" — absolutely is, for a personal corpus. Here's the stack that gets there.

### The honest reframe

What you want is **Recall@1 approaching 1.0 on queries you actually issue.** That's a very different (and far more achievable) target than "perfect on all possible queries." And it has a specific implication: **the system should learn from your queries.** With one user, every click is an unambiguous relevance label. Google would kill for that signal density.

### The retrieval cascade

```
                     ┌──────────────── QUERY ────────────────┐
                     │  "the part where he explains          │
                     │   why attention is O(n²)"             │
                     └───────────────────┬───────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │  QUERY UNDERSTANDING (local Qwen, ~50 ms)                       │
        │  • intent: moment | video | concept | question | temporal       │
        │  • entities: ["attention", "O(n²)", "complexity"]               │
        │  • filters: none                                                │
        │  • expansions: ["quadratic", "self-attention", "scaling"]       │
        └────────────────────────────────┬────────────────────────────────┘
                                         │
   ┌──────────┬──────────────┬───────────┴────────┬──────────────┬──────────┐
   ▼          ▼              ▼                    ▼              ▼          ▼
 FTS5      BGE dense    SigLIP text→frame    CLIP text→frame  OCR FTS   Concept
 (BM25)    (chunks)     (visual semantic)    (visual)         (on-screen graph
                                                               text)    neighbours
   └──────────┴──────────────┴────────────────────┴──────────────┴──────────┘
                                         │
                     ┌───────────────────▼───────────────────┐
                     │  RECIPROCAL RANK FUSION (k=60)        │
                     │  score(d) = Σ 1/(60 + rank_i(d))      │
                     │  → top 100 candidate moments          │
                     └───────────────────┬───────────────────┘
                                         │
                     ┌───────────────────▼───────────────────┐
                     │  RERANK (cross-encoder, ~100 ms)      │
                     │  bge-reranker-base over               │
                     │  (query, transcript_window)           │
                     │  + optional ColPali MaxSim over the   │
                     │    frame patches for visual queries   │
                     └───────────────────┬───────────────────┘
                                         │
                     ┌───────────────────▼───────────────────┐
                     │  TEMPORAL FUSION                      │
                     │  merge adjacent hits into moments;    │
                     │  a 3-hit cluster beats an isolated    │
                     │  higher-scoring single frame          │
                     └───────────────────┬───────────────────┘
                                         │
                     ┌───────────────────▼───────────────────┐
                     │  RESULTS: ranked (video, t_start,     │
                     │  t_end, why-it-matched, confidence)   │
                     └───────────────────────────────────────┘
```

### Why each stage is there

**Query understanding.** Different intents need different retrieval. "Find the video about X" wants video-level ranking. "Where does he say Y" wants moment-level. "How do I do Z" wants QA synthesis. "Videos from last March about..." wants a metadata filter first. Classifying this locally with the Qwen you already have costs ~50 ms and prevents the single biggest source of bad results — running the wrong search.

**Six retrievers, not one.** Each covers a distinct failure mode:

| Retriever | Catches |
|---|---|
| FTS5/BM25 | exact names, jargon, error codes, numbers, anything a dense model never saw |
| BGE dense (chunk text) | paraphrase, concept, "the idea of X" |
| SigLIP text→frame | visual semantics — "a whiteboard with a graph on it" |
| CLIP text→frame | second visual opinion; different training data, different failures |
| OCR FTS | on-screen text — slide titles, code, captions. **Enormously underrated for tutorial content.** |
| Concept-graph neighbours | expands to related concepts you didn't name |

The point isn't that any one is great. It's that **they fail independently**, and RRF is designed to exploit exactly that.

**RRF over score normalisation.** Per §4.2 — BM25 is unbounded, cosine is [-1,1], and there's no principled mapping. Ranks are comparable by construction, immune to outliers, and cheap.

**Cross-encoder reranking.** The biggest single-stage accuracy jump available, and affordable *because you're one user*. 100 candidates × a small cross-encoder ≈ 100 ms on CPU. Run it on every query.

**Temporal fusion is the video-specific bit** that generic RAG doesn't have. A single frame at t=347 scoring 0.81 is less trustworthy than three consecutive frames at t=344/346/348 scoring 0.72/0.75/0.71. Cluster hits within a window, score the cluster, and return the *span*. This is what turns frame retrieval into moment retrieval.

### Getting to "always first try": the feedback loop

With one user, you can do something the big platforms structurally cannot:

1. **Log every query and every click.** `(query_text, query_embedding, clicked_result, rank_of_click, dwell_time)`.
2. **Treat a click at rank > 1 as a training signal.** The thing you clicked *should* have been rank 1.
3. **Maintain a learned query→result memory.** If you've searched something similar before and clicked a specific moment, boost it. This alone will make repeat searches feel telepathic, and repeat searches are most of personal search.
4. **Periodically fine-tune the embedding model** on the collected `(query, positive_moment, negative_moments)` triplets. A few hundred labelled pairs is enough to meaningfully specialise a small embedding model to *your* corpus and *your* phrasing. This is the highest-ceiling item in the whole search section and it's only possible because there's one user.
5. **Surface "did this answer it?"** — one click, no friction. Frame.io's auto-pause-on-typing lesson applies: feedback UI must cost nothing or it won't be used.

### Explainability as an accuracy feature

Every result should say *why* it matched:

> **▶ 05:47 — "Attention is Quadratic" · 94%**
> matched: transcript "the complexity grows with the square of sequence length" · on-screen text "O(n²)" · concept `computational-complexity`

This is not decoration. When a result is wrong, the explanation tells you *which retriever* misfired, which is the difference between "search is broken" and "the OCR pass mis-read this slide." It's the debugging interface for your own retrieval stack.

### Things that will surprise you with how much they help

- **OCR text is gold for technical content.** Slide titles, code on screen, terminal output. Index it separately with its own FTS table and give it its own retriever in the fusion — it's high-precision and low-recall, which is exactly what RRF handles well.
- **Speaker diarisation**, even crude, enables "what did the interviewer ask" and dramatically improves chunk boundaries.
- **Word-level timestamps** (Whisper supports them) turn every word into a seek target and make transcript-driven navigation (§4.8) possible. Enable them at ingest — retrofitting means re-transcribing everything.
- **A "more like this" button** on any frame, powered by DINOv2, will find things text search structurally cannot — same person, same set, same visual style.

---

## 9. Instant video playback

Your requirement: search returns results, and clicking one plays *that moment* immediately. Here's what "immediately" requires.

### The three artifacts, generated at ingest

**1. HLS proxy (~480p, ~800 kbps).**
```bash
ffmpeg -i input.mp4 -vf scale=-2:480 -c:v libx264 -preset veryfast -crf 26 \
       -c:a aac -b:a 96k -f hls -hls_time 4 -hls_playlist_type vod \
       -hls_segment_filename 'seg_%04d.ts' index.m3u8
```
4-second segments. Why HLS and not just the MP4: seeking to t=347 in an HLS stream fetches *one 4-second segment*, not a byte range that the server has to compute and that may not align to a keyframe. Time-to-first-frame becomes ~one segment download.

**2. Sprite sheet + WebVTT** (per §4.8).
```bash
ffmpeg -i input.mp4 -vf "fps=1/2,scale=160:-1,tile=10x10" -frames:v 1 sprite_%03d.jpg
```
Note `fps=1/2` for a 5-minute video (150 thumbs, 2 sheets). **Scale the interval to duration** — use 1/2 for short clips, 1/10 for long ones. **Shard across multiple sheets.** And **align the VTT cue intervals exactly to the extraction interval** — that's the silent drift bug.

**3. Keyframe index.** A small JSON of keyframe timestamps, so the player can snap a seek to the nearest keyframe rather than decoding forward.

Cost: all three come from ffmpeg passes you can fold into the existing dual-tier extraction. On Kaggle's CPU allocation, a 5-minute video is well under a minute for all three.

### The serving path

```
browser → <video> + hls.js
            │
            ├─ index.m3u8 ────────▶ R2  (tiny, cached at edge)
            ├─ seg_0087.ts ───────▶ R2 if present, else Worker → Telegram range-proxy
            └─ sprite_001.jpg ────▶ R2  (immutable, cached forever)
```

**The R2 budget problem and its solution.** 10 GB of free R2 holds maybe 100–200 proxied videos. For a corpus of thousands you need a tiering policy:

- **Hot in R2:** anything opened in the last 30 days, anything in a search result in the last 7 days, everything the concept graph marks as a "core" video.
- **Cold in Telegram:** everything else, served through the Worker's range-proxy on demand, and promoted to R2 on first access.

That's just an LRU cache with R2 as the fast tier and Telegram as the backing store. Since R2 egress is free and Telegram bandwidth is free, the only cost of a miss is latency.

**The Telegram range-proxy** (§4.5) is the piece that makes cold playback work: MTProto `upload.getFile` with offset/limit, wrapped in an HTTP server that returns 206. Remember:
- offset **divisible by 4096**, limit a **power of two** — align to 1 MB
- **tgcrypto is mandatory** for throughput (AES-IGE in pure Python is the bottleneck)
- **round-robin multiple bot tokens** to spread FLOOD_WAIT

**Where does the proxy run?** Not in a Cloudflare Worker — the 10 ms CPU limit and the lack of a persistent MTProto connection make it a bad fit. This is the job for the **Oracle A1 box** (§11), or an HF Space. The Worker routes to it.

### The playback UX that makes it feel instant

- **Optimistic seek.** On click, immediately show the *frame image* you already have for that timestamp (you extracted it at ingest, it's in the sprite) while the video segment loads. The user sees the right frame in <100 ms even if playback starts at 800 ms. This is a big perceptual win for ~10 lines of code.
- **Preload the result under the cursor.** Hovering a search result starts fetching its manifest and first segment. By the time the click lands, playback is instant.
- **Play the moment, not the video.** Open at `t_start - 2s` (context lead-in), highlight the matched span on the scrubber, and offer "play just this moment" vs "play full video."
- **Never a black screen.** Your existing frame-loading design already has this instinct — scrubbing to an unloaded frame draws the nearest loaded one. Keep it.
- **Content-proportionate timeline** (§4.8): use the RAFT motion curve to size scrubber regions by density, so a dense 20-second passage isn't compressed into six pixels.

---

## 10. The knowledge layer and the roadmap engine

This is Output 3, the most ambitious part, and the part with no existing blueprint (§4.4). Here's how I'd build it so that it produces value early and gets better rather than being a two-year moonshot.

### The data model

```
Video ──has──▶ Chunk ──mentions──▶ Concept ──requires──▶ Concept
  │              │                    │
  │              ├──narrative (Qwen)  ├──wikidata_qid  (link out, don't absorb)
  │              ├──transcript span   ├──definition (best explanation found in corpus)
  │              └──frame refs        └──difficulty (inferred)
  │
  └──teaches──▶ Concept  (with a coverage score: how well does this video
                          actually explain this concept, not just mention it)
```

Two edge types do all the work:
- **`mentions`** — cheap, high-recall, extracted per chunk.
- **`requires`** — expensive, must be high-precision (§4.4: a false prerequisite is worse than a missing one).
- **`teaches`** — the important derived one: a video *mentions* fifty concepts and *teaches* two or three. Distinguishing these is what separates a roadmap from a tag cloud.

### Extraction pipeline

**Stage 1 — Concept mentions (per chunk, cheap).**
Local Qwen with a constrained prompt: "list the technical concepts explained or referenced in this passage; return JSON." Normalise aggressively — lowercase, lemmatise, then cluster near-duplicate concept strings with BGE embeddings (cosine > 0.9 → same concept). Without this normalisation you get "neural network," "neural networks," "NNs," and "neural nets" as four nodes and the graph is garbage.

**Stage 2 — Concept resolution (link out).**
Map each concept to a Wikidata QID where one exists. Per Balog & Kenter: **link, don't absorb.** This gives you free canonical names, aliases, and a coarse category hierarchy, and it means two videos using different vocabulary for the same thing collapse correctly.

**Stage 3 — Prerequisite extraction (expensive, precision-first).**
Three signals, and only emit an edge when at least two agree:

1. **Explicit textual cues.** "before we can talk about X, you need to understand Y", "assuming you know Y", "as we covered earlier". High precision, low recall. An LLM prompt tuned for this specific pattern.
2. **Corpus-order statistics.** If Y is consistently explained *before* X across multiple videos, and X's explanations reference Y but not vice versa, that's evidence of `Y → X`. This is a cheap, robust, purely statistical signal and it's the one the literature under-uses for personal corpora.
3. **Definitional dependency.** If the best definition found for X contains Y as a term, `Y → X` is likely.

**Stage 4 — DAG construction.**
Build the graph, then **break cycles** — this step is mandatory and easy to forget. Real extraction *will* produce cycles (A requires B requires A). Break them by removing the lowest-confidence edge in each cycle. Then a **topological sort is your learning path.**

**Stage 5 — Coverage scoring.**
For each `(concept, video)` pair, score how well the video actually teaches it: duration spent, whether it defines vs merely uses the term, presence of examples, whether OCR shows a slide titled with the concept. This is what lets the roadmap pick the *best* explanation for each step rather than the first one it finds.

### The roadmap output

For a goal concept, walk the DAG backwards to find prerequisites, topologically sort, and for each node pick the highest-coverage explanation in your corpus:

```
GOAL: "build a transformer from scratch"

  ├─ 1. Vectors & dot products        ✓ covered   → video #4421 @ 02:10  (8 min)
  ├─ 2. Softmax                       ✓ covered   → video #3390 @ 14:02  (4 min)
  ├─ 3. Attention mechanism           ✓ covered   → video #5567 @ 00:00  (22 min)
  │      ⤷ 3a. Q/K/V intuition        ✓ covered   → video #5567 @ 06:30
  │      ⤷ 3b. Why scaled             ⚠ weak      → video #2201 @ 31:15  (mention only)
  ├─ 4. Multi-head attention          ✓ covered   → video #5567 @ 18:40
  ├─ 5. Positional encoding           ✗ GAP       — no good explanation in corpus
  ├─ 6. Layer norm & residuals        ✓ covered   → video #1180 @ 09:55
  └─ 7. Full architecture assembly    ✓ covered   → video #5567 @ 41:20

  Estimated: 1 h 47 m across 5 videos.
  1 gap, 1 weak spot — [find videos on "positional encoding"]
```

**The gap detection is the killer feature and nobody has it.** A roadmap that tells you *what your collection is missing* turns the tool from a retrieval system into a curation advisor. It gives you a shopping list for what to collect next, which closes the loop with the capture plane.

### Making it useful before it's perfect

The failure mode of ambitious knowledge-graph projects is that they produce nothing until they're finished. Ship it in this order, each step independently useful:

1. **Concept tags on videos.** Just Stage 1 + 2. Immediately gives you faceted browsing and a concept cloud. Useful on day one.
2. **"Explain X" pages.** Aggregate every explanation of a concept across the corpus into one page, ranked by coverage. This is a genuinely great feature by itself and needs no prerequisite extraction at all.
3. **Related-concept navigation.** Co-occurrence edges only — cheap, and makes the graph browsable.
4. **Prerequisites, conservative.** Only high-confidence edges. Show them as "you might want to watch first" hints on a video page.
5. **Full roadmaps.** Once the DAG has enough density to be worth topologically sorting.
6. **Gap detection and collection suggestions.**

### Drop the external LLM dependency

Right now GraphRAG needs `VIOS_NIM_API_KEY` and silently does nothing without it. You already load a 7B VLM. Use it. Local Qwen for concept extraction and prerequisite detection is:
- free forever,
- not rate-limited,
- not a network dependency in a batch job,
- and good enough for this task, because the task is constrained extraction with a strict output schema, not open-ended reasoning.

Keep the API path as an optional quality upgrade for the final synthesis step, but nothing should silently no-op without it.

---

## 11. Infrastructure: enterprise-grade, single user, free

Here's the stack I'd actually run, with the reasoning.

```
┌────────────────────────────────────────────────────────────────────────┐
│  CLOUDFLARE  (always on, $0, no card required)                         │
│    Pages    — the SPA. Unlimited bandwidth. COOP/COEP headers for      │
│               SharedArrayBuffer (needed by sqlite-wasm-http).          │
│    Worker   — router; R2 signing; proxy to Oracle box; tiny JSON API.  │
│               100k req/day, 10 ms CPU each. Router only, no compute.   │
│    R2       — 10 GB, ZERO EGRESS. index.sqlite, vectors, sprites,      │
│               hot HLS proxies.                                         │
│    KV       — 1 GB. Config, manifest pointer. Careful: 1k writes/day.  │
│    Vectorize— optional. 5M vectors, 30M queried dims/mo.               │
│    Workers AI— 10k neurons/day. Query embedding without a WASM model.  │
│  ⚠ Free CDN ToS forbids hosting video + 512 MB per-file cache limit.   │
│    Serve video from R2/proxy, never through the CDN cache.             │
│  ⚠ Durable Objects have NO spending cap. Avoid, or guard hard.         │
└────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│  ORACLE CLOUD ALWAYS FREE  (always on, $0, card required for signup)   │
│    Ampere A1: 2 OCPU / 12 GB RAM  ⚠ HALVED on 15 June 2026 from 4/24   │
│    200 GB block storage (unchanged)                                    │
│    2× AMD micro instances (unchanged)                                  │
│    Runs: Telegram MTProto range-proxy (tgcrypto, multi-token)          │
│           heavier API endpoints, cross-encoder reranking               │
│           cron: nightly artifact sync, health checks                   │
│  ⚠ Pick a home region WITH A1 capacity — it cannot be changed later.   │
│  ⚠ Enforcement of the new cap is inconsistent; over-limit instances    │
│    may be shut down.                                                   │
└────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│  KAGGLE  (on demand, $0, ~30 GPU-h/week)                               │
│    2×T4, 15 GB RAM, ~57 GB disk, ~12 h sessions                        │
│    The only GPU. Batch ingest only. Never serves traffic.              │
└────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│  TELEGRAM  (permanent, $0, unlimited)                                  │
│    Channel A: videos (existing, immutable)                             │
│    Channel B: artifact bundles (new, private, chunked, manifested)     │
└────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│  GITHUB  (free)                                                        │
│    Code, Actions CI, Pages fallback, manifest backup                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Optional fourth leg: Hugging Face Spaces

Free CPU Basic gives **2 vCPU / 16 GB RAM** and sleeps after **48 hours** of inactivity — far more forgiving than Render's 15 minutes. That makes it viable for a "warm" Python service: the cross-encoder reranker, an on-demand embedding endpoint, or a fallback if the Oracle box is unavailable. ZeroGPU (H200, quota-based) is worth exploring for burst inference. Community GPU grants exist for qualifying open projects — if you ever open-source this, worth applying for.

### What I'd explicitly avoid

| Avoid | Why |
|---|---|
| Fly.io, Railway | No free tier as of 2026. Trial clocks, then a bill. |
| Render for the backend | 15-minute spin-down, 30–50 s cold wake. Kills the "instant" feeling. Fine for static. |
| Val Town | New free vals must be **public** as of May 2026. |
| Managed Postgres free tiers | They all expire, pause, or shrink. And you don't need Postgres (§5/G11). |
| Cloudflare Durable Objects | No spending cap on writes. One bad loop = a real bill. |
| Anything requiring a persistent WebSocket at the edge | Workers aren't built for it on the free tier. |

### The insight that makes "enterprise for free" work

Enterprise systems are expensive because of **concurrency, availability SLAs, and multi-tenancy**. You have exactly one user with no SLA. So:

- **No load balancing** — one user generates ~1 request per second at peak.
- **No horizontal scaling** — one box is a thousand times over-provisioned.
- **No hot-shard problem** — there's one shard.
- **No cache invalidation problem** — artifacts are immutable and content-addressed.
- **No auth system** — one Cloudflare Access rule, or a single bearer token.
- **No rate limiting** — the free tiers do it for you.

What's left is the part that actually makes a system feel enterprise-grade: **a great index, a great UI, and things that never break.** Those you can buy with care rather than money.

---

## 12. The website — UI, UX, and user flow

The bar you named: "make this current version look like nothing." Here's the design.

### Design principles

1. **Search is the homepage.** Not a library grid with a search box in the corner. A single input, centred, with recent queries below it. The corpus is too big to browse; browsing is the fallback, not the default.
2. **Text is the navigation surface** (§4.8). Transcripts, concepts, and OCR text are all clickable seek targets. This is the Descript/Reduct lesson and it's the highest-leverage UX decision available.
3. **The moment is the unit.** Results are moments with a thumbnail, a time range, a matched-text snippet, and a play button — not video cards you have to open and then hunt inside.
4. **Never a blank state.** Loading shows the nearest available frame. Empty search shows recent + suggested concepts. An error shows what still works.
5. **Keyboard first.** `⌘K` for search, `J/K` to move through results, `Space` to preview, `Enter` to open, `[`/`]` to nudge the moment boundaries. A single-user power tool should feel like Linear, not like a CMS.
6. **Every result explains itself** (§8).
7. **Dark by default**, high information density, no marketing chrome. You are the only user; you never need to be sold to.

### Screen 1 — Search (the home)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ⌘K   the part where he explains why attention is quadratic          [⏎] │
│       ─────────────────────────────────────────────────────────────────  │
│       intent: moment · concepts: attention, complexity · 4,102 videos    │
└──────────────────────────────────────────────────────────────────────────┘

  ┌────────────┐  Attention Is All You Need — Explained          94%  ▶
  │  [thumb]   │  05:47 → 06:33  ·  video #5567  ·  22 Mar 2026
  │  ▁▃█▅▂▁    │  "…the complexity grows with the square of the sequence
  └────────────┘   length, so doubling context quadruples the compute…"
                   matched: transcript · on-screen "O(n²)" · concept complexity

  ┌────────────┐  Transformers From Scratch, Part 3               81%  ▶
  │  [thumb]   │  31:15 → 31:52  ·  video #2201  ·  09 Jan 2026
  │  ▁▂▄▇▄▂    │  "…that's the quadratic bottleneck everyone talks about…"
  └────────────┘   matched: transcript · concept complexity

  [ 8 more moments ]   [ Ask this instead → ]   [ Show as timeline ]
```

Details that matter:
- **The tiny waveform/motion bar** under each thumbnail is the RAFT motion curve for that moment. It tells you at a glance whether it's a static talking head or something visual — surprisingly useful for deciding what to open.
- **"Ask this instead"** converts a search into a QA query without retyping. The two modes are the same index; the UI should make switching free.
- **"Show as timeline"** pivots the same results onto a chronological axis across all videos — see §13.
- **Hover preloads** the HLS manifest and first segment (§9).

### Screen 2 — Moment player

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ◀ back                          video #5567 · Attention Explained        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                        [ VIDEO — opens at 05:45 ]                        │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ ▶ ⏸  05:47 / 42:18   ━━━━━━━━━━█▓▓▓━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                ▲ matched moment      ● ● ●  ← other hits │
│                        [ sprite preview on hover ]                       │
├──────────────────┬───────────────────────────────────────────────────────┤
│ TRANSCRIPT       │  CONTEXT                                              │
│                  │                                                       │
│ 05:41 so if we   │  Concepts here                                        │
│ 05:44 look at    │   • self-attention        • computational complexity  │
│ 05:47 ▸the       │   • sequence length                                   │
│  complexity      │                                                       │
│  grows with the  │  On screen (OCR)                                      │
│  square of the◂  │   "Attention(Q,K,V) = softmax(QKᵀ/√d)V"               │
│ 05:52 sequence   │   "O(n²·d)"                                           │
│ 05:55 length…    │                                                       │
│                  │  Objects: whiteboard, person, marker                  │
│ [search inside]  │                                                       │
│                  │  Elsewhere in your library                            │
│                  │   3 other videos explain this → [see all]             │
│                  │                                                       │
│                  │  Prerequisites                                        │
│                  │   softmax · matrix multiplication → [roadmap]         │
└──────────────────┴───────────────────────────────────────────────────────┘
```

The Frame.io lessons applied:
- **Bidirectional linkage** — click a transcript line, seek; seek, and the transcript auto-scrolls and highlights.
- **Markers on the scrubber** for every other hit from your query, so you can bounce between matches within a video without going back to results.
- **Sprite preview on hover** (§4.8/§9).
- **Auto-pause when you start typing** in the notes/search-inside field.

The Descript lesson: transcript for structure, scrubber for precision. Both always visible.

### Screen 3 — Ask (QA)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ❓  What are all the different ways people in my library explain        │
│      backpropagation?                                                    │
└──────────────────────────────────────────────────────────────────────────┘

  Across 4,102 videos I found 7 distinct explanations:

  1. **The chain-rule derivation** — the most common (5 videos).
     Treats backprop as repeated application of the chain rule.
     Best version: #3391 @ 12:04 [▶] — works a full example by hand.

  2. **The computational-graph framing** (3 videos).
     Nodes and edges, forward values and backward gradients.
     Best version: #4820 @ 08:15 [▶]

  3. **The "blame assignment" intuition** (2 videos) …

  ─────────────────────────────────────────────────────────────────────────
  Sources: 11 moments across 9 videos      [show all]  [export as note]
  Confidence: high — 9 independent sources agree on the core mechanism
```

Non-negotiable: **every claim carries a clickable timestamp citation.** If a sentence can't be traced to a moment, it doesn't get written. That's what makes the QA layer trustworthy enough to actually rely on, and it's what makes it *your* library's answer rather than a generic LLM answer.

### Screen 4 — Knowledge / Roadmap

The graph is a means, not an end. Default view is a **roadmap** (a linear, checkable path). The force-directed graph is a secondary "explore" tab, because pretty graph visualisations are famously unusable for actually finding anything.

```
  Learn: build a transformer from scratch          [1h 47m across 5 videos]

  ●━━━━●━━━━●━━━━◐━━━━○━━━━●━━━━●
  1    2    3    3b   5    6    7

  ✓ 1. Vectors & dot products      #4421 @ 02:10   8m    [▶]
  ✓ 2. Softmax                     #3390 @ 14:02   4m    [▶]
  ✓ 3. Attention mechanism         #5567 @ 00:00  22m    [▶]
  ◐ 3b. Why scaled by √d           #2201 @ 31:15   —     weak coverage
  ✗ 5. Positional encoding             GAP — nothing in your library
       [ suggest search terms ]  [ mark as known ]
  ✓ 6. Layer norm & residuals      #1180 @ 09:55   6m    [▶]
  ✓ 7. Assembling the full model   #5567 @ 41:20  14m    [▶]
```

The gap row is the feature. It's actionable, it's honest, and it turns the tool into something that shapes what you collect next.

### Screen 5 — Library / browse

The fallback for when you don't know what you're looking for. Faceted by concept, date, channel, duration, and "has X on screen." Virtualised grid, preview-on-hover (play a 3-second loop from the most visually distinctive moment, not from t=0 — t=0 is almost always a title card).

### Screen 6 — System / health

Not an afterthought. This is what makes it feel like a real system rather than a script:
- Bundle version, age, size, and whether the local copy matches the remote manifest
- Ingest progress: queue depths, DLQ contents, throughput, ETA
- Per-model versions, and **which videos need reprocessing** because a model changed
- Search quality metrics from the eval harness (§5/G7) over time
- Storage: R2 usage, Telegram parts, what's hot vs cold

### The user flows

**Flow A — "I remember a thing" (the 80% case).**
`⌘K` → type → results in <300 ms → hover previews → click → opens at the moment → done. Target: **under 10 seconds from thought to frame.**

**Flow B — "I want to understand something."**
`⌘K` → question → Ask mode → synthesised answer with citations → click a citation → lands in the player at that moment with the transcript open → "elsewhere in your library" shows the other explanations.

**Flow C — "I want to learn something."**
Knowledge tab → type a goal → roadmap generates → work through it → checked steps persist → gaps become a collection to-do list.

**Flow D — "I found something great and want to keep it."**
Send to the Telegram bot → priority queue → processed within minutes → appears in search. The existing bot flow, unchanged; it's already the right design.

**Flow E — "I want this data somewhere else."**
Download `index.sqlite` → open in any SQLite client, Datasette, a notebook, or an MCP server for Claude. That's Output 4, satisfied by the artifact being the product.

---

## 13. Feature catalogue

Grouped by how much they'd change day-to-day use.

### Tier 1 — Would change how you use it

| Feature | What it is |
|---|---|
| **Cross-video moment reels** | Search returns moments across videos; select several and stitch into a single playable reel. This is Reduct's headline capability (§4.8) and it's the natural endpoint of moment retrieval. |
| **"Explain X" pages** | Every explanation of a concept across the whole corpus, ranked by coverage quality, on one page. Probably the single best feature in the list. |
| **Gap detection** | The roadmap tells you what your collection is *missing*. Turns the tool into a curation advisor. |
| **Visual similarity search** | DINOv2-powered "more frames like this." Finds same-person, same-setting, same-style content that text search structurally cannot. |
| **Search inside a video** | A second search box scoped to the open video. Obvious in hindsight, transformative in use. |
| **Query memory** | Learns from clicks (§8). Repeat searches become instant and telepathic. |

### Tier 2 — Substantial quality-of-life

| Feature | What it is |
|---|---|
| **Timeline view** | Pivot search results onto a chronological axis — "show me everything about diffusion models in the order it was published." Reveals how understanding of a topic evolved. |
| **Saved searches / smart collections** | A search that re-runs; new matching videos appear automatically. |
| **Annotations on moments** | Frame.io-style timestamped notes, searchable alongside transcripts. Your commentary becomes part of the index. |
| **Speaker identification** | Even coarse diarisation enables "what did the guest say" and improves chunking. |
| **Chapter auto-detection** | Scene + topic boundaries → auto-chapters on the scrubber. |
| **Export to Markdown/Obsidian** | A moment or roadmap exports as a note with a deep link back. |
| **CLI** | `vios search "…"`, `vios ask "…"`. Because sometimes a terminal is faster than a browser. |
| **MCP server** | Expose the corpus to Claude as a tool. Then you can ask Claude questions about your video library in any conversation. This is a genuinely powerful and cheap integration. |

### Tier 3 — Nice, later

| Feature | What it is |
|---|---|
| **Duplicate/near-duplicate detection** | Perceptual hashing + DINOv2 — find re-uploads and clips of the same source. |
| **Auto-generated summaries per video** | One paragraph, one screen, cached. |
| **"Watch later" queue with resume points** | Trivial to build once you have moment-level state. |
| **Diff two explanations** | Side-by-side comparison of how two videos explain the same concept. |
| **Sentiment / energy curve** | Audio-derived; useful for finding "the exciting part." |
| **Public share links** | Signed, expiring Worker URLs for a single moment. |
| **Mobile PWA** | The SPA is already static; a manifest and a service worker gets you an installable app almost free. |
| **Ambient mode** | Random moments from the corpus, shuffled. Genuinely good for rediscovery. |

### Anti-features — things I'd deliberately not build

- **Multi-user, roles, permissions.** You are the only user. Every hour spent here is an hour not spent on retrieval quality.
- **A real-time collaborative editor.** No collaborators.
- **A recommendation feed.** You collect deliberately; an algorithmic feed would work against that.
- **Video editing.** Descript exists. Your differentiator is retrieval and knowledge, not production.
- **A mobile-first responsive redesign of everything.** Make search + player work on mobile; leave the roadmap and system screens desktop-only. Ruthless scoping is what makes a solo project ship.

---

## 14. Out-of-the-box ideas from other domains

You explicitly asked for solutions from outside the domain. These are the cross-pollinations I think are most likely to actually work.

**1. Git's content-addressed object store → artifact bundles.**
Name every artifact part by its content hash. Deduplication becomes automatic, re-upload of unchanged parts becomes a no-op, corruption becomes detectable, and "which bundle version is this" becomes a single string. Git solved distributed content sync in 2005 and the solution transfers cleanly (§7).

**2. LSM trees → incremental bundle deltas.**
Base + deltas + periodic compaction. Exactly how LevelDB/RocksDB handle write amplification, and exactly the right shape for "I processed 200 more videos, don't make me re-upload 5 GB" (§7).

**3. Game engine LOD (level of detail) → the retrieval cascade.**
Games render distant objects with fewer polygons. Your search should score distant candidates with cheaper models: FTS over everything → dense over the top 10k → cross-encoder over the top 100 → late-interaction MaxSim over the top 20. Cost scales down as candidate count scales up. This is exactly what §8 describes, but LOD is the cleaner mental model for *why* it's correct.

**4. Ad-tech real-time bidding → your query understanding budget.**
RTB systems make a full auction decision in under 100 ms by hard-budgeting every stage. Adopt the discipline: query classification 50 ms, retrieval 100 ms, rerank 100 ms, render 50 ms. If a stage exceeds budget, it returns its best-so-far rather than blocking. Your search will never feel slow because it *can't*.

**5. Spaced-repetition scheduling (Anki/SM-2) → the learning roadmap.**
Your roadmap knows what you've watched. Add a review schedule: "you watched the attention explanation 3 weeks ago; here's the 90-second recap moment." You already have moment-level granularity, which is exactly what makes micro-review possible. This is a genuinely novel combination — nobody has spaced repetition over video moments from a personal corpus.

**6. Torrent piece-selection → Telegram chunk prefetch.**
BitTorrent's rarest-first and endgame-mode strategies solve "fetch many parts over a flaky channel efficiently." Your Telegram restore is the same problem. Endgame mode in particular (request the last few pieces from multiple sources simultaneously) maps onto multi-bot-token fetching.

**7. Database query planners → explainable search.**
`EXPLAIN` shows why a query chose an index. Your search should have the same: a debug panel showing each retriever's top results, their ranks, the RRF contributions, and the rerank deltas. When search feels wrong, you'll know *which stage* was wrong in 10 seconds instead of guessing (§8).

**8. CRDTs / local-first → the "integrate into any UI" goal.**
If you ever want multiple clients writing annotations (phone + desktop + CLI), the local-first playbook (an append-only op log, merged on sync) avoids ever needing a central write server. Your data is append-mostly, which is the easy case.

**9. Music streaming's gapless playback → moment reels.**
Spotify preloads the next track's first seconds during the current one. For a stitched moment reel across videos, preload the next moment's HLS segment while the current plays. Same trick, and it's what makes a reel feel like one video rather than a slideshow.

**10. Newspaper morgues / archival science → the coverage score.**
Archivists distinguish *mention* from *treatment* — a name appearing in a caption versus an article about that person. Your `mentions` vs `teaches` edge distinction (§10) is exactly this, and archival practice has a century of thinking about how to score it.

**11. Compiler intermediate representations → the ingest pipeline.**
Compilers pass through IRs so passes are independent and reorderable. Define an explicit intermediate artifact between each ingest stage (media → frames+audio, frames → notes+vectors, notes → chunks, chunks → concepts). Then you can rerun any single stage without redoing the others — which is what makes "I upgraded the embedding model" a 2-hour job instead of a 2-day one.

**12. CDN cache tiering → R2-hot / Telegram-cold.**
Standard CDN practice, applied to a free stack (§9). The unusual part is that your origin is a chat app.

**13. Observability's "golden signals" → search health.**
Latency, traffic, errors, saturation — but for retrieval: p50/p95 query latency, queries/day, zero-result rate, and rank-of-click distribution. The zero-result rate and rank-of-click distribution together are a near-perfect proxy for "is search getting better," and both come free from logging (§8).

**14. Wikipedia's "citation needed" → confidence surfacing.**
When the QA layer synthesises from a single weak source, say so. An explicit low-confidence marker is far more useful than a confident wrong answer, and it costs one line of UI.

---

## 15. Repository reorganisation

Current state: **37 tracked files, flat at the root**, mixing entrypoints, libraries, HTML, and dead code. It reads like a notebook that grew — which it is. Here's the target.

### Proposed layout (uv workspace, per §4.9)

```
vios/
├── README.md                       # what it is, how to run it, in 60 seconds
├── pyproject.toml                  # VIRTUAL ROOT — no [project] table
├── uv.lock                         # one lock for the whole workspace
├── .gitignore  .gitattributes  .editorconfig
├── Makefile                        # make ingest / make serve / make eval
│
├── docs/
│   ├── ARCHITECTURE.md             # the current, accurate architecture
│   ├── MASTERPLAN.md               # this file
│   ├── DECISIONS/                  # ADRs — one file per irreversible choice
│   │   ├── 0001-telegram-as-storage.md
│   │   ├── 0002-sqlite-as-canonical-store.md
│   │   ├── 0003-no-expandable-segments.md
│   │   └── 0004-one-gpu-process.md
│   ├── SCHEMA.md                   # the DB contract — this is your API docs
│   └── RUNBOOK.md                  # what to do when it breaks
│
├── packages/                       # shared libraries — never import an app
│   ├── vios-core/                  # config, logging, paths, errors, types
│   │   └── src/vios_core/
│   ├── vios-store/                 # SQLite schema, migrations, queries, vectors
│   │   └── src/vios_store/
│   ├── vios-queue/                 # the current queue_manager.py, unchanged
│   │   └── src/vios_queue/
│   ├── vios-telegram/              # client, chunked upload/download, range proxy
│   │   └── src/vios_telegram/
│   ├── vios-media/                 # ffmpeg: frames, proxies, HLS, sprites, VTT
│   │   └── src/vios_media/
│   ├── vios-models/                # ONE model registry, ONE loader, ONE owner
│   │   └── src/vios_models/
│   ├── vios-index/                 # embedding, FTS, RRF, rerank, temporal fusion
│   │   └── src/vios_index/
│   └── vios-knowledge/             # concepts, prerequisites, DAG, roadmaps
│       └── src/vios_knowledge/
│
├── apps/
│   ├── ingest/                     # the Kaggle batch job (was boot.py + workers)
│   │   └── src/ingest/
│   ├── api/                        # FastAPI — the local/Oracle serve process
│   │   └── src/api/
│   ├── bot/                        # Telegram bot (was part of omni_engine.py)
│   │   └── src/bot/
│   └── cli/                        # vios search / ask / sync / eval
│       └── src/cli/
│
├── web/                            # the SPA — its own toolchain, not Python
│   ├── src/
│   │   ├── routes/                 # search · player · ask · knowledge · system
│   │   ├── lib/                    # sqlite-http client, player, RRF in JS
│   │   └── styles/
│   ├── package.json
│   └── vite.config.ts
│
├── workers/                        # Cloudflare Workers
│   └── edge/                       # router, R2 signing, range proxy
│
├── eval/                           # the accuracy harness (§5/G7)
│   ├── golden/queries.jsonl        # hand-labelled query → answer set
│   └── run.py                      # Recall@k, MRR, nDCG — run on every change
│
├── notebooks/
│   └── kaggle_launcher.ipynb       # the one-cell launcher
│
├── scripts/
│   ├── setup_kaggle.sh
│   └── bundle_restore.sh
│
└── tests/
    ├── unit/
    └── integration/
```

### The mapping from today

| Today | Goes to |
|---|---|
| `config.py` | `packages/vios-core/` |
| `logger.py` | `packages/vios-core/` |
| `queue_manager.py` | `packages/vios-queue/` (essentially unchanged — it's good) |
| `boot.py` | `apps/ingest/` |
| `frame_worker.py` | `packages/vios-media/` + `apps/ingest/` |
| `model_manager.py` + `omni_models.py` | **merge** → `packages/vios-models/` |
| `omni_db.py` + the SQLite code in `v17_backend.py` | **merge** → `packages/vios-store/` |
| `omni_engine.py` (51 KB!) | **split** → `apps/bot/`, `apps/ingest/`, `packages/vios-index/` |
| `omni_prompts.py` | `packages/vios-knowledge/prompts/` |
| `ui_server.py` + `v17_backend.py` | `apps/api/` |
| `v17_ui.html` (67 KB) + `omni_dashboard.html` | **rewrite** → `web/` |
| `admin_backend.py`, `admin_ui.html`, `main_ui.html`, `local_preview.py`, `push.py`, `test_harness.py`, `public/placeholder-*` | **delete** |

### The rules that keep it clean

1. **Apps depend on packages. Packages never depend on apps. Sibling apps never import each other.** This is the rule; everything else is detail.
2. **No file over ~500 lines.** `omni_engine.py` at 51 KB and `v17_ui.html` at 67 KB are both past the point where a reader can hold them in their head. Splitting them is most of the perceived "professionalism" win.
3. **One `uv.lock` at the root.** Every app resolves against identical dependency versions.
4. **The virtual root has no `[project]` table** — just `[tool.uv.workspace]`.
5. **Every irreversible decision gets an ADR.** You already write excellent explanatory comments (the `PYTORCH_CUDA_ALLOC_CONF` note, the `_gpu_dtype` docstring). ADRs are those comments, promoted to where a reader will find them.
6. **`docs/SCHEMA.md` is the API documentation.** Since the DB is the product (Output 4), its schema is the public contract.

### On the "professional" feeling specifically

Three things do most of the work, and none of them are the directory tree:

- **A README that shows the thing in the first screen.** A screenshot or GIF of a search returning a moment, then three commands. Nobody reads past the fold.
- **CI that runs on every push.** Even just ruff + a schema check + the eval harness. A green badge signals "this is maintained" more than any amount of structure.
- **`docs/DECISIONS/`.** A repo that explains *why* reads as engineered. A repo that only explains *what* reads as generated.

### A migration warning

Do this in **one commit, mechanically, with no behaviour changes.** Move files, fix imports, verify it still runs, commit. Then make changes in subsequent commits. Mixing a reorganisation with a refactor produces a diff nobody — including you in three months — can review.

---

## 16. Phased roadmap

Each phase is independently valuable and leaves the system working. Nothing here requires finishing the next thing to be useful.

### Phase 0 — Stop the bleeding *(days)*
The current system works; make it not waste resources.
- One GPU process; merge `model_manager` + `omni_models` model loading. Kills G5 and G6.
- Apply the fatal-CUDA-context policy to the CV loop (O3).
- Replace the NIM dependency with local Qwen (O4).
- Delete dead files (G13).

### Phase 1 — Permanence *(1–2 weeks)*
The most important phase. Nothing else matters until artifacts survive.
- Define the bundle format and manifest (§7).
- Implement seal → chunk → upload to a private artifact channel (wrap rclone/teldrive rather than writing MTProto upload by hand).
- Implement restore-on-boot, transparently, so no other module changes.
- Verify: kill a session mid-ingest, start a fresh one, confirm it resumes with zero reprocessing.

### Phase 2 — Consolidation *(2–3 weeks)*
- Collapse Postgres + Qdrant + Neo4j into SQLite + sidecar vectors (G11).
- One canonical schema, with `schema_version` and forward-only migrations (G12).
- One ingest pipeline; one set of tables (G3).
- Build Stage B — frame and chunk embeddings written to the canonical store (G4).
- Repository reorganisation (§15), as a single mechanical commit.

### Phase 3 — Search quality *(2–3 weeks)*
- **Build the eval harness first** (G7). 100–200 golden queries. Get a baseline number.
- Hybrid retrieval + RRF (§8). Measure.
- Cross-encoder reranking. Measure.
- Temporal fusion into moments. Measure.
- Query understanding and intent classification. Measure.
- Explainability panel (which retriever matched, and why).

### Phase 4 — The serve plane *(3–4 weeks)*
- R2 sync of the artifact bundle.
- Cloudflare Worker: router + R2 signing.
- Oracle A1 box: Telegram range-proxy (tgcrypto, multi-token) + reranker.
- Media artifacts at ingest: HLS proxies, sprite sheets, WebVTT (§9).
- The SPA: search screen and player screen only. Ship those two before building anything else.

### Phase 5 — Knowledge *(ongoing)*
Ship in the order from §10 — each step is independently useful:
concept tags → "Explain X" pages → related-concept nav → conservative prerequisites → roadmaps → gap detection.

### Phase 6 — Polish and reach
- Ask/QA screen with citation-first synthesis.
- Moment reels, saved searches, annotations.
- CLI and MCP server (cheap, high leverage).
- PWA.
- Query-memory learning loop and embedding fine-tuning (§8).

### If you only do three things

1. **Phase 1** — permanence. Without it, nothing accumulates.
2. **The eval harness** (start of Phase 3) — without it, "improve search" is unfalsifiable.
3. **Hybrid + RRF + rerank** — the single largest accuracy jump available, and you already have both halves built, just unfused.

---

## 17. Risks, failure modes, and what I'd worry about

Being straight with you about what could go wrong.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Telegram removes the artifact channel** | Low but real | Severe | Bundle must be reproducible from the video corpus. Keep the manifest in GitHub. Consider a second backing store (R2 for the DB, which is the irreplaceable part — the media can be regenerated). |
| **Telegram rate limits / FLOOD_WAIT during restore** | Medium | Annoying | Multiple bot tokens, exponential backoff, resumable part downloads. |
| **Oracle reclaims the A1 instance** | Medium — they cut the free tier in half in June 2026 with no announcement | Moderate | Design so the Oracle box is *optional*: everything critical works from Cloudflare + browser; the box only accelerates. Keep an HF Space as a warm standby. |
| **R2 free tier changes** | Low | Moderate | 10 GB is the constraint already; the tiering design (§9) means R2 is a cache, not a store. |
| **Kaggle changes GPU quotas** | Medium | Moderate | Compute is already stateless and disposable. Colab, HF ZeroGPU, or a rented spot GPU are drop-in — that's the point of the four-plane split. |
| **The knowledge graph produces garbage** | **High** | Moderate | This is the least-proven part (§4.4 — nobody has done it on noisy personal corpora). Mitigate by shipping in the order from §10, where the first three steps are useful even if prerequisites never work. Be conservative on edges. |
| **Scope collapse — too much at once** | **High** | Severe | This is the real risk. The plan is deliberately phased so each stage ships something usable. Resist building the roadmap engine before search is good. |
| **The reorganisation breaks working code** | Medium | Moderate | One mechanical commit, no behaviour changes, verified before any refactor. |
| **Embedding model upgrade invalidates the index** | Certain, eventually | Moderate | The manifest records model versions per artifact (§7/G12). Reprocessing is then a targeted query, not a full rebuild. |

### The thing I'd worry about most

Not any single technical risk — it's that **the system becomes too complex for one person to hold.** Right now it's five datastores, four processes, two pipelines, and 37 files. Every item in §6 that *removes* something (one store, one GPU process, one pipeline) is worth more than any item that adds capability, because it buys you the headroom to add capability later without the whole thing becoming unmaintainable.

Bias toward deletion. It's the only thing that scales for a solo project.

---

## 18. Open questions for you

Things I need your call on, because they change the design and I can't infer them:

1. **Corpus size, now and target.** How many videos, and what's the realistic ceiling — 5k, 50k, 500k? Under ~50k, brute-force vector search in NumPy beats an ANN index on accuracy and simplicity. Over that, the calculus changes.

2. **Typical video length.** 30-second clips and 2-hour lectures need very different chunking, sprite intervals, and proxy budgets.

3. **Do you have Telegram Premium?** 4 GB per file vs 2 GB changes the chunking math and roughly halves the part count.

4. **Is the corpus mostly one domain** (e.g. ML/tech tutorials) or genuinely broad? A single domain makes the concept graph dramatically easier and makes embedding fine-tuning very effective.

5. **Are you willing to run one always-on box** (Oracle A1, free but requires a card for signup and some sysadmin), or should the design assume Cloudflare-only? This is the biggest architectural fork in §11.

6. **How much do you care about mobile?** Full parity is significantly more work than "search and player work on a phone."

7. **Public or private repo?** If public, an HF community GPU grant becomes plausible, and the code-quality bar rises.

8. **Is the Telegram bot still your primary capture path**, or would you rather capture from a browser extension / share sheet?

None of these block starting Phase 0 or Phase 1.

---

## 19. Sources

### Video RAG, retrieval, and evaluation
- Video RAG pipeline architecture and clip-vs-video processing — [Twelve Labs / video RAG architecture research](https://www.twelvelabs.io/)
- Hybrid search, BM25 + dense, Reciprocal Rank Fusion (k=60), two-stage retrieval and cross-encoder reranking; T2-RAGBench comparisons; RAG evaluation thresholds (RAGAS, DeepEval, Promptfoo)
- ColPali / ColQwen late interaction: SigLIP-So400m encoder, 1024 patch embeddings per page (32×32 over 448×448) at 128-d, MaxSim scoring from ColBERT; ColQwen2 (Qwen2-VL, Apache 2.0); Video-ColBERT (CVPR 2025); CLaMR; ColQwen-Omni

### Knowledge graphs and learning paths
- Prerequisite relation extraction and learning-path generation; KnowLP / EDU-Graph-RAG (Knowledge Structure Graph Generation + PPO "P-agent" sequencing); topological sort of the prerequisite DAG
- Personal Knowledge Graphs — Balog & Kenter (2019), "link to public KGs rather than integrate"

### Telegram as storage and streaming
- [teldrive](https://github.com/tgdrive/teldrive) — chunked upload, Postgres metadata, rclone `teldrive` backend (512 MB default chunk), WebDAV
- TG-FileStreamBot (Go / gotgproto, round-robin over up to 50 bot tokens), stremio-telegram-debrid (Pyrogram + tgcrypto + Uvicorn, HTTP 206 seeking), Telegram-Stremio, webbridgebot
- MTProto `upload.getFile` offset/limit constraints (offset divisible by 4096, limit a power of two); Bot API `getFile` 20 MB cap; tgcrypto as the AES-IGE throughput requirement

### Free infrastructure
- [Cloudflare free tier limits](https://developers.cloudflare.com/workers/platform/limits/) — Workers 100k req/day & 10 ms CPU; R2 10 GB with zero egress; D1 5 GB; Vectorize; KV; Durable Objects; Queues; Workers AI. CDN ToS video prohibition and 512 MB file cache limit.
- Oracle Cloud Always Free — Ampere A1 halved to 2 OCPU / 12 GB on 15 June 2026: [Oracle Free VPS Review 2026](https://space-node.net/blog/oracle-vps-free-tier-review-2026)
- [Render vs Railway vs Fly.io pricing 2026](https://dev.to/pavel-hostim/render-vs-railway-vs-flyio-pricing-compared-2026-2e5p) · [Fly.io free tier 2026](https://www.saaspricepulse.com/blog/flyio-free-tier-2026) · [Railway vs Fly.io vs Render](https://www.saaspricepulse.com/compare/railway-vs-flyio-vs-render) · [Solo developer comparison](https://devtoolpicks.com/blog/railway-vs-render-vs-fly-io-solo-developers-2026)
- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) · [Using GPU Spaces](https://huggingface.co/docs/hub/en/spaces-gpus)
- [Deno Deploy pricing and limits](https://docs.deno.com/deploy/pricing_and_limits/) · [Deno Deploy free tier](https://www.freetiers.com/directory/deno-deploy)
- [Val Town limits](https://www.val.town/limits) · [Val Town pricing](https://www.val.town/pricing) · [Changelog, May 2026 — public-vals restriction](https://blog.val.town/changelog-05262026)
- [Best free hosting platforms 2026 — Appwrite](https://appwrite.io/blog/post/free-hosting-platform) · [Awesome Web Hosting 2026](https://github.com/iSoumyaDey/Awesome-Web-Hosting-2026) · [Best free hosting — TechPlained](https://www.techplained.com/best-free-hosting) · [Free cloud deploy platforms ranked](https://snapdeploy.dev/blog/free-cloud-deployment-platforms-2026-comparison)
- Self-hosted PaaS: [Coolify vs Dokku vs CapRover](https://ownkube.io/blog/self-hosted-paas-comparison-2026) · [Dokploy vs Coolify](https://introserv.com/blog/dokploy-vs-coolify-complete-comparison-of-the-best-self-hosted-paas-platforms-for-vps-and-dedicated-servers-2026/) · [Best self-hosted PaaS 2026](https://deployhandbook.com/best/self-hosted-paas)

### Database over HTTP
- `sql.js-httpvfs` (phiresky) — SQLite VFS over HTTP Range, three read heads with prefetch ramping, index-or-full-download constraint
- `sqlite-wasm-http` — successor, requires COOP/COEP for SharedArrayBuffer
- `sqlite3vfshttp` (Go) — presigned S3/R2 URLs
- DuckDB-WASM + Parquet — analytical scans, no persistent indexes

### Video UI/UX
- [Video player UI: best examples, patterns & UX tips — Eleken](https://www.eleken.co/blog-posts/video-player-ui)
- [Build scrub-bar thumbnail previews with FFmpeg and a WebVTT sprite](https://dev.to/masonwritescode/build-scrub-bar-thumbnail-previews-with-ffmpeg-and-a-webvtt-sprite-3ei2)
- [Storyboard thumbnails: the scrub-bar preview your players are missing](https://nikodev1.medium.com/storyboard-thumbnails-the-scrub-bar-preview-your-players-are-missing-8ee4182ea5f4)
- [Bitmovin — WebVTT based thumbnails](https://developer.bitmovin.com/playback/docs/webvtt-based-thumbnails) · [Thumbnail preview support](https://developer.bitmovin.com/playback/docs/thumbnail-preview-support)
- [Radiant Media Player — preview thumbnails](https://www.radiantmediaplayer.com/docs/latest/preview-thumbnails.html)
- [hls.js issue #2662 — "Does hls.js support thumbnail seeking?" (open since 2020)](https://github.com/video-dev/hls.js/issues/2662)
- [video_sprites (jronallo) — ffmpeg + ImageMagick sprite generator](https://github.com/jronallo/video_sprites/wiki)
- [Videojs VTT thumbnails showcase](https://www.nuevodevel.com/nuevo/showcase/thumbnailsvtt)
- [Descript business breakdown — Contrary Research](https://research.contrary.com/company/descript) · [Best transcript-based video editing tools 2026](https://scriptcut.io/blog/best-transcript-based-video-editing-tools) · [How to fine-tune video edits in Descript](https://www.optiwebdesign.com/2025/08/29/how-to-fine-tune-video-edits-in-descript/)
- [Reduct vs Descript 2026](https://vidnotes.app/blog/157-Reduct-vs-Descript-2026-Which-Video-Transcription-Tool-Is-Right) · [Reduct — transcription software](https://reduct.video/blog/transcription-software-for-video/)
- [Frame.io — commenting on your media](https://help.frame.io/en/articles/9105251-commenting-on-your-media) · [Frame.io review — Videomaker](https://www.videomaker.com/review-frame-io/) · [Frame.io review after a year — CineD](https://www.cined.com/frame-io-review-after-a-year-of-use/) · [Frame.io video workflows](https://frame.io/enterprise/video-workflows)
- [Video asset management — Cloudinary](https://cloudinary.com/guides/digital-asset-management/video-asset-management) · [Video asset management — Canto](https://www.canto.com/glossary/video-asset-management/) · [Video asset management software 2026 — Filestage](https://filestage.io/blog/video-asset-management-software/)
- [Context-aware video searching (patent) — content-proportionate timelines](https://patents.justia.com/patent/20210098026)
- [Incorporating timeline scrubbing in a custom editor](https://palospublishing.com/incorporating-timeline-scrubbing-in-a-custom-editor/) · [Playing with video scrubbing animations on the web](https://www.ghosh.dev/posts/playing-with-video-scrubbing-animations-on-the-web/)

### Repository structure
- uv workspaces: `apps/` + `packages/`, virtual root (no `[project]` table), single `uv.lock`, strict dependency direction
- Apache Airflow on uv workspaces — 120+ distributions, 700+ dependencies, 3,600+ contributors (Jarek Potiuk, FOSDEM 2026)
- `prek` — monorepo-aware pre-commit replacement; Una — building uv workspaces into distributables

---

## Closing thought

The two halves of this system each worked before you merged them, and the merge is now working too. That's the hard part done — you have a functioning multimodal video intelligence pipeline running on free hardware, which most people never get to.

What's missing isn't capability. It's **permanence** (nothing survives the session), **unity** (two pipelines, two stores, one corpus), and **a face** (no serve plane). Those three are the whole gap between "an impressive notebook" and "a system I use every day for the next decade."

And the good news in the research: almost everything you need already exists as a free, proven technique. Telegram-as-storage has working implementations. SQLite-over-HTTP-range means your database can be queried from a browser with no server at all. RRF and cross-encoder reranking are the biggest accuracy wins available and you already have both retrievers built. Cloudflare's zero-egress R2 removes the one cost that normally kills personal video projects.

The genuinely novel part — prerequisite ordering over a noisy personal video corpus — has no blueprint. That's the part worth taking your time with, and the part that would be worth writing up if it works.

