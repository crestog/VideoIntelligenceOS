# Atlas

A search engine for the video archive.

Atlas reads the database bundles the harvester posts to the Telegram channel,
merges every one of them into a single local database, and serves a site where
you type a sentence and get the videos that contain that moment — playable
instantly, with everything the database knows about them alongside.

It is a **reader**. It has no harvester, no GPU worker and no queue. It never
writes to the channel. Running it cannot damage anything upstream, and running
it twice is the same as running it once.

```bash
python atlas_boot.py
```

---

## What happens when you start it

1. **Probe.** One call to the Bot API confirms the channel is reachable and
   returns the pinned message. The exporter pins every manifest, so the newest
   bundle is known before any scanning starts.
2. **Scan.** Bots cannot list channel history — `messages.getHistory` returns
   `BOT_METHOD_INVALID`. So Atlas finds the head message id by posting a dot,
   reading its id and deleting it, then walks backwards in batches of 190
   through `getMessages`, collecting every manifest it passes.
3. **Import.** Each bundle's parts are downloaded, joined, decompressed and
   replayed into SQLite. Every bundle is a full snapshot, so later ones
   overwrite the rows they share and contribute the rows they added. Import is
   a merge, not a restore.
4. **Index.** Every text column in every table is found by inspection and
   turned into timestamped passages. FTS5 is built immediately; the vector
   index builds on a background thread.
5. **Serve.** The site is up from second one and reports its own progress. You
   can search a partially-imported archive while the rest is still arriving.

Steps 2–4 re-run on demand from the **Sources** tab.

---

## The four tabs

**Search** — type a sentence. Results are videos, each showing a *moment
ribbon*: the reel's full duration as a strip, with the passages that matched lit
as coloured blocks at their real timestamps. Click a block and the player opens
at that second.

**Library** — every video, sorted and filtered by what the data actually
supports. Filters are built from the values present, so no filter is ever
offered that returns nothing.

**Data** — the raw database. Every table, every column, with the role Atlas
inferred for it (key / timestamp / searchable content), and a row browser for
each one. This is where you check what a bundle actually contained.

**Sources** — which bundles were imported, which columns feed search, whether
the channel is reachable, how large the video cache is, and a live log.

---

## How the search works

Two retrievers always run, and their results are fused.

**BM25** over the passage text, through SQLite's FTS5 with a Porter stemmer, so
`running` finds `ran`. Exact on names, numbers and jargon. Blind to paraphrase.

**Dense vectors** — `bge-small-en-v1.5`, 384 dimensions, CLS-pooled and
L2-normalised. Understands that "someone cooking pasta" and "a man boiling
spaghetti" are the same request. Hopeless at `@nikocado`.

They are fused with **Reciprocal Rank Fusion**:

```
score(passage) = Σ  1 / (60 + rank_from_that_retriever)
```

RRF discards the scores and keeps only the ranks, which sidesteps the fact that
BM25 scores are unbounded and cosine similarities live in [-1, 1] — you cannot
add them without knowing each retriever's distribution, and that distribution
changes per query. A passage ranked 1st by one retriever and 40th by the other
beats one ranked 15th by both, which is what you want: strong evidence of
either kind is worth more than being vaguely plausible to both.

After fusion, three adjustments:

- **Source weight.** A narrative Qwen wrote while watching the video is
  stronger evidence than an object label. Weights live in `config.SOURCE_WEIGHT`.
- **Phrase bonus.** Neither retriever rewards word order. A passage that
  literally contains the typed phrase gets a small, capped boost.
- **Grouping.** Passages are grouped into videos. A video scores its best
  moment plus a sharply damped contribution from the rest, with a bonus when
  the matches come from *different kinds* of evidence — two sources agreeing is
  a stronger signal than one source twice.

### Why it is fast

There is no vector database. The whole matrix is a flat `float32` file
(200k passages ≈ 307 MB) loaded into RAM once, and a query is a single
`(N,384) @ (384,)` matmul — memory-bandwidth-bound, a few milliseconds, and
*exact*. `argpartition` selects the top-k in O(N) without sorting the rest. An
ANN index would add a service, a build step, a tuning knob and an approximation
in exchange for making a fast thing slightly faster. At ten million passages
that trade flips; at this size it does not.

On top of that: a per-query LRU cache on the server, a result cache in the
browser, and a precomputed `video_index` table so rendering a page of results
costs one query instead of one per card.

---

## Why playback is instant

Nothing makes a 20 MB transfer instant, so the trick is to have already done it.

- **Local first.** If Atlas runs on the machine that harvested the reel, the
  file is on disk and no network is touched at all. This is the usual case on
  Kaggle.
- **Speculative prefetch.** The search handler starts downloading the top
  results *before anything is clicked*. By the time your eye reaches the first
  card, the file behind it is usually resident.
- **Hover intent.** Cards request their video on `pointerenter` and on keyboard
  focus. The 200–400 ms between deciding and clicking is free head start.
- **Byte-range serving.** Playback goes through a hand-written HTTP 206
  responder, so seeking works, playback starts on the first chunk, and a
  partially-downloaded file serves the bytes it already has.
- **Media fragments.** Clicking a moment 40 s in appends `#t=40`, so the
  browser requests that range first rather than the 40 s before it.

The cache is LRU by access time against `ATLAS_VIDEO_CACHE_GB` (default 12) and
lives on the scratch disk, which is wiped between sessions anyway. Nothing in it
is precious — every file is re-fetchable from the channel.

---

## Why a changed database does not break it

No file outside `reflect.py` names a table or a column. Everything is inferred
at runtime from the live schema:

| What | How it is decided |
|---|---|
| Which column identifies a video | name shape *and* a check that the table is not a lookup table |
| Which columns are timestamps | name shape, numeric type, plausible range |
| Which columns hold searchable text | text type, not an id/path/hash/status field |
| What kind of evidence a column is | table and column name mapped to narrative / speech / visual / ocr / caption / meta |
| Which tables join to which | foreign-key-shaped columns pointing at another table's key |

A hash of the whole schema is stored with the index. When a bundle arrives whose
schema differs — a new column, a new table, a dropped field — the hash changes
and the index rebuilds itself, picking the new column up as searchable text with
no code change. Tables Atlas has never seen appear in the **Data** tab and in a
video's record automatically.

The join key across all of it is the Telegram message id. Postgres writes
`tg1234`, `lake.db` writes `1234`, a manifest writes `"1234"` — all three
normalise to the digits, which is also what makes any video fetchable from the
channel without a mapping table.

---

## Environment

Credentials are read from the environment and **have no fallback values**. An
earlier revision of the harvester carried a live bot token as a default and
published it to a public repository; that rule now applies to every program
here.

| Variable | Required | What it does |
|---|---|---|
| `VIOS_BOT_TOKEN` / `TELEGRAM_BOT_TOKEN` | yes | reads the channel, downloads parts under 20 MB |
| `VIOS_API_ID` / `TELEGRAM_API_ID` | for large files | MTProto — parts and videos over 20 MB |
| `VIOS_API_HASH` / `TELEGRAM_API_HASH` | for large files | as above |
| `ATLAS_CHANNEL_ID` | no | defaults to the VIOS channel |
| `NGROK_AUTH_TOKEN` | on Kaggle | public URL for the site |
| `ATLAS_HOME` | no | where the database lives (default `/kaggle/working/atlas`) |
| `ATLAS_CACHE_DIR` | no | where videos are cached (default `/kaggle/temp/atlas`) |
| `ATLAS_VIDEO_CACHE_GB` | no | cache budget, default 12 |
| `ATLAS_EMBED_DEVICE` | no | `auto` / `cpu` / `cuda` — `auto` refuses the GPU unless it has room |

With only a bot token, Atlas works: it imports bundles whose parts are under
20 MB and plays videos under 20 MB. With MTProto it handles everything.

Without `sentence-transformers` or `numpy` it still runs — search falls back to
lexical-only and says so in the interface rather than failing.

---

## Flags

```bash
python atlas_boot.py --port 7000     # listen somewhere else
python atlas_boot.py --no-tunnel     # localhost only, no ngrok
python atlas_boot.py --no-scan       # serve what is imported, skip the channel
```

---

## Layout

```
atlas/
  config.py     paths, credentials, retrieval constants
  tgchannel.py  the channel: Bot API + MTProto, one interface
  pgdump.py     replays a plain pg_dump into SQLite — no PostgreSQL needed
  ingest.py     scanning the channel, importing bundles as merges
  reflect.py    every assumption about schema shape, in one file
  index.py      passages, FTS5, video_index, the vector file
  encoder.py    bge-small, CLS-pooled, degrades to nothing gracefully
  search.py     BM25 + dense + RRF + grouping
  media.py      resolve, prefetch, 206 range serving, posters, eviction
  server.py     the API surface
  web/          index.html · atlas.css · atlas.js
atlas_boot.py   the one command
```

Bundles carry the database, not the media. Frames were never shipped, so the
image vectors from the harvester's Qdrant are not here and could not be used if
they were. Atlas builds its own **text** index from the narratives, transcripts
and frame notes that *are* in the bundle — which is why search works on the
words in the archive from the moment the first bundle lands.
