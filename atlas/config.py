"""
Atlas configuration.

Atlas is a reader. It owns no harvester, no GPU worker and no queue — it pulls
finished bundles out of the Telegram channel and serves what is inside them.
That is why this file shares nothing with the harvester's config: the two
programs have different lifetimes, different disks and different failure modes,
and a shared config would tie Atlas's startup to services it never uses.

Storage, in the order it matters:

  ATLAS_HOME   /kaggle/working/atlas   survives the session, small (~19.5 GB
                                       shared quota). Holds atlas.db, the
                                       vector file and the manifest cache —
                                       everything that would otherwise force a
                                       re-download and re-embed.
  CACHE_DIR    /kaggle/temp/atlas      dies with the session, large. Holds
                                       video files and posters, which are
                                       re-fetchable from the channel in
                                       seconds. Nothing here is precious.

Override either with ATLAS_HOME / ATLAS_CACHE_DIR.
"""

import os

# ── Disks ─────────────────────────────────────────────────────────────────
_ON_KAGGLE = os.path.isdir("/kaggle/working")
_HERE = os.path.dirname(os.path.abspath(__file__))

ATLAS_HOME = os.environ.get("ATLAS_HOME") or (
    "/kaggle/working/atlas" if _ON_KAGGLE
    else os.path.join(os.path.dirname(_HERE), "atlas_data"))

CACHE_DIR = os.environ.get("ATLAS_CACHE_DIR") or (
    "/kaggle/temp/atlas" if os.path.isdir("/kaggle/temp")
    else os.path.join(ATLAS_HOME, "cache"))

DB_PATH      = os.path.join(ATLAS_HOME, "atlas.db")
BUNDLE_DIR   = os.path.join(ATLAS_HOME, "bundles")     # downloaded parts
VECTOR_PATH  = os.path.join(ATLAS_HOME, "moments.vec")  # flat float32 matrix
VECTOR_META  = os.path.join(ATLAS_HOME, "moments.vec.json")
# One flat matrix per frame-vector space, written the same way and read the same
# way. Unlike `moments.vec` these survive a reindex untouched: they are keyed by
# `(video_key, frame_idx)`, which no rebuild reassigns, where moment vectors are
# keyed by `moments.id`, which every rebuild does.
FRAME_VEC_DIR = os.path.join(ATLAS_HOME, "frames")
SESSION_DIR  = os.path.join(ATLAS_HOME, "session")
VIDEO_CACHE  = os.path.join(CACHE_DIR, "video")
POSTER_CACHE = os.path.join(CACHE_DIR, "poster")
WEB_DIR      = os.path.join(_HERE, "web")

# Where the capture plane put the files, and where it wrote its ledger. Atlas
# does not own either one — it reads them, read-only, to settle identity: the
# ledger is the only place that records "message 38 is the reel DZDNyKgv70R",
# and the media folder is where two byte-identical files under different names
# can be caught by hashing them. Both are optional; on Kaggle after a restore
# there may be no ledger at all, and identity then falls back to what the
# `video` rows themselves say. Overridable because the desktop app keeps its
# data in one folder and Kaggle keeps it in three.
MEDIA_DIR    = os.environ.get("ATLAS_MEDIA_DIR") or os.path.join(
    ATLAS_HOME, "media")
LEDGER_PATH  = os.environ.get("ATLAS_LEDGER_PATH") or os.path.join(
    ATLAS_HOME, "capture_ledger.db")

for _d in (ATLAS_HOME, BUNDLE_DIR, SESSION_DIR, VIDEO_CACHE, POSTER_CACHE,
           FRAME_VEC_DIR):
    os.makedirs(_d, exist_ok=True)


# ── Telegram ──────────────────────────────────────────────────────────────
# Env-only, deliberately with no fallback literals. An earlier revision of the
# harvester carried a live bot token as a default and published it to a public
# repository; the rule that came out of that applies to every program in this
# repo, including this one. A missing secret fails loudly at startup instead of
# silently authenticating as whatever was last committed.
def _secret(*names, default=""):
    for name in names:
        val = os.environ.get(name)
        if val:
            return val.strip()
    return default


def _int_secret(*names, default=0):
    raw = _secret(*names)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


# Read per access, not once at import. See the long note in the root config.py:
# a frozen `BOT_TOKEN` makes a credential's absence permanent, and a Kaggle
# Secrets store that throttles the launcher for a minute then costs the whole
# session its channel. Atlas is a reader and the channel is everything it reads,
# so this one has no cheaper failure mode than that.
#
# No default for the channel id either. Not a key, but it is the address of
# somebody's private archive and this repo is public; a wrong-but-present id is
# also the quietest way to fail, since Atlas would read a channel that is not
# yours and report nothing wrong.
_TELEGRAM_ENV = {
    "API_ID":     ("ATLAS_API_ID", "VIOS_API_ID", "TELEGRAM_API_ID"),
    "API_HASH":   ("ATLAS_API_HASH", "VIOS_API_HASH", "TELEGRAM_API_HASH"),
    "BOT_TOKEN":  ("ATLAS_BOT_TOKEN", "VIOS_BOT_TOKEN", "TELEGRAM_BOT_TOKEN"),
    "CHANNEL_ID": ("ATLAS_CHANNEL_ID", "VIOS_CHANNEL_ID",
                   "TELEGRAM_CHANNEL_ID"),
}
_TELEGRAM_INT = ("API_ID", "CHANNEL_ID")


def __getattr__(name: str):
    """PEP 562: `config.BOT_TOKEN` is a lookup, evaluated now.

    Not module globals, so there is nothing to go stale. `from config import
    BOT_TOKEN` would still snapshot — every reader in this package uses
    `config.NAME`, which is why this works.
    """
    if name in _TELEGRAM_ENV:
        names = _TELEGRAM_ENV[name]
        return (_int_secret(*names) if name in _TELEGRAM_INT
                else _secret(*names))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(_TELEGRAM_ENV))


def missing_secrets() -> list:
    """Which Telegram secrets are absent, right now, by the name to export."""
    return [names[-1] for attr, names in _TELEGRAM_ENV.items()
            if not __getattr__(attr)]


def telegram_ready() -> bool:
    return not missing_secrets()


# ── Retrieval ─────────────────────────────────────────────────────────────
# bge-small-en-v1.5: 33M params, 384 dimensions, and the smallest model that is
# still genuinely good at retrieval. The whole matrix for 200k moments is
# 200k x 384 x 4B = 307 MB, which stays resident in RAM — so a query is one
# matmul against memory, not a trip to a vector database. At this corpus size
# exhaustive search beats an ANN index on both latency and recall, and it
# removes a service from the deployment.
EMBED_MODEL = os.environ.get("ATLAS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM   = int(os.environ.get("ATLAS_EMBED_DIM", "384"))
EMBED_BATCH = int(os.environ.get("ATLAS_EMBED_BATCH", "128"))
# "auto" takes the GPU only when it has room to spare; "cpu" never does. Atlas
# can share a machine with the harvester's Qwen shards, and a second process
# taking VRAM is how the GPU worker dies mid-narrative.
EMBED_DEVICE = os.environ.get("ATLAS_EMBED_DEVICE", "auto").lower()

# Reuse the harvester's model cache when this is the same machine — the weights
# may already be on disk, and Kaggle's scratch is the right place for them
# either way. These mirror what config.py sets for the harvester.
_SCRATCH = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else CACHE_DIR
HF_CACHE = os.environ.get("HF_HOME") or os.path.join(
    _SCRATCH, "model_cache", "huggingface")
ST_CACHE = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.path.join(
    _SCRATCH, "model_cache", "sentence_transformers")
# bge-v1.5 was trained with an asymmetric instruction on the query side only.
# Dropping it costs a few points of recall, so it is not optional.
EMBED_QUERY_PREFIX = ("Represent this sentence for searching relevant "
                      "passages: ")

# Candidate depth per retriever before fusion. 200 is past the point where
# deeper retrieval changes the top 20, and keeps the fuse itself trivial.
CANDIDATES   = int(os.environ.get("ATLAS_CANDIDATES", "200"))
RRF_K        = 60      # the constant from the original RRF paper
MOMENT_GAP_S = 6.0     # hits closer than this in one video are one moment
QUERY_CACHE  = 256     # LRU entries

# ── Image search ──────────────────────────────────────────────────────────
# The frame vectors the processing plane writes, searched in their own spaces.
# `siglip2` and `clip` are two different geometries; a query is only ever
# compared against the space it was produced in.
#
# The resident matrix is bounded by construction rather than by hope: 62 videos
# at ~900 frames x 1152 dims x 4 B is 257 MB, and an archive four times that size
# would quietly exhaust a Kaggle kernel. `VSEARCH_MAX_MB` picks a frame stride so
# the matrix fits and records it in the meta — so growth costs *resolution*,
# never memory, and the number is visible rather than inferred.
#
# The stride is only the coarse pass: the top videos are re-ranked against their
# full-rate rows read straight from `vec_payload`, so the answer is frame-exact.
VSEARCH_MAX_MB     = int(os.environ.get("ATLAS_VSEARCH_MAX_MB", "256"))
VSEARCH_SPACES     = ("siglip2", "clip")

# CLIP ViT-L/14 for the query side: 1.7 GB against SigLIP2-so400m's ~4 GB, both
# towers in one checkpoint, and stronger on proper nouns and logos — which is
# what a search box actually receives. It is loaded only on the first image or
# text-into-image query, so a session that never runs one never pays for it.
VSEARCH_MODEL      = os.environ.get("ATLAS_VSEARCH_MODEL",
                                    "openai/clip-vit-large-patch14")
# CPU, and not "auto". The processing plane owns both cards for the whole
# session; a query encoder that takes VRAM is how a GPU worker dies mid-pass.
VSEARCH_DEVICE     = os.environ.get("ATLAS_VSEARCH_DEVICE", "cpu").lower()
# How many videos the coarse pass promotes to the frame-exact re-rank.
VSEARCH_CANDIDATES = int(os.environ.get("ATLAS_VSEARCH_CANDIDATES", "24"))

# The same checkpoint's text tower, exported to ONNX — the fallback for hosts
# where torch cannot run at all.
#
# This is not a performance option. On the laptop that reads these archives,
# Windows Smart App Control is enforced, and seven of the nine DLLs shipped in
# `torch/lib` are unsigned, including the 291 MB core. Importing torch there
# raises `OSError: [WinError 4551] An Application Control policy has blocked this
# file`. No older release fixes it — PyTorch has never signed its Windows
# binaries — and the policy has no user allowlist, so the only way to make torch
# run is to turn SAC off, which cannot be undone without reinstalling Windows.
# ONNX Runtime is signed and loads, so the way past it is to stop needing torch.
#
# **It must be an export of `VSEARCH_MODEL` and not merely a CLIP.** CLIP's two
# towers share a space only because they were trained together; a text tower from
# a different run scores frames at chance while looking like it works, because
# the vectors still normalise and still rank. Measured against 5,923 YOLO object
# claims used as an answer key, this export reaches 21.2% precision at 24 against
# a 4.03% base rate — 5.3x chance across twelve labels, none below it — which is
# the evidence that the towers match. Change one of these two names and that
# check has to be re-run.
#
# Only the text tower is fetched: 472 MB against 1.63 GB for the pair, on a host
# with about 1 GB of free RAM. The image tower is a separate 1.16 GB file, so
# `search_image` stays torch-only until something needs it.
VSEARCH_ONNX_REPO  = os.environ.get("ATLAS_VSEARCH_ONNX_REPO",
                                    "Xenova/clip-vit-large-patch14")
VSEARCH_ONNX_TEXT  = os.environ.get("ATLAS_VSEARCH_ONNX_TEXT",
                                    "onnx/text_model.onnx")

# Relative trust in each kind of evidence. Qwen's narrative is a model looking
# at the video and describing it, so it outranks an object list; OCR is exact
# but often noise from a watermark. These are multipliers on the fused score.
SOURCE_WEIGHT = {
    "narrative": 1.30,
    "speech":    1.15,
    "visual":    1.00,
    "ocr":       0.85,
    "caption":   0.95,
    "meta":      0.70,
}

# ── Media ─────────────────────────────────────────────────────────────────
# The Bot API caps getFile at 20 MB, which is under the size of a long reel, so
# downloads go over MTProto and this is only the fallback threshold.
HTTP_DOWNLOAD_LIMIT = 20 * 1024 * 1024
VIDEO_CACHE_GB      = float(os.environ.get("ATLAS_VIDEO_CACHE_GB", "12"))
# Playback streams 1 MiB chunks straight out of the channel, so warming a result
# means fetching two chunks — the head where playback starts and the tail where
# a phone-written mp4 keeps its moov atom — not the whole file. At 2 MiB per
# video instead of 30, a whole page of results can be warmed for less than one
# old-style prefetch.
PREFETCH_TOP_N      = int(os.environ.get("ATLAS_PREFETCH", "12"))

# ── Server ────────────────────────────────────────────────────────────────
PORT = int(os.environ.get("ATLAS_PORT", "7000"))
