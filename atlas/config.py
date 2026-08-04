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
SESSION_DIR  = os.path.join(ATLAS_HOME, "session")
VIDEO_CACHE  = os.path.join(CACHE_DIR, "video")
POSTER_CACHE = os.path.join(CACHE_DIR, "poster")
WEB_DIR      = os.path.join(_HERE, "web")

for _d in (ATLAS_HOME, BUNDLE_DIR, SESSION_DIR, VIDEO_CACHE, POSTER_CACHE):
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


API_ID    = _int_secret("ATLAS_API_ID", "VIOS_API_ID", "TELEGRAM_API_ID")
API_HASH  = _secret("ATLAS_API_HASH", "VIOS_API_HASH", "TELEGRAM_API_HASH")
BOT_TOKEN = _secret("ATLAS_BOT_TOKEN", "VIOS_BOT_TOKEN", "TELEGRAM_BOT_TOKEN")

# The channel id is an address, not a key, so it does carry a default.
CHANNEL_ID = _int_secret("ATLAS_CHANNEL_ID", "VIOS_CHANNEL_ID",
                         "TELEGRAM_CHANNEL_ID", default=-1003762735924)

_SECRET_NAMES = (("API_ID", API_ID, "TELEGRAM_API_ID"),
                 ("API_HASH", API_HASH, "TELEGRAM_API_HASH"),
                 ("BOT_TOKEN", BOT_TOKEN, "TELEGRAM_BOT_TOKEN"))


def missing_secrets() -> list:
    """Which Telegram secrets are absent, by the name the launcher exports."""
    return [env for _a, val, env in _SECRET_NAMES if not val]


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
