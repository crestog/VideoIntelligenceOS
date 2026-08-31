"""
Image search: find frames that look like a thing.

The processing plane runs SigLIP2 and CLIP over roughly nine hundred frames of
every video and writes both as packed per-frame matrices. Until `ingest`'s
payload lane existed those bytes were discarded at the door, so this whole
capability was one regex away from working and looked like a missing feature.

## Three modes, and only one of them needs a model

1. **Frame to frames.** "More like this moment", and near-duplicate detection
   across the archive. The query vector is already in the database, so this is
   pure cosine against a resident matrix — no model, no download, working the
   instant the first shard lands, in either space. It is the mode to reach for.
2. **Image to frames.** A screenshot, embedded with CLIP's image tower.
3. **Text to frames.** The same checkpoint's *text* tower, so "red car at night"
   queries the pixels rather than the captions. This is the qualitative jump, and
   it costs nothing beyond mode 2 because one checkpoint carries both towers.

Modes 2 and 3 only ever run in the `clip` space, because that is the space the
query encoder produces. Comparing a CLIP vector against a SigLIP2 one is not a
worse answer, it is a meaningless one — two geometries with no relation — so the
space is not a preference here, it is a constraint the code enforces.

## Why a flat file and not a vector database

The same argument `search.py` makes for moments, with the numbers moved: 62
videos at ~900 frames is ~56,000 vectors, and one float32 matrix of that is
257 MB resident. A query is `matrix @ q` — memory-bandwidth-bound, a few
milliseconds, and *exact*. An ANN index would add a service, a build step and an
approximation to make a fast thing slightly faster.

## Bounded memory, exact answers

Those two are usually a trade, and the resolution is two stages:

- The resident matrix is **strided**. `VSEARCH_MAX_MB` picks the stride, so an
  archive ten times this size costs resolution in the coarse pass rather than
  memory, and the stride is recorded in the meta where it can be read.
- The coarse pass ranks *videos*. The top `VSEARCH_CANDIDATES` of them are then
  re-ranked against their **full-rate** rows, read straight out of
  `vec_payload` — so the frame that comes back is the best frame, not the best
  strided sample near it.

## Why this survives a reindex

`moments.vec` needs a `build_id` guard because it is keyed by `moments.id`, which
every rebuild reassigns. Frame vectors are keyed by `(video_key, frame_idx)`,
which no rebuild touches — so these files are valid across any number of index
rebuilds, and can be built incrementally as each shard lands.
"""

import json
import os
import sqlite3
import threading
import time

from . import config
from .tgchannel import log

# One resident entry per space: {"vecs", "ids", "dim", "stride", "videos"}.
_LOCK = threading.RLock()
_RESIDENT: dict = {}
_STATE: dict = {"built_at": 0.0, "building": False, "detail": "", "spaces": {}}

# Debounce for `build_if_due`: vector bytes seen since the last rebuild, and when
# that rebuild was. 45 s is long enough that a cold scan of a hundred shards
# rebuilds a handful of times instead of a hundred, and short enough that image
# search is live within a minute of the first spine shard landing.
_PENDING_BYTES = 0
_LAST_BUILD = 0.0
_BUILD_MIN_GAP = float(os.environ.get("ATLAS_VSEARCH_MIN_GAP", "45"))

# The query encoder, loaded once and only when a query needs it.
_ENC_LOCK = threading.RLock()
_ENCODER = None
_ENC_TRIED = False
_ENC_ERROR = ""

_ITEMSIZE = {"f16": 2, "f32": 4}


# ══════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════
def _paths(space: str) -> tuple:
    """`(vec, ids, meta)` for one space. The space name is part of the filename
    because two spaces must never be able to land in the same matrix."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in space)[:32]
    base = os.path.join(config.FRAME_VEC_DIR, f"frames-{safe}")
    return base + ".vec", base + ".ids", base + ".json"


# ══════════════════════════════════════════════════════════════════════════
# READING THE PAYLOADS
# ══════════════════════════════════════════════════════════════════════════
def _unpack(row, np):
    """One `vec_payload` frame row as `(frame_indices, (n, dim) float32)`.

    Returns `(None, None)` when the row does not describe a readable buffer.
    Every check here is a real failure mode of a torn shard rather than defensive
    habit: a truncated payload arrives with `n * dim` not matching `len(data)`,
    and reshaping it would either raise or — worse, with the wrong dtype —
    succeed and produce noise that looks like an embedding.
    """
    dim = int(row["dim"] or 0)
    n = int(row["n"] or 0)
    dtype = str(row["dtype"] or "f16")
    item = _ITEMSIZE.get(dtype)
    if not (dim > 0 and n > 0 and item) or row["data"] is None:
        return None, None
    if len(row["data"]) != n * dim * item:
        return None, None
    mat = np.frombuffer(row["data"],
                        dtype=np.float16 if dtype == "f16" else np.float32)
    mat = mat.reshape(n, dim).astype(np.float32)
    if row["frames"] is not None and len(row["frames"]) == n * 4:
        idx = np.frombuffer(row["frames"], dtype=np.int32).astype(np.int64)
    else:
        # A row whose frame list did not survive still carries usable vectors;
        # assuming contiguity is wrong in general but it is the only reading
        # available, and it is recorded as a guess in the meta's `assumed` count.
        idx = np.arange(n, dtype=np.int64)
    return idx, mat


def _l2(mat, np):
    """L2-normalise rows in place-ish. The writers already normalise, so this is
    idempotent — and it is what makes a dot product a cosine, which every
    ranking below assumes."""
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return mat / norm


def frame_rows(conn: sqlite3.Connection, space: str,
               video_key: str = "") -> list:
    """`vec_payload` frame rows for one space, optionally one video.

    The row factory is set on the *cursor*, not the connection: this takes a
    connection Atlas hands round to every other reader, and flipping its
    `row_factory` under them is a side effect nothing here needs.
    """
    sql = ("SELECT uid, video_key, dim, n, dtype, frames, data "
           "FROM vec_payload WHERE kind='frame_vector' AND space=?")
    args = [space]
    if video_key:
        sql += " AND video_key=?"
        args.append(video_key)
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        return cur.execute(sql + " ORDER BY video_key", args).fetchall()
    except sqlite3.Error:
        return []


def _space_shape(conn: sqlite3.Connection, space: str) -> tuple:
    """`(videos, frames, dim)` held for a space, without decoding anything.

    Cheap, because the stride has to be chosen *before* any buffer is read — a
    build that decoded everything first to decide how much to keep would need the
    memory the stride exists to avoid.
    """
    try:
        row = conn.execute(
            "SELECT COUNT(DISTINCT video_key), COALESCE(SUM(n),0), "
            "COALESCE(MAX(dim),0) FROM vec_payload "
            "WHERE kind='frame_vector' AND space=?", (space,)).fetchone()
    except sqlite3.Error:
        return 0, 0, 0
    return int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)


def _stride_for(frames: int, dim: int) -> int:
    """The smallest stride whose resident matrix fits `VSEARCH_MAX_MB`.

    Growth costs resolution, never memory. Returning 1 for an archive that fits
    is the common case today and the point of the ceiling being generous.
    """
    if frames <= 0 or dim <= 0:
        return 1
    budget = max(config.VSEARCH_MAX_MB, 16) * 1048576
    per = dim * 4
    room = max(int(budget // per), 1)
    stride = 1
    while frames // stride > room:
        stride += 1
    return stride


# ══════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════
def build(conn: sqlite3.Connection, spaces=None) -> dict:
    """Write and load the frame index for each space that has payloads.

    Safe to call after every shard import: it is a full rebuild per space, which
    at this size is seconds, and a full rebuild cannot leave the ids and the
    matrix disagreeing the way an append that failed halfway could.
    """
    out = {}
    try:
        import numpy as np
    except ImportError:
        _STATE["detail"] = "numpy missing — image search unavailable"
        log("image search unavailable — numpy missing")
        return {"error": "numpy missing"}

    with _LOCK:
        _STATE["building"] = True
    try:
        for space in (spaces or config.VSEARCH_SPACES):
            got = _build_space(conn, space, np)
            if got:
                out[space] = got
    finally:
        with _LOCK:
            _STATE["building"] = False
            _STATE["built_at"] = time.time()
            _STATE["spaces"] = {**_STATE.get("spaces", {}), **out}
    return out


def _build_space(conn: sqlite3.Connection, space: str, np) -> dict:
    videos, frames, dim = _space_shape(conn, space)
    if not frames or not dim:
        return {}
    stride = _stride_for(frames, dim)

    rows = frame_rows(conn, space)
    keep_v, keep_i, assumed, bad = [], [], 0, 0
    ordinals: dict = {}
    for r in rows:
        idx, mat = _unpack(r, np)
        if idx is None:
            bad += 1
            continue
        if r["frames"] is None:
            assumed += 1
        key = r["video_key"]
        if key not in ordinals:
            ordinals[key] = len(ordinals)
        ordv = ordinals[key]
        if stride > 1:
            idx, mat = idx[::stride], mat[::stride]
        if not len(idx):
            continue
        keep_v.append(_l2(mat, np))
        # `(video_ord << 32) | frame_idx` in one int64, the same trick the moment
        # index uses to keep the sidecar a flat array rather than a second table.
        # 32 bits of frame index is 4 billion frames per video; the ordinal is
        # local to this build and resolved through `videos` in the meta.
        keep_i.append((np.int64(ordv) << np.int64(32))
                      | idx.astype(np.int64))
    if not keep_v:
        return {}

    mat = np.concatenate(keep_v, axis=0).astype(np.float32)
    ids = np.concatenate(keep_i, axis=0).astype(np.int64)
    vec_p, ids_p, meta_p = _paths(space)
    try:
        mat.tofile(vec_p + ".tmp")
        ids.tofile(ids_p + ".tmp")
        os.replace(vec_p + ".tmp", vec_p)
        os.replace(ids_p + ".tmp", ids_p)
    except OSError as exc:
        log(f"image index {space}: could not write — {type(exc).__name__}: {exc}")
        return {}

    order = [k for k, _v in sorted(ordinals.items(), key=lambda kv: kv[1])]
    meta = {"space": space, "dim": int(dim), "count": int(len(ids)),
            "stride": int(stride), "videos": order,
            "frames_total": int(frames), "videos_total": int(videos),
            "assumed_frame_ids": int(assumed), "unreadable_rows": int(bad),
            "resident_mb": round(mat.nbytes / 1048576, 1),
            "built_at": time.time()}
    try:
        with open(meta_p, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    except OSError:
        pass

    with _LOCK:
        _RESIDENT[space] = {"vecs": mat, "ids": ids, "dim": int(dim),
                            "stride": int(stride), "videos": order}
    log(f"image index {space} resident — {len(ids)} frames x {dim}d "
        f"({mat.nbytes / 1048576:.0f} MB, stride {stride}, "
        f"{len(order)} videos)"
        + (f", {bad} unreadable row(s)" if bad else ""))
    return meta


def build_if_due(conn: sqlite3.Connection, added_bytes: int = 0,
                 force: bool = False) -> dict:
    """Rebuild after an import, but not once per shard during a cold scan.

    `build` is a full rebuild per space, which is seconds at this size and the
    only shape that cannot leave the matrix and its ids disagreeing. Called
    straight from the import loop that is fine during a live run — roughly one
    shard a minute — and wasteful during a cold scan of a hundred shards, where
    it would rebuild the same matrix a hundred times.

    So the policy lives here rather than at the two call sites: bytes accumulate,
    and a build happens when enough time has passed or when the caller says the
    scan is over. `force` is that end-of-scan call, and it is what guarantees the
    index reflects everything imported even if every interval was skipped.
    """
    global _PENDING_BYTES, _LAST_BUILD
    with _LOCK:
        _PENDING_BYTES += max(int(added_bytes or 0), 0)
        if not _PENDING_BYTES and not force:
            return {}
        due = force or (time.time() - _LAST_BUILD) >= _BUILD_MIN_GAP
        if not due:
            return {}
        _PENDING_BYTES = 0
        _LAST_BUILD = time.time()
    return build(conn)


def reload(spaces=None) -> dict:
    """Load already-written index files into RAM. Used at server start."""
    out = {}
    try:
        import numpy as np
    except ImportError:
        return out
    for space in (spaces or config.VSEARCH_SPACES):
        vec_p, ids_p, meta_p = _paths(space)
        try:
            with open(meta_p, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            dim = int(meta.get("dim") or 0)
            if dim <= 0:
                continue
            vecs = np.fromfile(vec_p, dtype=np.float32)
            ids = np.fromfile(ids_p, dtype=np.int64)
            if vecs.size % dim or len(ids) != vecs.size // dim:
                log(f"image index {space} ignored — {vecs.size} floats against "
                    f"{len(ids)} ids at {dim}d; it rebuilds on the next import")
                continue
            vecs = vecs.reshape(-1, dim)
        except (OSError, ValueError, KeyError):
            continue
        with _LOCK:
            _RESIDENT[space] = {"vecs": vecs, "ids": ids, "dim": dim,
                                "stride": int(meta.get("stride") or 1),
                                "videos": list(meta.get("videos") or [])}
            _STATE["spaces"] = {**_STATE.get("spaces", {}), space: meta}
        out[space] = meta
        log(f"image index {space} resident — {len(ids)} frames x {dim}d "
            f"({vecs.nbytes / 1048576:.0f} MB, stride {meta.get('stride')})")
    return out


def ready(space: str = "") -> bool:
    with _LOCK:
        if space:
            return space in _RESIDENT
        return bool(_RESIDENT)


def state() -> dict:
    """Everything a caller needs to explain an empty result.

    An image search that returns nothing has four possible causes — no payloads
    imported, no numpy, no encoder, or a genuinely unlike archive — and they are
    indistinguishable from an empty list. This is what tells them apart.
    """
    with _LOCK:
        spaces = {}
        for space in config.VSEARCH_SPACES:
            res = _RESIDENT.get(space)
            meta = (_STATE.get("spaces") or {}).get(space) or {}
            spaces[space] = {
                "resident": bool(res),
                "frames": int(len(res["ids"])) if res else 0,
                "dim": int(res["dim"]) if res else int(meta.get("dim") or 0),
                "stride": int(res["stride"]) if res else 1,
                "videos": len(res["videos"]) if res else 0,
                "resident_mb": (round(res["vecs"].nbytes / 1048576, 1)
                                if res else 0.0),
                "frames_total": int(meta.get("frames_total") or 0),
                "unreadable_rows": int(meta.get("unreadable_rows") or 0),
            }
        return {
            "spaces": spaces,
            "building": bool(_STATE.get("building")),
            "built_at": _STATE.get("built_at") or 0.0,
            "detail": _STATE.get("detail") or "",
            "max_mb": config.VSEARCH_MAX_MB,
            "candidates": config.VSEARCH_CANDIDATES,
            "encoder": {"model": config.VSEARCH_MODEL,
                        "device": config.VSEARCH_DEVICE,
                        "loaded": _ENCODER is not None,
                        "tried": _ENC_TRIED,
                        "error": _ENC_ERROR},
            "query_space": "clip",
        }


# ══════════════════════════════════════════════════════════════════════════
# THE QUERY ENCODER — CLIP, both towers, on the CPU
# ══════════════════════════════════════════════════════════════════════════
class _Clip:
    """CLIP's two towers behind one object. Returns L2-normalised float32."""

    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model, processor, device

    def _norm(self, t):
        return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def text(self, query: str):
        import torch
        with torch.no_grad():
            batch = self.processor(text=[query], return_tensors="pt",
                                   padding=True, truncation=True,
                                   max_length=77).to(self.device)
            return self._norm(self.model.get_text_features(**batch))[0] \
                .cpu().float().numpy()

    def image(self, data: bytes):
        import io

        import torch
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        with torch.no_grad():
            batch = self.processor(images=[img], return_tensors="pt") \
                .to(self.device)
            return self._norm(self.model.get_image_features(**batch))[0] \
                .cpu().float().numpy()


def get_encoder():
    """Load CLIP once, on first use. Returns None if it cannot be had.

    Mirrors `encoder.get_encoder` — module singleton, tried-once, logs and
    degrades rather than raising — with two departures, both forced:

    * It pins **CPU** by default. The processing plane owns both cards for the
      whole session, and a query encoder that takes VRAM is how a GPU worker dies
      mid-pass. One query on four vCPUs is around 0.3 s, which is fine for a
      search box.
    * It loads on the first query rather than at import, so a session that only
      ever uses mode 1 — which needs no model at all — never pays the 1.7 GB.
    """
    global _ENCODER, _ENC_TRIED, _ENC_ERROR
    with _ENC_LOCK:
        if _ENCODER is not None or _ENC_TRIED:
            return _ENCODER
        _ENC_TRIED = True

        for var, path in (("HF_HOME", config.HF_CACHE),):
            os.environ.setdefault(var, path)
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                pass
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            _ENC_ERROR = f"torch/transformers missing ({exc})"
            log(f"image query encoder unavailable — {_ENC_ERROR}. Frame-to-frame "
                f"search still works; it needs no model.")
            return None

        device = config.VSEARCH_DEVICE
        if device.startswith("cuda") and not (
                getattr(torch, "cuda", None) and torch.cuda.is_available()):
            device = "cpu"
        if device == "cpu":
            try:
                torch.set_num_threads(max(2, (os.cpu_count() or 4) - 1))
            except Exception:                          # noqa: BLE001
                pass

        t0 = time.time()
        try:
            model = CLIPModel.from_pretrained(config.VSEARCH_MODEL).eval() \
                .to(device)
            proc = CLIPProcessor.from_pretrained(config.VSEARCH_MODEL)
        except Exception as exc:                       # noqa: BLE001
            _ENC_ERROR = f"{type(exc).__name__}: {str(exc)[:200]}"
            log(f"image query encoder failed to load — {_ENC_ERROR}")
            return None
        _ENCODER = _Clip(model, proc, device)
        log(f"image query encoder ready — {config.VSEARCH_MODEL} on {device} "
            f"in {time.time() - t0:.0f}s")
        return _ENCODER


def warm() -> bool:
    """Load the encoder now rather than on the first query."""
    return get_encoder() is not None


# ══════════════════════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════════════════════
def _coarse(space: str, q, np, limit: int) -> list:
    """Video keys ranked by their best strided frame. Returns `[(key, score)]`.

    Every reel is scored, and that is the whole of this function.

    Taking the best N frames overall and grouping them by reel sounds like the
    same thing and is not, because frames inside one reel are near-duplicates of
    each other. A reel that matches well does not contribute one high score, it
    contributes hundreds of nearly identical ones, and they crowd out reels that
    match slightly less well. Measured on this archive: the top 1,536 of 32,302
    frames came from **nine** reels of thirty. The other twenty-one never
    reached `_exact`, so they could not be returned at any rank, however well
    they matched — the shortlist was not narrowing the search, it was deciding
    it.

    A segmented maximum gives each reel its own best frame in one pass, which is
    what "rank the videos" meant all along. It is also cheaper than the sort it
    replaces (0.35 ms against 0.95 ms here) and stays negligible against the
    matmul above it: 6.7 ms at a million frames, where the matmul is ~240 ms.

    `np.maximum.at` is the *unbuffered* form and is required. The buffered
    `best[ordv] = np.maximum(best[ordv], sims)` reads every slot before writing
    any, so among frames sharing a reel only the last one counts.
    """
    with _LOCK:
        res = _RESIDENT.get(space)
    if not res or not len(res["ids"]):
        return []
    if len(q) != res["dim"]:
        return []
    videos = res["videos"]
    if not videos:
        return []
    sims = res["vecs"] @ q.astype(np.float32)
    # A frame id carries its reel's ordinal in the high word.
    ordv = res["ids"].astype(np.int64) >> 32
    keep = ordv < len(videos)
    if not keep.all():
        # A payload naming a reel this index never resolved is skipped rather
        # than fatal, exactly as the old row-at-a-time loop skipped it.
        ordv, sims = ordv[keep], sims[keep]
    if not len(sims):
        return []
    # Below every real cosine, so a reel with no surviving frame stays out.
    best = np.full(len(videos), -2.0, dtype=np.float32)
    np.maximum.at(best, ordv, sims)
    return [(videos[int(i)], float(best[int(i)]))
            for i in np.argsort(-best)[:limit] if best[int(i)] > -2.0]


def _exact(conn, space: str, q, np, keys: list, limit: int) -> list:
    """Frame-exact re-rank inside a shortlist of videos.

    This is the half that makes a strided index give an unstrided answer: the
    coarse pass only has to get the *video* right, and full-rate rows for two
    dozen videos are a few megabytes read from `vec_payload` on demand.
    """
    hits = []
    for key in keys:
        for r in frame_rows(conn, space, key):
            idx, mat = _unpack(r, np)
            if idx is None or len(idx) == 0:
                continue
            sims = _l2(mat, np) @ q.astype(np.float32)
            n = min(len(sims), max(limit, 8))
            top = np.argpartition(-sims, n - 1)[:n] if len(sims) > n \
                else np.arange(len(sims))
            for i in top:
                hits.append((key, int(idx[i]), float(sims[i])))
    hits.sort(key=lambda h: -h[2])
    return hits[:limit]


def _fps(conn, keys: list) -> dict:
    """`{video_key: fps}` for turning a frame index into a timestamp.

    `video.fps` is what Atlas holds — the honest per-frame timestamp list lives
    in the extractor's manifest on the processing plane's scratch disk and never
    crosses into a shard, so `frame_idx / fps` is the best available reading. On
    a variable-rate phone video it can drift, which is why the response carries
    `frame_idx` as well: the frame number is exact, the seconds are derived.

    A video with no usable `fps` gets no `t` rather than a guessed 30 — a wrong
    `t` seeks the player to the wrong moment, where a missing one just opens the
    video at the start.
    """
    out: dict = {}
    keys = list(dict.fromkeys(k for k in keys if k))
    if not keys:
        return out
    try:
        marks = ",".join("?" * len(keys))
        cur = conn.execute(
            f"SELECT video_key, fps, duration FROM video "
            f"WHERE video_key IN ({marks})", keys)
        for key, fps, duration in cur:
            fps = float(fps or 0)
            if fps > 0:
                out[key] = fps
    except sqlite3.Error:
        # `video` is a reflected table, so early in a database's life it may not
        # carry these columns yet. Timestamps are a convenience on top of an
        # exact frame index; losing them is not losing the hit.
        pass
    return out


def frame_time(conn, video_key: str, frame_idx: int):
    """Seconds for one `(video, frame)`, or None when `fps` is not known."""
    fps = _fps(conn, [video_key]).get(video_key)
    return round(max(int(frame_idx), 0) / fps, 3) if fps else None


def frame_for_time(conn, video_key: str, seconds: float):
    """The frame index at `seconds`, or None when `fps` is not known.

    The inverse of `frame_time`, and the one a player needs: the interface has a
    playhead, never a frame number.
    """
    fps = _fps(conn, [video_key]).get(video_key)
    return int(round(max(float(seconds), 0.0) * fps)) if fps else None


def _shape(conn, space: str, hits: list) -> list:
    fps = _fps(conn, [h[0] for h in hits])
    out = []
    for key, idx, score in hits:
        f = fps.get(key)
        out.append({"video_key": key, "frame_idx": idx,
                    "t": round(idx / f, 2) if f else None,
                    "score": round(score, 4), "space": space})
    return out


def _resident_hits(space: str, q, np, limit: int, exclude_key: str):
    """Frame-exact hits straight from an unstrided resident matrix, or `None`.

    `_exact` exists to make a *strided* index give an unstrided answer: the
    coarse pass narrows to a shortlist of reels, and the full-rate rows for
    those reels are re-read from `vec_payload` because the resident matrix threw
    most of them away. When nothing was strided away, that second read fetches
    and decompresses the very vectors `_coarse` just multiplied.

    The cost is not theoretical. Widening the shortlist from nine reels to
    twenty-three took a query from 110 ms to 700 ms on this archive, and every
    one of those milliseconds re-read RAM-resident data. So an unstrided index
    answers from the matrix, and `_exact` keeps the job it was written for.

    Returns `None` — not `[]` — when this path does not apply, so that "the
    index is strided" stays distinguishable from "nothing matched".
    """
    with _LOCK:
        res = _RESIDENT.get(space)
    if not res or res.get("stride", 1) != 1 or not len(res["ids"]):
        return None
    ids, videos = res["ids"], res["videos"]
    sims = res["vecs"] @ q.astype(np.float32)
    ordv = ids.astype(np.int64) >> 32
    keep = ordv < len(videos)
    if exclude_key:
        try:
            keep = keep & (ordv != videos.index(exclude_key))
        except ValueError:
            pass                      # not in this index, so nothing to drop
    if not keep.all():
        sims, ordv, ids = sims[keep], ordv[keep], ids[keep]
    if not len(sims):
        return []
    n = min(len(sims), max(int(limit), 1))
    top = (np.argpartition(-sims, n - 1)[:n] if len(sims) > n
           else np.arange(len(sims)))
    top = top[np.argsort(-sims[top])]
    return [(videos[int(ordv[i])], int(ids[i] & 0xFFFFFFFF), float(sims[i]))
            for i in top]


def search_vector(conn: sqlite3.Connection, q, space: str,
                  limit: int = 40, exclude_key: str = "") -> dict:
    """Rank frames against a query vector already in `space`."""
    try:
        import numpy as np
    except ImportError:
        return {"hits": [], "reason": "numpy missing"}
    if not ready(space):
        return {"hits": [], "reason": f"no resident index for {space}"}
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(q))
    if n == 0:
        return {"hits": [], "reason": "the query vector is empty"}
    q = q / n

    hits = _resident_hits(space, q, np, limit, exclude_key)
    if hits is not None:
        # Every reel in the index was compared frame by frame, so the count
        # reported is the truth and not the size of a shortlist.
        with _LOCK:
            res = _RESIDENT.get(space) or {}
        seen = len(res.get("videos") or ())
        if exclude_key and exclude_key in (res.get("videos") or ()):
            seen -= 1
        if not hits:
            return {"hits": [], "space": space, "searched_videos": seen,
                    "reason": "nothing in this space resembles it"}
        return {"hits": _shape(conn, space, hits), "space": space,
                "searched_videos": seen}

    shortlist = [k for k, _s in _coarse(space, q, np,
                                        config.VSEARCH_CANDIDATES)
                 if k != exclude_key]
    if not shortlist:
        return {"hits": [], "reason": "nothing in this space resembles it"}
    hits = [h for h in _exact(conn, space, q, np, shortlist, limit * 2)
            if h[0] != exclude_key][:limit]
    return {"hits": _shape(conn, space, hits), "space": space,
            "searched_videos": len(shortlist)}


def similar_to_frame(conn: sqlite3.Connection, video_key: str, frame_idx: int,
                     space: str = "siglip2", limit: int = 40,
                     same_video: bool = False) -> dict:
    """Mode 1 — frames that look like this frame. Needs no model.

    The query vector is read out of the database at full rate, not out of the
    strided resident matrix, so asking about a frame the stride skipped is a
    normal query rather than a miss.
    """
    try:
        import numpy as np
    except ImportError:
        return {"hits": [], "reason": "numpy missing"}
    q, used = None, int(frame_idx)
    for r in frame_rows(conn, space, video_key):
        idx, mat = _unpack(r, np)
        if idx is None or not len(idx):
            continue
        where = np.nonzero(idx == int(frame_idx))[0]
        if len(where):
            q, used = mat[int(where[0])], int(frame_idx)
            break
        if q is None:
            # The exact frame was not embedded — a frame that failed to decode
            # leaves a hole, and the tiers do not all cover every index. The
            # nearest embedded frame is the honest answer to "more like this
            # moment", and `query.frame_used` says which frame answered so a
            # caller is never told it got the frame it asked for.
            near = int(np.argmin(np.abs(idx - int(frame_idx))))
            q, used = mat[near], int(idx[near])
    if q is None:
        return {"hits": [],
                "reason": f"{video_key} has no {space} frame vectors"}
    out = search_vector(conn, q, space, limit,
                        exclude_key="" if same_video else video_key)
    out["query"] = {"video_key": video_key, "frame_idx": int(frame_idx),
                    "frame_used": used, "space": space}
    return out


def search_text(conn: sqlite3.Connection, query: str,
                limit: int = 40) -> dict:
    """Mode 3 — text into the image space, through CLIP's text tower."""
    enc = get_encoder()
    if enc is None:
        return {"hits": [], "reason": _ENC_ERROR or "no query encoder"}
    try:
        q = enc.text(query)
    except Exception as exc:                           # noqa: BLE001
        return {"hits": [],
                "reason": f"encode failed: {type(exc).__name__}: "
                          f"{str(exc)[:120]}"}
    out = search_vector(conn, q, "clip", limit)
    out["query"] = {"text": query, "space": "clip"}
    return out


def search_image(conn: sqlite3.Connection, data: bytes,
                 limit: int = 40) -> dict:
    """Mode 2 — an uploaded screenshot, through CLIP's image tower."""
    enc = get_encoder()
    if enc is None:
        return {"hits": [], "reason": _ENC_ERROR or "no query encoder"}
    try:
        q = enc.image(data)
    except Exception as exc:                           # noqa: BLE001
        return {"hits": [],
                "reason": f"could not read the image: {type(exc).__name__}: "
                          f"{str(exc)[:120]}"}
    out = search_vector(conn, q, "clip", limit)
    out["query"] = {"image_bytes": len(data), "space": "clip"}
    return out
