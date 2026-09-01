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
from .hfcompat import projected
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

# How far apart two hits from one reel must sit, and how many one reel may own.
#
# Frames are extracted at the reel's own rate, so consecutive indices are the
# same instant photographed twice. Ranking them honestly puts them side by side:
# asking this archive for twenty-four frames like a given frame returned
# twenty-four frames from *one* reel spanning three to ten distinct seconds —
# the same picture repeated, because the poster cache is keyed per second and
# literally served one file for ten of them.
#
# 1.5 s is chosen against the frame rate rather than against taste: at 30 fps it
# is a 45-frame separation, wide enough that a hand has moved and a cut has
# landed, narrow enough that a fast montage still contributes several moments.
#
# Shot boundaries were the obvious alternative and are not usable here. The
# `shot` table covers all thirty reels but wildly unevenly — one reel is a single
# shot spanning 0-73 s, so "one hit per shot" would return one frame for the
# longest reel in the archive while giving a 41-shot reel forty-one.
_SPREAD_S = float(os.environ.get("ATLAS_VSEARCH_SPREAD_S", "1.5"))
_PER_VIDEO = int(os.environ.get("ATLAS_VSEARCH_PER_VIDEO", "6"))

# Used only to convert `_SPREAD_S` into a frame gap for a reel whose `fps` is
# unknown. `_fps` deliberately refuses to guess 30 for a *timestamp*, because a
# wrong `t` seeks the player to the wrong moment. A wrong gap merely spaces the
# results slightly differently, so a guess is affordable here and nowhere else.
_SPREAD_FPS = 30.0

# Candidates ranked before spreading. The spread discards near-duplicates, so it
# needs more than `limit` to choose from; when it still comes up short the pool
# widens once rather than returning a thin page.
_POOL = 64
_POOL_MIN = 4096


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
                        # Which route answered, and what it can do. The two are
                        # not equivalent: the ONNX fallback carries the text
                        # tower only, so "loaded" is true while an uploaded
                        # image still cannot be encoded. A screen that shows
                        # only "loaded" would offer a mode that cannot run.
                        "runtime": ("torch" if isinstance(_ENCODER, _Clip)
                                    else "onnx" if isinstance(_ENCODER, _Onnx)
                                    else ""),
                        "can_text": _ENCODER is not None,
                        "can_image": isinstance(_ENCODER, _Clip),
                        "error": _ENC_ERROR},
            "query_space": "clip",
        }


# ══════════════════════════════════════════════════════════════════════════
# THE QUERY ENCODER — CLIP, through torch where it runs and ONNX where it cannot
# ══════════════════════════════════════════════════════════════════════════
class BadImage(ValueError):
    """The bytes are not a picture — as distinct from the tower having raised.

    Exists so `search_image` can tell two failures apart that arrive at the same
    `except` and mean opposite things. Decoding is the caller's fault and the fix
    is a different file; the model call raising is *this build's* fault and the
    fix is a traceback. Reported as one, the interface tells somebody their
    screenshot is corrupt when the encoder is what broke — the same shape of lie
    as reading a class name called `BaseModelOutputWithPooling` and concluding a
    model was missing. So the decode step raises this, and nothing else does.
    """


class _Onnx:
    """CLIP's text tower, as a graph, with no torch anywhere beneath it.

    Exists because torch is not always *allowed* to run. On a Windows host with
    Smart App Control enforced, `import torch` raises an Application Control
    error — its DLLs are unsigned, and the policy has no exception list. ONNX
    Runtime is signed, `tokenizers` is signed, and between them that is the whole
    text tower, so the search that "needs a model" needs no torch.

    Two details are load-bearing and neither is obvious:

    * **The graph pools at the EOS position itself.** It takes `input_ids` alone —
      no attention mask — and finds where the sentence ends with an argmax over
      the ids, which works because CLIP's EOS is the highest id in its
      vocabulary. So padding is unnecessary, and it is also harmless: measured,
      padding to 77 with EOS gives a vector identical to the unpadded one to the
      last bit, because argmax takes the *first* maximum. This encodes one query
      at a time and does not pad.

    * **The output must be `text_embeds`.** CLIP ViT-L/14 has `hidden_size` 768
      and `projection_dim` 768, so a pooled hidden state and a projected
      embedding are indistinguishable by shape — and only the projected one lives
      in the space the image tower wrote. Checked by name at load, because
      getting this wrong produces a search that ranks confidently in a space
      nothing else occupies.
    """

    def __init__(self, sess, tok, out: str):
        self.sess, self.tok, self.out = sess, tok, out

    def text(self, query: str):
        import numpy as np
        # Guard the *text*, not the token ids. CLIP's post-processor always wraps
        # the sequence in `<|startoftext|>`/`<|endoftext|>`, so an empty phrase
        # still tokenises to two ids and still produces a perfectly well-formed
        # vector — one that points at whatever "nothing" means in this space and
        # would return two dozen arbitrary frames as if they matched.
        if not (query or "").strip():
            raise ValueError("the query is empty")
        ids = self.tok.encode(query).ids[:77]
        v = self.sess.run([self.out],
                          {"input_ids": np.asarray([ids], dtype=np.int64)})[0]
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def image(self, data: bytes):
        # The vision tower is a separate 1.16 GB export and is not fetched. Say
        # so, rather than failing somewhere further down with a shape error.
        raise RuntimeError(
            "the ONNX fallback carries the text tower only — searching by an "
            "uploaded image needs torch, or the vision tower export")


class _Clip:
    """CLIP's two towers behind one object. Returns L2-normalised float32.

    Both towers go through `hfcompat.projected`, which is the whole of what makes
    this work on transformers 5.x — see that module. The two calls must use it or
    neither: a query read one way and frames read the other are not in the same
    space, and nothing about that failure looks like a failure.
    """

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
            got = projected(self.model.get_text_features(**batch),
                            "text features")
            return self._norm(got)[0].cpu().float().numpy()

    def image(self, data: bytes):
        import io

        import torch
        from PIL import Image
        # Decode alone, in its own guard, so that `BadImage` means the upload and
        # nothing else. Everything after this line is the model, and if it raises
        # the caller must not describe it as an unreadable picture.
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:                       # noqa: BLE001
            raise BadImage(f"{type(exc).__name__}: {exc}") from exc
        with torch.no_grad():
            batch = self.processor(images=[img], return_tensors="pt") \
                .to(self.device)
            got = projected(self.model.get_image_features(**batch),
                            "image features")
            return self._norm(got)[0].cpu().float().numpy()


def _load_torch():
    """CLIP through torch — both towers. Returns `(encoder, reason)`.

    The import is guarded against `Exception`, not `ImportError`, and that is not
    defensive padding. An installed-but-forbidden torch raises `OSError:
    [WinError 4551] An Application Control policy has blocked this file` while
    loading its DLLs, which an `ImportError` clause does not catch — so the
    narrower guard turned a missing model into a 500 on a machine where torch was
    present and blocked. The difference between "absent" and "refused" belongs in
    the message, not in whether the request survives.
    """
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        return None, f"torch/transformers missing ({exc})"
    except Exception as exc:                           # noqa: BLE001
        return None, f"torch present but unusable — {type(exc).__name__}: " \
                     f"{str(exc)[:160]}"

    device = config.VSEARCH_DEVICE
    if device.startswith("cuda") and not (
            getattr(torch, "cuda", None) and torch.cuda.is_available()):
        device = "cpu"
    if device == "cpu":
        try:
            torch.set_num_threads(max(2, (os.cpu_count() or 4) - 1))
        except Exception:                              # noqa: BLE001
            pass

    t0 = time.time()
    try:
        model = CLIPModel.from_pretrained(config.VSEARCH_MODEL).eval().to(device)
        proc = CLIPProcessor.from_pretrained(config.VSEARCH_MODEL)
    except Exception as exc:                           # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"
    log(f"image query encoder ready — {config.VSEARCH_MODEL} on {device} "
        f"in {time.time() - t0:.0f}s")
    return _Clip(model, proc, device), ""


def _load_onnx():
    """The same checkpoint's text tower, as a graph. Returns `(encoder, reason)`.

    Deliberately reaches past transformers for the tokenizer. On the host this
    fallback exists for, transformers is installed and entirely unusable:
    attribute access on its lazy module resolves through torch, so
    `transformers.AutoTokenizer` raises the same Application Control error the
    model classes do. `CLIPTokenizerFast` is also gone in transformers 5.x. But
    `tokenizer.json` *is* the whole tokenizer — merges, vocabulary, and the
    post-processor that wraps the sequence in `<|startoftext|>`/`<|endoftext|>` —
    and the `tokenizers` extension that reads it loads fine. Fewer moving parts,
    and none of them torch.
    """
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        return None, f"onnxruntime/tokenizers missing ({exc})"
    except Exception as exc:                           # noqa: BLE001
        return None, f"onnxruntime unusable — {type(exc).__name__}: " \
                     f"{str(exc)[:160]}"

    t0 = time.time()
    try:
        from huggingface_hub import hf_hub_download
        # The tokenizer comes from the checkpoint the *index* was built with, so
        # a wrong ONNX repo cannot quietly bring its own vocabulary along.
        tok_path = hf_hub_download(config.VSEARCH_MODEL, "tokenizer.json")
        graph = hf_hub_download(config.VSEARCH_ONNX_REPO,
                                config.VSEARCH_ONNX_TEXT)
    except Exception as exc:                           # noqa: BLE001
        return None, f"text tower unavailable — {type(exc).__name__}: " \
                     f"{str(exc)[:160]}"

    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = max(2, (os.cpu_count() or 4) - 1)
        sess = ort.InferenceSession(graph, so,
                                    providers=["CPUExecutionProvider"])
        tok = Tokenizer.from_file(tok_path)
    except Exception as exc:                           # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"

    names = [o.name for o in sess.get_outputs()]
    if "text_embeds" not in names:
        # Refuse rather than pool something plausible. An unprojected vector
        # normalises and ranks exactly like a real one; the failure would show up
        # as bad results, not as an error, and nobody would know where to look.
        return None, ("that ONNX export has no `text_embeds` output "
                      f"(got {names}) — it is not a projected text tower")

    log(f"image query encoder ready — {config.VSEARCH_ONNX_REPO} text tower "
        f"via onnxruntime in {time.time() - t0:.0f}s (no torch)")
    return _Onnx(sess, tok, "text_embeds"), ""


def get_encoder():
    """Load CLIP once, on first use. Returns None if it cannot be had.

    Mirrors `encoder.get_encoder` — module singleton, tried-once, logs and
    degrades rather than raising — with three departures, all forced:

    * It pins **CPU** by default. The processing plane owns both cards for the
      whole session, and a query encoder that takes VRAM is how a GPU worker dies
      mid-pass. One query on four vCPUs is around 0.3 s, which is fine for a
      search box.
    * It loads on the first query rather than at import, so a session that only
      ever uses mode 1 — which needs no model at all — never pays the 1.7 GB.
    * It falls back from torch to an ONNX text tower. Both towers are wanted
      where torch runs; where torch is *forbidden* — Smart App Control, which
      blocks unsigned DLLs and cannot be switched off without reinstalling
      Windows — the text tower alone still answers the query people actually
      type. Order matters: torch first, because it is the only one of the two
      that can also encode an uploaded image.
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

        why = []
        for name, loader in (("torch", _load_torch), ("onnx", _load_onnx)):
            enc, reason = loader()
            if enc is not None:
                _ENCODER = enc
                return _ENCODER
            why.append(f"{name}: {reason}")
            log(f"image query encoder — {name} route unavailable: {reason}")

        # Both reasons, because "no encoder" sends people to the wrong problem.
        # A blocked torch and a missing download need opposite fixes.
        _ENC_ERROR = "; ".join(why)
        return None



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


def _spread(hits: list, fps: dict, limit: int,
            gap_s: float = None, per_video: int = None) -> list:
    """One hit per moment. Takes ranked `[(key, idx, score)]`, returns a subset.

    Ranking by score alone is correct and unreadable. Frames next to each other
    in one reel are the same photograph, so the honest top twenty-four is
    twenty-four views of one second, and the interface shows the user a wall of
    one answer — the failure the archive's own rule names: one answer per card.

    Greedy from the top, so the best frame of any moment is the one kept and no
    hit is ever replaced by a worse neighbour. A frame is dropped only when the
    same reel already holds a hit within `gap_s` seconds of it, or when that reel
    has already filled its share of the page.

    The gap is per reel, because it is stated in seconds and frame indices are
    not. `any()` runs over at most `per_video` accepted indices — six by default
    — so this costs nothing next to the matmul that produced the scores.
    """
    gap_s = _SPREAD_S if gap_s is None else float(gap_s)
    cap = _PER_VIDEO if per_video is None else int(per_video)
    if gap_s <= 0 and cap <= 0:
        return hits[:limit]
    kept: list = []
    taken: dict = {}
    for key, idx, score in hits:
        acc = taken.setdefault(key, [])
        if cap > 0 and len(acc) >= cap:
            continue
        gap = max(1, int(round(gap_s * (fps.get(key) or _SPREAD_FPS))))
        if any(abs(idx - a) < gap for a in acc):
            continue
        acc.append(idx)
        kept.append((key, idx, score))
        if len(kept) >= limit:
            break
    return kept


def _shape(conn, space: str, hits: list, fps: dict = None) -> list:
    fps = _fps(conn, [h[0] for h in hits]) if fps is None else fps
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

    `limit` here is a *candidate* count, not a page size. The caller spreads the
    result to one hit per moment and so needs more rows than it will show.

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


# Every empty result carries a `cause` beside its `reason`. The prose is for a
# person and changes freely; the cause is a fixed token an interface may branch
# on, and it exists because branching on the prose does not work. The frames lane
# decided "that search needs a model, and it is not installed here" by testing
# the reason against /torch|transformers|module|encoder|model/i — which matched
# `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`
# on the word *Model* inside a class name, and told somebody to install a model
# that was already loaded and working. Nine causes, and only one of them means
# anything is missing:
#
#   no_numpy         numpy is not importable
#   no_index         nothing resident in that space — the index is not built
#   no_vectors       that reel has no frame vectors in that space
#   empty_query      the query vector normalises to zero
#   no_match         the archive was compared, and nothing in it resembles this
#   no_encoder       no query encoder could be loaded at all
#   encode_failed    an encoder *is* loaded and the call raised — a defect here,
#                    not a missing install, and it needs a traceback not a download
#   no_vision_tower  the text tower is present and the vision tower is not
#   bad_image        the upload is not a decodable image
#
# A tenth, `bad_query`, belongs to the vocabulary but is emitted by the route in
# `atlas/server.py` rather than by anything here: it means the request was never
# a search — a frame reference that does not parse, a space that cannot hold a
# CLIP vector — and it is separate from `no_match` because "that was not a
# question" and "the answer is nothing" are different sentences.
def search_vector(conn: sqlite3.Connection, q, space: str,
                  limit: int = 40, exclude_key: str = "") -> dict:
    """Rank frames against a query vector already in `space`."""
    try:
        import numpy as np
    except ImportError:
        return {"hits": [], "cause": "no_numpy", "reason": "numpy missing"}
    if not ready(space):
        return {"hits": [], "cause": "no_index",
                "reason": f"no resident index for {space}"}
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(q))
    if n == 0:
        return {"hits": [], "cause": "empty_query",
                "reason": "the query vector is empty"}
    q = q / n
    limit = max(int(limit), 1)

    with _LOCK:
        res = _RESIDENT.get(space) or {}
    # `ids` is an ndarray, so `or ()` would ask it for a truth value and raise.
    # `videos` is a plain list and is read that way below.
    ids = res.get("ids")
    total = 0 if ids is None else len(ids)
    pool = min(total, max(limit * _POOL, _POOL_MIN)) or limit

    cands = _resident_hits(space, q, np, pool, exclude_key)
    if cands is not None:
        # Every reel in the index was compared frame by frame, so the count
        # reported is the truth and not the size of a shortlist.
        seen = len(res.get("videos") or ())
        if exclude_key and exclude_key in (res.get("videos") or ()):
            seen -= 1
        fps = _fps(conn, [h[0] for h in cands])
        hits = _spread(cands, fps, limit)
        if len(hits) < limit and pool < total:
            # The pool was all one shot. Widening is cheaper than showing a thin
            # page, and it happens at most once.
            cands = _resident_hits(space, q, np, total, exclude_key) or cands
            fps = _fps(conn, [h[0] for h in cands])
            hits = _spread(cands, fps, limit)
        if not hits:
            return {"hits": [], "space": space, "searched_videos": seen,
                    "cause": "no_match",
                    "reason": "nothing in this space resembles it"}
        return {"hits": _shape(conn, space, hits, fps), "space": space,
                "searched_videos": seen}

    shortlist = [k for k, _s in _coarse(space, q, np,
                                        config.VSEARCH_CANDIDATES)
                 if k != exclude_key]
    if not shortlist:
        return {"hits": [], "cause": "no_match",
                "reason": "nothing in this space resembles it"}
    cands = [h for h in _exact(conn, space, q, np, shortlist,
                               max(limit * _POOL, _POOL_MIN))
             if h[0] != exclude_key]
    fps = _fps(conn, [h[0] for h in cands])
    hits = _spread(cands, fps, limit)
    return {"hits": _shape(conn, space, hits, fps), "space": space,
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
        return {"hits": [], "cause": "no_numpy", "reason": "numpy missing"}
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
        return {"hits": [], "cause": "no_vectors",
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
        return {"hits": [], "cause": "no_encoder",
                "reason": _ENC_ERROR or "no query encoder"}
    try:
        q = enc.text(query)
    except Exception as exc:                           # noqa: BLE001
        # `encode_failed`, never `no_encoder`: the encoder above loaded, so
        # nothing is missing and no download fixes this. It is a fault in the
        # encode path — the shape transformers hands back changed once already,
        # which is what `hfcompat` is for — and the interface must send somebody
        # to a traceback rather than to a model registry.
        return {"hits": [], "cause": "encode_failed",
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
        return {"hits": [], "cause": "no_encoder",
                "reason": _ENC_ERROR or "no query encoder"}
    if isinstance(enc, _Onnx):
        # An encoder is loaded and this mode still cannot run, which no wording
        # about reading the image would convey. The text tower is 472 MB and the
        # vision tower 1.16 GB; only the first is fetched.
        return {"hits": [], "cause": "no_vision_tower",
                "reason": "this host has the text tower only — "
                          "searching by an uploaded image needs "
                          "torch, which cannot run here"}
    if not data:
        # Reached when a multipart part arrives with no filename, which is what
        # `FormData.append('file', blob)` sends for a pasted screenshot if the
        # third argument is omitted. Worth its own sentence: "could not read the
        # image" sends somebody to look at a picture that was never uploaded.
        return {"hits": [], "cause": "bad_image",
                "reason": "the upload was empty — no image bytes arrived"}
    try:
        q = enc.image(data)
    except BadImage as exc:
        return {"hits": [], "cause": "bad_image",
                "reason": f"could not read the image: {str(exc)[:120]}"}
    except Exception as exc:                           # noqa: BLE001
        # The vision tower is loaded and it raised. Not `bad_image`: the picture
        # decoded, so blaming the file is a wrong answer, and one that costs
        # somebody a round of re-exporting a screenshot that was always fine.
        return {"hits": [], "cause": "encode_failed",
                "reason": f"encoding the picture failed: "
                          f"{type(exc).__name__}: {str(exc)[:120]}"}
    out = search_vector(conn, q, "clip", limit)
    out["query"] = {"image_bytes": len(data), "space": "clip"}
    return out
