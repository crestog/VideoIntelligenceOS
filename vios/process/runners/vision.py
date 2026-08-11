"""
vios.process.runners.vision — what is on the screen, in every frame of it.

Eight passes over the **complete** frame set, and they are deliberately
redundant. Two OCR engines read the same pixels because a detector-plus-
recogniser drops stylised type while an end-to-end vision-language model invents
plausible words, and the two failures do not overlap. Two embedding towers index
the same frames because SigLIP and CLIP retrieve differently, so a moment found
by either is found. Objects come from a closed vocabulary with boxes, faces from
a geometric detector, scale from a depth model.

These passes used to read `frames/index.json` — one keyframe per shot. A
900-frame reel was judged on about thirty images, which is why a caption that
appeared for two seconds between cuts existed nowhere in the database. They now
read `allframes`, which the structure stage already extracted and *proved*
complete, and every frame goes through every model that wants pixels.

Total coverage is affordable because of two things and not because anything is
skipped:

  batching     a T4 costs roughly the same for one image as for thirty-two, so
               a 900-frame reel is ~28 forward passes, not 900
  run-length   identical consecutive readings collapse into one row spanning
               `frame_idx…frame_hi`. "SUBSCRIBE was on screen for frames
               100–142" is one row that still answers "what was on screen at
               frame 117" exactly. Per-frame numbers go to packed columnar
               arrays instead of a row each

Nothing here is asked for a timestamp. Every claim carries the frame index and
the time the extractor read out of the container's presentation timestamps; the
store derives the shot. A model's opinion about when something happened is not
evidence, and this is the layer where that rule is enforced by construction.
"""

from __future__ import annotations

import hashlib
import math

from .. import registry
from .base import Emission, Job, SkipPass, device_and_dtype, torch_dtype


def _np():
    import numpy  # noqa: PLC0415
    return numpy


def _read(path: str):
    import cv2  # noqa: PLC0415
    return cv2.imread(path)


def _batch(job: Job, default: int = 32) -> int:
    return max(1, int(job.params.get("batch", default)))


def _coverage(job: Job, read: int) -> dict:
    """The completeness note every pass in this module returns.

    Written as `frames` against `frames_available` on purpose: a pass that read
    812 of 900 frames must be visibly different in the database from one that
    read all 900, and the only way to keep that honest is to record both
    numbers every time rather than only when they differ.
    """
    total = job.frame_count()
    note = {"frames": read, "frames_available": total or read}
    if total and read < total:
        note["incomplete"] = total - read
    return note


# ══════════════════════════════════════════════════════════════════════════
# visual-embed — SigLIP-2, every frame, read by four other things
# ══════════════════════════════════════════════════════════════════════════

_SIGLIP = "visual-embed"


def _siglip(job: Job) -> dict:
    """The SigLIP-2 tower, cached under the visual-embed component's key.

    Loaded through the registry entry rather than `job.component` because the
    tagger and the aesthetic probe both need this model while being different
    components — and because the cohort packer counted its 1.8 GB exactly once,
    under this key, on the strength of that sharing.
    """
    comp = registry.BY_ID[_SIGLIP]
    device, dtype = device_and_dtype(job.resources)

    def loader():
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415
        model = AutoModel.from_pretrained(
            comp.model, torch_dtype=torch_dtype(dtype) if device == "cuda"
            else torch.float32)
        proc = AutoProcessor.from_pretrained(comp.model)
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
        return {"model": model, "processor": proc, "device": device}

    return job.cache.get(comp.load_key, loader)


def _embed_frames(job: Job, bundle: dict, space: str) -> tuple:
    """Run an image tower over every frame. Returns `(indices, times, matrix)`.

    Shared by both embedding passes because the loop is identical and only the
    tower differs — and because a divergence between how SigLIP and CLIP are
    fed would make their disagreement uninterpretable, which is the one thing
    that would waste having both.
    """
    np = _np()
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    model, proc = bundle["model"], bundle["processor"]
    device = bundle["device"]
    idxs: list = []
    times: list = []
    blocks: list = []

    for b_idx, b_t, b_paths in job.frame_batches(_batch(job, 32)):
        images, keep_i, keep_t = [], [], []
        for i, t, p in zip(b_idx, b_t, b_paths):
            try:
                images.append(Image.open(p).convert("RGB"))
            except OSError:
                continue
            keep_i.append(int(i))
            keep_t.append(float(t))
        if not images:
            continue
        with torch.no_grad():
            inputs = proc(images=images, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(
                        model.dtype)
            vecs = model.get_image_features(**inputs)
            vecs = (vecs / vecs.norm(dim=-1, keepdim=True)
                    ).float().cpu().numpy()
        blocks.append(vecs)
        idxs.extend(keep_i)
        times.extend(keep_t)

    if not blocks:
        raise SkipPass(f"no frame could be read for the {space} tower")
    return idxs, times, np.concatenate(blocks, axis=0)
def _pool_by_shot(job: Job, idxs, times, matrix) -> list:
    """Mean-pool per-frame embeddings into one vector per shot.

    The shot-level rows are kept because `tag` and the aesthetic probe read
    `store.vectors_for(key, "siglip2")`, and because a shot vector is the right
    granularity for "what is this scene about" while a frame vector is the right
    one for "find this exact instant". Both are true at once; storing only the
    finer one would silently change what those two passes mean.

    Mean pooling of L2-normalised embeddings, re-normalised, is the standard
    construction: a shot whose frames are all alike lands on them, and a shot
    that changes mid-way lands between.
    """
    np = _np()
    buckets: dict = {}
    for row, t in enumerate(times):
        buckets.setdefault(job.shot_at(float(t)), []).append(row)
    out = []
    for shot_idx in sorted(buckets):
        v = matrix[buckets[shot_idx]].mean(axis=0)
        v = v / (float(np.linalg.norm(v)) + 1e-9)
        out.append((shot_idx, v))
    return out


def visual_embed(job: Job) -> Emission:
    """SigLIP-2 over every frame, plus per-shot and whole-video pooling.

    The per-frame matrix is what makes "find the moment that looks like this"
    answerable to the frame; it is stored as one packed row per video rather
    than 900 rows of floats, because the registry's own note about `perframe`
    — 27 million single-row-queried rows — is what that costs otherwise.
    """
    np = _np()
    try:
        import torch  # noqa: PLC0415  (imported for the clear skip message)
        from PIL import Image  # noqa: PLC0415,F401
    except ImportError:
        raise SkipPass("torch/Pillow are not installed") from None

    bundle = _siglip(job)
    idxs, times, matrix = _embed_frames(job, bundle, "siglip2")

    em = Emission()
    em.frame_vector_set("siglip2", idxs, matrix)
    for shot_idx, v in _pool_by_shot(job, idxs, times, matrix):
        em.vector("siglip2", [float(x) for x in v], shot_idx=shot_idx)
    pooled = matrix.mean(axis=0)
    pooled = pooled / (float(np.linalg.norm(pooled)) + 1e-9)
    em.vector("siglip2", [float(x) for x in pooled])

    em.notes = {**_coverage(job, len(idxs)), "dim": int(matrix.shape[1]),
                "space": "siglip2"}
    return em


# ══════════════════════════════════════════════════════════════════════════
# clip-embed — a second visual space over the identical frames
# ══════════════════════════════════════════════════════════════════════════

def clip_embed(job: Job) -> Emission:
    """CLIP ViT-L/14 over every frame.

    Two embedding spaces over identical frames is the entire point. SigLIP and
    CLIP were trained on different data with different objectives and they fail
    differently: a query that lands nothing in one often lands in the other, so
    a moment retrievable by either is retrievable. Where they disagree about
    what a frame resembles, that disagreement is itself evidence — which is why
    both are stored rather than averaged into one consensus vector that would
    be worse than both.

    fp16, because Kaggle's T4s are sm_75: no BF16, no FP8, no FlashAttention-2.
    """
    try:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415,F401
        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415
    except ImportError:
        raise SkipPass("transformers/torch/Pillow are not installed") from None

    device, dtype = device_and_dtype(job.resources)

    def loader():
        model = CLIPModel.from_pretrained(
            job.component.model,
            torch_dtype=torch_dtype(dtype) if device == "cuda"
            else torch.float32)
        proc = CLIPProcessor.from_pretrained(job.component.model)
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
        return {"model": model, "processor": proc, "device": device}

    bundle = job.cache.get(job.component.load_key, loader)
    idxs, times, matrix = _embed_frames(job, bundle, "clip")

    em = Emission()
    em.frame_vector_set("clip", idxs, matrix)
    for shot_idx, v in _pool_by_shot(job, idxs, times, matrix):
        em.vector("clip", [float(x) for x in v], shot_idx=shot_idx)
    em.notes = {**_coverage(job, len(idxs)), "dim": int(matrix.shape[1]),
                "space": "clip"}
    return em


def label_matrix(job: Job, labels) -> tuple:
    """Embed a label vocabulary once, and cache the matrix for the cohort.

    Returns `(matrix, names)` or `(None, names)` when the tower cannot be
    loaded — the tagger treats that as a skip rather than an error, because a
    session that cannot reach Hugging Face should still produce a database from
    everything else.

    SigLIP's processor needs `padding="max_length"`; with ordinary padding the
    text tower silently produces degraded embeddings, and the symptom is not an
    exception but a similarity matrix that ranks nothing usefully.
    """
    names = list(labels)
    digest = hashlib.sha1(
        " ".join(names).encode("utf-8")).hexdigest()[:10]
    key = f"siglip2-labels:{digest}"
    if job.cache.has(key):
        return job.cache.get(key, lambda: None), names

    try:
        import torch  # noqa: PLC0415
        bundle = _siglip(job)
    except Exception as exc:
        job.note(f"label embedding unavailable ({type(exc).__name__})")
        return None, names

    model, proc, device = bundle["model"], bundle["processor"], bundle["device"]
    with torch.no_grad():
        inputs = proc(text=names, padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        vecs = model.get_text_features(**inputs)
        vecs = (vecs / vecs.norm(dim=-1, keepdim=True)).float().cpu().numpy()
    return job.cache.get(key, lambda: vecs), names


# ══════════════════════════════════════════════════════════════════════════
# aesthetic — classical measures per frame, plus a zero-shot probe
# ══════════════════════════════════════════════════════════════════════════

_GOOD = ("a beautiful, well composed photograph with pleasing lighting",
         "a crisp professional video frame, sharp and well exposed")
_BAD = ("a blurry, badly lit amateur snapshot",
        "an underexposed noisy video frame with motion blur")


def aesthetic(job: Job) -> Emission:
    """Sharpness, exposure, clipping, noise and colourfulness — per frame.

    All five are closed-form measures over pixels, which is why they are the
    ones to trust when they disagree with the model probe below. They land as
    packed `frame_metric` arrays rather than claims: five numbers × 900 frames
    is 4,500 rows as evidence and five rows as columns, and no query wants them
    one at a time.

    The probe is the difference between two cosine similarities: how much more
    the frame looks like the "good" prompts than the "bad" ones. It is a
    *relative* score, useful for ranking frames within this archive, and it is
    not the LAION aesthetic MLP — that head is trained on CLIP ViT-L/14
    features and would be meaningless applied to SigLIP vectors. Saying so here
    matters more than having a number that looks official: a score whose
    provenance is wrong poisons every pattern found downstream.
    """
    import cv2  # noqa: PLC0415
    np = _np()

    em = Emission()
    idxs: list = []
    series: dict = {k: [] for k in ("sharpness", "exposure", "clipping",
                                    "noise", "colourfulness")}

    for b_idx, _b_t, b_paths in job.frame_batches(_batch(job, 32)):
        for i, path in zip(b_idx, b_paths):
            img = _read(path)
            if img is None:
                continue
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(grey, cv2.CV_64F).var())
            exposure = float(grey.mean() / 255.0)
            # Clipping, not brightness: the share of pixels pinned at either
            # end. A frame can average 0.5 and still have blown highlights, and
            # it is the clipping that makes footage unusable.
            clipped = float(((grey <= 2) | (grey >= 253)).mean())
            # Noise as the median absolute deviation of a high-pass residual —
            # cheap, and robust to the edges that would fool a plain variance.
            residual = grey.astype("float32") - cv2.GaussianBlur(
                grey.astype("float32"), (0, 0), 1.2)
            noise = float(np.median(np.abs(residual)) * 1.4826)
            # Hasler–Süsstrunk colourfulness, the standard closed-form metric.
            b, g, r = (img[:, :, c].astype("float32") for c in range(3))
            rg, yb = r - g, 0.5 * (r + g) - b
            colourful = float(math.hypot(rg.std(), yb.std())
                              + 0.3 * math.hypot(rg.mean(), yb.mean()))

            idxs.append(int(i))
            series["sharpness"].append(sharp)
            series["exposure"].append(exposure)
            series["clipping"].append(clipped)
            series["noise"].append(noise)
            series["colourfulness"].append(colourful)

    if not idxs:
        raise SkipPass("no readable frames")

    for name, values in series.items():
        em.frame_metric(name, idxs, values)

    # Whole-video summaries stay claims: they are what a human reads and what a
    # cross-archive query filters on, and there are five of them, not 4,500.
    sharps = series["sharpness"]
    exposures = series["exposure"]
    em.claim("style", "sharpness",
             "soft" if np.median(sharps) < 60 else
             "sharp" if np.median(sharps) > 250 else "normal",
             num=round(float(np.median(sharps)), 2))
    em.claim("style", "exposure",
             "underexposed" if np.mean(exposures) < 0.3 else
             "overexposed" if np.mean(exposures) > 0.72 else "well exposed",
             num=round(float(np.mean(exposures)), 4))
    em.claim("style", "clipping",
             num=round(float(np.mean(series["clipping"])), 5))
    em.claim("style", "noise", num=round(float(np.median(series["noise"])), 3))
    em.claim("style", "colourfulness",
             num=round(float(np.mean(series["colourfulness"])), 2))

    probe = _aesthetic_probe(job, em, idxs)
    em.notes = {**_coverage(job, len(idxs)), "probe": probe}
    return em


def _aesthetic_probe(job: Job, em: Emission, idxs) -> str:
    """Per-frame relative quality against the frame embeddings, if they exist.

    Reads the packed `frame_vector` row written by `visual-embed` and falls back
    to the per-shot vectors, because a database restored from a shard predating
    the per-frame tables still has the shot rows and a coarser score is worth
    more than none. Returns which source was used, so the note says so.
    """
    np = _np()
    matrix, names = label_matrix(job, list(_GOOD) + list(_BAD))
    if matrix is None:
        return "none"
    good_n = len(_GOOD)

    def score(vec) -> float:
        v = np.asarray(vec, dtype="float32")
        v = v / (float(np.linalg.norm(v)) + 1e-9)
        sims = matrix @ v
        return float(sims[:good_n].mean() - sims[good_n:].mean())

    packed = job.store.frame_vectors(job.key, "siglip2")
    if packed:
        row = packed[0]
        em.frame_metric("aesthetic", row["frames"],
                        [score(v) for v in row["values"]])
        return "per-frame"

    rows = job.store.vectors_for(job.key, "siglip2")
    scored = 0
    for r in rows:
        if r.get("shot_idx") is None:
            continue
        em.claim("style", "aesthetic", num=round(score(r["values"]), 5),
                 shot_idx=r["shot_idx"], confidence=0.5)
        scored += 1
    if not scored:
        job.note("no visual vectors yet — the aesthetic probe was skipped, "
                 "the classical measures were not")
    return "per-shot" if scored else "none"


# ══════════════════════════════════════════════════════════════════════════
# ocr — PP-OCRv5, every frame, at source resolution
# ══════════════════════════════════════════════════════════════════════════

def _paddle_lines(result) -> list:
    """Normalise PaddleOCR's return shape across 2.x and 3.x.

    The two versions return different structures for the same call and the
    3.x rename happened mid-2025, so pinning a version in the notebook is a
    promise about a package index rather than about behaviour. Reading both
    shapes costs twenty lines and removes an entire class of Kaggle-only
    breakage.
    """
    out: list = []
    if not result:
        return out
    first = result[0]
    if isinstance(first, dict):
        for page in result:
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = (page.get("rec_polys") or page.get("dt_polys") or [])
            for i, text in enumerate(texts):
                out.append((str(text),
                            float(scores[i]) if i < len(scores) else 0.0,
                            polys[i] if i < len(polys) else None))
        return out
    pages = result if isinstance(first, list) else [result]
    for page in pages:
        for line in (page or []):
            try:
                box, (text, score) = line[0], line[1]
            except (TypeError, ValueError, IndexError):
                continue
            out.append((str(text), float(score), box))
    return out


def _rect(poly, w: int, h: int) -> dict:
    np = _np()
    try:
        pts = np.asarray(poly, dtype="float32").reshape(-1, 2)
    except (ValueError, TypeError):
        return {}
    x0, y0 = pts[:, 0].min() / w, pts[:, 1].min() / h
    x1, y1 = pts[:, 0].max() / w, pts[:, 1].max() / h
    return {"x": round(float(x0), 4), "y": round(float(y0), 4),
            "w": round(float(x1 - x0), 4), "h": round(float(y1 - y0), 4)}


def _screen_text(lines) -> str:
    """The frame's readable text as one canonical string.

    Sorted and joined so that two frames showing the same words in a different
    detection order collapse into one run instead of alternating forever. The
    individual strings are still emitted separately below; this is only the key
    that decides where a run begins and ends.
    """
    return " │ ".join(sorted({t for t in lines if t}))


def ocr(job: Job) -> Emission:
    """Read the burned-in text of every frame, then collapse it in time.

    Reels caption themselves, and those words never appear in the transcript.
    The previous version read one keyframe per shot, so a caption that appeared
    and vanished between two cuts was invisible — which is most of them, since
    on-screen text is cut to the beat and not to the shot.

    Every frame is read now, at **source resolution**: a 24 px caption on a
    1080×1920 reel is 8 px at the 384 px analysis size, which is below what any
    recogniser can read. Identical consecutive readings become one run-length
    row, so a caption held for two seconds is one claim spanning sixty frames
    and still answers "what was on screen at frame 117" exactly.
    """
    langs = list(job.component.params.get("languages", ["en"]))
    floor = float(job.component.params.get("min_confidence", 0.6))

    def loader():
        from paddleocr import PaddleOCR  # noqa: PLC0415
        gpu = bool(job.resources.get("gpu_count"))

        # PaddleOCR has had three constructor generations and Kaggle's image
        # pins whichever one it pins this month. 2.x takes use_gpu/show_log;
        # 3.x removed both and raises ValueError("Unknown argument: use_gpu")
        # rather than TypeError, which is why the original fallback — an
        # `except TypeError` — never fired and every OCR pass in the run
        # declined. So the kwargs are tried in order, newest first, and the
        # guard catches both exception types plus the RuntimeError some builds
        # raise from their own argument checker.
        #
        # Newest first matters: on 2.x the 3.x kwargs are rejected immediately
        # and cost nothing, whereas the reverse order would silently accept
        # `use_gpu` on a 3.x build that has quietly started ignoring it.
        attempts = [
            ({"lang": None, "use_textline_orientation": True,
              "device": "gpu:0" if gpu else "cpu"}, "3.x"),
            ({"lang": None, "device": "gpu:0" if gpu else "cpu"}, "3.x minimal"),
            ({"lang": None, "use_angle_cls": True, "show_log": False,
              "use_gpu": gpu}, "2.x"),
            ({"lang": None}, "bare"),
        ]

        engines, why = {}, {}
        for lang in langs:
            for kwargs, label in attempts:
                try:
                    engines[lang] = PaddleOCR(**{**kwargs, "lang": lang})
                    break
                except (TypeError, ValueError, RuntimeError) as exc:
                    why[lang] = f"{label}: {type(exc).__name__}: {exc}"
                except ImportError:
                    raise
                except Exception as exc:      # noqa: BLE001
                    # A download failure or a missing model file is not an
                    # argument problem and retrying with fewer kwargs will not
                    # help — stop on this language and keep the others.
                    why[lang] = f"{label}: {type(exc).__name__}: {exc}"
                    break
        for lang in langs:
            if lang not in engines and lang in why:
                job.note(f"PP-OCR could not initialise '{lang}' — {why[lang]}")
        return engines

    try:
        engines = job.cache.get(job.component.load_key, loader)
    except ImportError:
        raise SkipPass("paddleocr is not installed") from None
    if not engines:
        raise SkipPass("no OCR language could be initialised")

    em = Emission()
    readings: list = []              # (frame_idx, frame_t, canonical text)
    per_string: dict = {}            # text -> [frame_idx, frame_t, ...]
    regions: dict = {}               # text -> (score, rect)
    read = 0
    failures = 0

    for b_idx, b_t, b_paths in job.frame_batches(_batch(job, 8), tier="full"):
        for i, t, path in zip(b_idx, b_t, b_paths):
            img = _read(path)
            if img is None:
                continue
            read += 1
            h, w = img.shape[:2]
            found: dict = {}
            for lang, engine in engines.items():
                try:
                    result = (engine.predict(img) if hasattr(engine, "predict")
                              else engine.ocr(img, cls=True))
                except Exception as exc:
                    failures += 1
                    if failures <= 3:
                        job.note(f"{lang} OCR failed on frame {i}: "
                                 f"{type(exc).__name__}: {exc}")
                    continue
                for text, score, poly in _paddle_lines(result):
                    text = text.strip()
                    if not text or score < floor:
                        continue
                    prev = found.get(text)
                    if prev is None or score > prev[0]:
                        found[text] = (score, _rect(poly, w, h))

            readings.append((int(i), float(t), _screen_text(found)))
            for text, (score, rect) in found.items():
                spans = per_string.setdefault(text, [])
                spans.append((int(i), float(t)))
                best = regions.get(text)
                if best is None or score > best[0]:
                    regions[text] = (score, rect)

    if not read:
        raise SkipPass("no frame could be read")
    if failures:
        job.note(f"{failures} per-frame OCR failures out of {read} frames")

    # The canonical-screen run: what the frame said, as one row per stable span.
    rows = em.frame_runs("ocr", "screen_text", readings, confidence=0.8)

    # Each individual string as its own run, so a caption that persists while
    # another appears beside it is still one claim rather than being broken by
    # its neighbour's arrival.
    for rank, (text, spans) in enumerate(
            sorted(per_string.items(), key=lambda kv: -len(kv[1]))):
        score, rect = regions.get(text, (0.0, {}))
        lo_i, lo_t = spans[0]
        prev_i, prev_t = spans[0]
        for i, t in spans[1:] + [(None, None)]:
            if i is not None and i - prev_i <= 1:
                prev_i, prev_t = i, t
                continue
            em.frame_claim(lo_i, lo_t, "ocr", "text", text,
                           frame_hi=(prev_i if prev_i != lo_i else None),
                           frame_t1=(prev_t if prev_i != lo_i else None),
                           confidence=round(score, 4), ordinal=rank)
            if rect:
                em.frame_claim(lo_i, lo_t, "ocr", "text_region",
                               {**rect, "text": text},
                               frame_hi=(prev_i if prev_i != lo_i else None),
                               confidence=round(score, 4), ordinal=rank)
            if i is None:
                break
            lo_i, lo_t = i, t
            prev_i, prev_t = i, t

    covered = sum(1 for _, _, v in readings if v)
    em.claim("ocr", "text_density",
             num=round(len(per_string) / max(job.duration or 1.0, 0.001), 3))
    em.claim("ocr", "text_coverage", num=round(covered / max(read, 1), 4))
    em.notes = {**_coverage(job, read), "strings": len(per_string),
                "runs": rows, "frames_with_text": covered,
                "languages": langs, "tier": "full"}
    return em


# ══════════════════════════════════════════════════════════════════════════
# ocr-alt — Florence-2, the same frames, a different architecture
# ══════════════════════════════════════════════════════════════════════════

def ocr_alt(job: Job) -> Emission:
    """The same frames, a different architecture, and both answers kept.

    Where this and PP-OCR agree, the words can be trusted; where they diverge,
    both are stored and the interface can show the disagreement. Agreement
    between two unrelated systems is a far stronger signal than either one's own
    confidence score, which is calibrated only against its own training set.

    Florence-2 generates rather than detects, so it runs one frame at a time
    and is the slowest pass in the perception stage. That is accepted: the cost
    of the second reader is time, and time is explicitly not the constraint.
    """
    try:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from transformers import (AutoModelForCausalLM,  # noqa: PLC0415
                                  AutoProcessor)
    except ImportError:
        raise SkipPass("transformers/torch are not installed") from None

    device, dtype = device_and_dtype(job.resources)

    def loader():
        model = AutoModelForCausalLM.from_pretrained(
            job.component.model, trust_remote_code=True,
            torch_dtype=torch_dtype(dtype) if device == "cuda"
            else torch.float32)
        proc = AutoProcessor.from_pretrained(job.component.model,
                                             trust_remote_code=True)
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
        return {"model": model, "processor": proc}

    try:
        bundle = job.cache.get(job.component.load_key, loader)
    except Exception as exc:
        raise SkipPass(f"Florence-2 could not be loaded: "
                       f"{type(exc).__name__}") from None
    model, proc = bundle["model"], bundle["processor"]

    em = Emission()
    readings: list = []
    read, failures = 0, 0

    for b_idx, b_t, b_paths in job.frame_batches(_batch(job, 4), tier="full"):
        images, keep = [], []
        for i, t, path in zip(b_idx, b_t, b_paths):
            try:
                images.append(Image.open(path).convert("RGB"))
            except OSError:
                continue
            keep.append((int(i), float(t)))
        if not images:
            continue
        read += len(images)
        try:
            with torch.no_grad():
                inputs = proc(text=["<OCR>"] * len(images), images=images,
                              return_tensors="pt")
                if device == "cuda":
                    inputs = {k: (v.to("cuda").to(model.dtype)
                                  if k == "pixel_values" else v.to("cuda"))
                              for k, v in inputs.items()}
                out = model.generate(input_ids=inputs["input_ids"],
                                     pixel_values=inputs["pixel_values"],
                                     max_new_tokens=256, num_beams=1,
                                     do_sample=False)
            decoded = proc.batch_decode(out, skip_special_tokens=False)
        except Exception as exc:
            failures += len(images)
            if failures <= 8:
                job.note(f"Florence-2 failed on frames "
                         f"{keep[0][0]}–{keep[-1][0]}: "
                         f"{type(exc).__name__}: {exc}")
            continue
        for (i, t), raw, image in zip(keep, decoded, images):
            try:
                parsed = proc.post_process_generation(
                    raw, task="<OCR>", image_size=image.size)
                text = str(parsed.get("<OCR>", "")).strip()
            except Exception:
                text = ""
            readings.append((i, t, text))

    if not read:
        raise SkipPass("no frame could be read")
    if failures:
        job.note(f"{failures} of {read} frames failed in Florence-2")

    rows = em.frame_runs("ocr", "text", readings, confidence=0.7)
    if not rows:
        raise SkipPass("Florence-2 read no text in any frame")
    covered = sum(1 for _, _, v in readings if v)
    em.notes = {**_coverage(job, read), "runs": rows,
                "frames_with_text": covered, "failures": failures,
                "tier": "full"}
    return em

# ══════════════════════════════════════════════════════════════════════════
# detect — YOLO11, every frame
# ══════════════════════════════════════════════════════════════════════════

def detect(job: Job) -> Emission:
    """Boxes, classes, counts and screen share, from a closed vocabulary.

    Eighty classes detected reliably beat a thousand guessed, because these
    outputs are inputs to later arithmetic. "The subject fills 40% of frame in
    the hook and 12% by the payoff" is a computable sentence only if screen
    share is a number attached to a box, on every frame, not on a keyframe that
    happened to be sampled.

    Two shapes come out of this. Presence is a **run** per class — "person from
    frame 0 to 418" is one row that answers any frame in that span — and screen
    share is a packed per-frame **metric** per class, because a share is a
    number to plot and threshold, not a string to search.
    """
    p = job.component.params

    def loader():
        from ultralytics import YOLO  # noqa: PLC0415
        model = YOLO(f"{job.component.weights}.pt")
        if job.resources.get("gpu_count"):
            model.to("cuda")
        return model

    try:
        model = job.cache.get(job.component.load_key, loader)
    except ImportError:
        raise SkipPass("ultralytics is not installed") from None

    em = Emission()
    idxs: list = []
    times: list = []
    per_frame: list = []             # [{cls: (count, max_share)}]
    totals: dict = {}
    read, failures = 0, 0

    for b_idx, b_t, b_paths in job.frame_batches(_batch(job, 16), tier="full"):
        try:
            results = model.predict(list(b_paths),
                                    conf=float(p.get("conf", 0.35)),
                                    imgsz=int(p.get("imgsz", 960)),
                                    verbose=False)
        except Exception as exc:
            failures += len(b_paths)
            if failures <= 8:
                job.note(f"detection failed on frames {b_idx[0]}–{b_idx[-1]}: "
                         f"{type(exc).__name__}: {exc}")
            continue
        for offset, res in enumerate(results):
            if offset >= len(b_idx):
                break
            read += 1
            idxs.append(int(b_idx[offset]))
            times.append(float(b_t[offset]))
            frame: dict = {}
            per_frame.append(frame)
            names = getattr(res, "names", {}) or {}
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            h, w = res.orig_shape if hasattr(res, "orig_shape") else (1, 1)
            for b in boxes:
                cls = names.get(int(b.cls[0]), str(int(b.cls[0])))
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
                share = ((x1 - x0) * (y1 - y0)) / max(float(w) * float(h), 1.0)
                count, best = frame.get(cls, (0, 0.0))
                frame[cls] = (count + 1, max(best, share))
                totals[cls] = totals.get(cls, 0) + 1

    if not read:
        raise SkipPass("no frame could be read")
    if failures:
        job.note(f"{failures} frames failed in detection out of "
                 f"{read + failures}")
    if not totals:
        raise SkipPass("nothing detected above the confidence threshold "
                       f"in any of {read} frames")

    # Presence runs, one series per class over every frame that was read.
    runs = 0
    for cls in sorted(totals, key=lambda c: -totals[c]):
        runs += em.frame_runs(
            "visual", "object",
            ((i, t, cls if cls in f else None)
             for i, t, f in zip(idxs, times, per_frame)))
        em.frame_metric(f"share:{cls}", idxs,
                        [f.get(cls, (0, 0.0))[1] for f in per_frame])
        em.frame_metric(f"count:{cls}", idxs,
                        [f.get(cls, (0, 0.0))[0] for f in per_frame])

    em.frame_metric("objects", idxs,
                    [sum(c for c, _ in f.values()) for f in per_frame])

    # Whole-video totals stay claims — this is what a library-wide query reads.
    for rank, (cls, count) in enumerate(
            sorted(totals.items(), key=lambda kv: -kv[1])[:20]):
        present = sum(1 for f in per_frame if cls in f)
        em.claim("visual", "object", cls, num=count, ordinal=rank,
                 confidence=round(min(present / max(read, 1), 1.0), 4))
    em.notes = {**_coverage(job, read), "classes": len(totals),
                "detections": sum(totals.values()), "runs": runs,
                "tier": "full"}
    return em


# ══════════════════════════════════════════════════════════════════════════
# faces — geometry and continuity, never identity
# ══════════════════════════════════════════════════════════════════════════

def faces(job: Job) -> Emission:
    """Count, scale and continuity — geometry, not identity.

    Face height relative to frame height is the most reliable proxy for shot
    scale on people-centred video. Embeddings are used only to link one face to
    the same face later, they are compared inside this function, and they are
    never written to the database. The archive should be able to answer "does
    the presenter return after the b-roll" without ever being able to answer
    "who is this".

    Per frame now rather than per keyframe, which is what makes the continuity
    claim mean anything: a track that survives a cut is only visible if both
    sides of the cut were looked at.
    """
    np = _np()

    def loader():
        from insightface.app import FaceAnalysis  # noqa: PLC0415
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if job.resources.get("gpu_count")
                     else ["CPUExecutionProvider"])
        app = FaceAnalysis(name=job.params.get("pack", "buffalo_l"),
                           providers=providers)
        app.prepare(ctx_id=0 if job.resources.get("gpu_count") else -1,
                    det_size=(640, 640))
        return app

    try:
        app = job.cache.get(job.component.load_key, loader)
    except ImportError:
        raise SkipPass("insightface is not installed") from None
    except Exception as exc:
        raise SkipPass(f"insightface could not start: "
                       f"{type(exc).__name__}") from None

    em = Emission()
    tracks: list = []
    idxs: list = []
    counts: list = []
    scales: list = []
    scale_runs: list = []
    track_runs: list = []
    read, with_face, failures = 0, 0, 0

    for b_idx, b_t, b_paths in job.frame_batches(_batch(job, 8), tier="full"):
        for i, t, path in zip(b_idx, b_t, b_paths):
            img = _read(path)
            if img is None:
                continue
            read += 1
            h = img.shape[0]
            try:
                detected = app.get(img)
            except Exception as exc:
                failures += 1
                if failures <= 3:
                    job.note(f"face detection failed on frame {i}: "
                             f"{type(exc).__name__}: {exc}")
                detected = []

            idxs.append(int(i))
            counts.append(len(detected))
            if not detected:
                scales.append(0.0)
                scale_runs.append((int(i), float(t), None))
                track_runs.append((int(i), float(t), None))
                continue
            with_face += 1

            # The largest face carries the shot-scale reading: a background
            # extra should not decide whether this is a close-up.
            lead = max(detected, key=lambda f: (f.bbox[3] - f.bbox[1]))
            y0, y1 = float(lead.bbox[1]), float(lead.bbox[3])
            scale = (y1 - y0) / max(h, 1)
            scales.append(scale)
            scale_runs.append((int(i), float(t),
                               "close-up" if scale > 0.45 else
                               "medium" if scale > 0.18 else "wide"))

            vec = getattr(lead, "normed_embedding", None)
            if vec is None:
                track_runs.append((int(i), float(t), None))
                continue
            vec = np.asarray(vec, dtype="float32")
            best, best_sim = -1, 0.0
            for n, ref in enumerate(tracks):
                sim = float(ref @ vec)
                if sim > best_sim:
                    best, best_sim = n, sim
            if best_sim < 0.45:
                tracks.append(vec)
                best = len(tracks) - 1
            track_runs.append((int(i), float(t), f"person {best + 1}"))

    if not read:
        raise SkipPass("no frame could be read")
    if failures:
        job.note(f"{failures} of {read} frames failed in face detection")
    if not with_face:
        raise SkipPass(f"no faces in any of {read} frames")

    em.frame_metric("face_count", idxs, counts)
    em.frame_metric("face_scale", idxs, scales)
    em.frame_runs("visual", "face_scale", scale_runs, confidence=0.85)
    em.frame_runs("visual", "face_track", track_runs, confidence=0.7)
    em.claim("visual", "face_track", f"{len(tracks)} distinct people",
             num=len(tracks))
    em.claim("visual", "face_presence", num=round(with_face / max(read, 1), 4))
    em.notes = {**_coverage(job, read), "tracks": len(tracks),
                "frames_with_face": with_face, "tier": "full"}
    return em


# ══════════════════════════════════════════════════════════════════════════
# depth — shot scale where there is no face to measure
# ══════════════════════════════════════════════════════════════════════════

def depth(job: Job) -> Emission:
    """Relative depth over every frame, reduced to two numbers and a label.

    Depth Anything's output is *relative* — it has no metric units and comparing
    its raw values between two videos means nothing — so everything is
    normalised within the frame before it is recorded. A close-up has a flat,
    near depth histogram in the centre of frame; a wide shot has a long tail.

    The two per-frame numbers land as packed metrics and the derived label as
    run-length claims, which is the same split as everywhere else: numbers to
    plot, words to search.
    """
    np = _np()
    try:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from transformers import (AutoImageProcessor,  # noqa: PLC0415
                                  AutoModelForDepthEstimation)
    except ImportError:
        raise SkipPass("transformers/torch are not installed") from None

    device, dtype = device_and_dtype(job.resources)

    def loader():
        # The registry ships the transformers-layout mirror. The alternates
        # exist because the original checkpoint has no `model_type` in its
        # config.json and `AutoModelForDepthEstimation` cannot dispatch without
        # one — a whole run's worth of depth was lost to that. Trying the `-hf`
        # form of whatever is configured means a future registry edit that drops
        # the suffix degrades to a warning instead of a dead pass.
        want = job.component.model
        candidates = [want]
        if not want.endswith("-hf"):
            candidates.append(want + "-hf")
        last = None
        for repo in candidates:
            try:
                model = AutoModelForDepthEstimation.from_pretrained(
                    repo,
                    torch_dtype=torch_dtype(dtype) if device == "cuda"
                    else torch.float32)
                proc = AutoImageProcessor.from_pretrained(repo)
            except Exception as exc:          # noqa: BLE001
                last = exc
                continue
            if repo != want:
                job.note(f"depth loaded from {repo} — {want} is not in "
                         f"transformers layout")
            model.eval()
            if device == "cuda":
                model = model.to("cuda")
            return {"model": model, "processor": proc}
        raise RuntimeError(
            f"no loadable depth checkpoint among {', '.join(candidates)}: "
            f"{type(last).__name__}: {last}")

    bundle = job.cache.get(job.component.load_key, loader)
    model, proc = bundle["model"], bundle["processor"]

    em = Emission()
    idxs: list = []
    spreads: list = []
    nears: list = []
    labels: list = []
    read, failures = 0, 0

    for b_idx, b_t, b_paths in job.frame_batches(_batch(job, 16)):
        images, keep = [], []
        for i, t, path in zip(b_idx, b_t, b_paths):
            try:
                images.append(Image.open(path).convert("RGB"))
            except OSError:
                continue
            keep.append((int(i), float(t)))
        if not images:
            continue
        try:
            with torch.no_grad():
                inputs = proc(images=images, return_tensors="pt")
                if device == "cuda":
                    inputs = {k: v.to("cuda").to(model.dtype)
                              for k, v in inputs.items()}
                out = model(**inputs).predicted_depth
            maps = out.float().cpu().numpy()
        except Exception as exc:
            failures += len(images)
            if failures <= 8:
                job.note(f"depth failed on frames {keep[0][0]}–{keep[-1][0]}: "
                         f"{type(exc).__name__}: {exc}")
            continue

        for n, (i, t) in enumerate(keep):
            d = maps[n]
            d = d.squeeze()
            lo, hi = float(d.min()), float(d.max())
            d = (d - lo) / max(hi - lo, 1e-6)
            h, w = d.shape
            centre = d[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
            spread = float(centre.std())
            near = float(centre.mean())
            read += 1
            idxs.append(i)
            spreads.append(spread)
            nears.append(near)
            labels.append((i, t,
                           "close-up" if spread < 0.12 and near > 0.55 else
                           "wide" if spread > 0.24 else "medium"))

    if not read:
        raise SkipPass("no readable frames")
    if failures:
        job.note(f"{failures} frames failed in depth estimation")

    em.frame_metric("depth_spread", idxs, spreads)
    em.frame_metric("depth_near", idxs, nears)
    em.frame_runs("style", "shot_scale", labels, confidence=0.6)
    em.claim("style", "depth_spread",
             num=round(float(np.median(spreads)), 4))
    em.notes = {**_coverage(job, read), "failures": failures,
                "tier": "analysis"}
    return em
