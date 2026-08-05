"""
vios.process.runners.vision — what is on the screen.

Seven passes over the keyframes, and they are deliberately redundant. Two OCR
engines read the same pixels because a detector-plus-recogniser drops stylised
type while an end-to-end vision-language model invents plausible words, and the
two failures do not overlap. Objects come from a closed vocabulary with boxes,
faces from a geometric detector, scale from a depth model, and the embedding
that four other things read is computed exactly once.

Nothing here is asked for a timestamp. Every pass is handed a frame that belongs
to a known shot and returns claims against that shot index; the store looks up
t0 and t1 from the shot table. A model's opinion about when something happened
is not evidence, and this is the layer where that rule is enforced by
construction rather than by prompt.
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


# ══════════════════════════════════════════════════════════════════════════
# visual-embed — computed once, read by four things
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


def visual_embed(job: Job) -> Emission:
    """One vector per shot, one pooled vector per video.

    The pooled vector is the mean of the shot vectors, re-normalised. Mean
    pooling of L2-normalised embeddings is the standard construction and it has
    the property that matters here: a video whose shots are all similar lands
    near them, and a video that changes subject three times lands between,
    which is exactly what "how similar are these two reels overall" should mean.
    """
    np = _np()
    frames = job.frames()
    if not frames:
        raise SkipPass("no keyframes")
    try:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        raise SkipPass("torch/Pillow are not installed") from None

    bundle = _siglip(job)
    model, proc, device = bundle["model"], bundle["processor"], bundle["device"]
    batch = int(job.params.get("batch", 8))
    em, pooled, n = Emission(), None, 0

    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch]
        job.heartbeat(f"frame {start}/{len(frames)}")
        images, idxs = [], []
        for f in chunk:
            try:
                images.append(Image.open(f["path"]).convert("RGB"))
                idxs.append(int(f["shot_idx"]))
            except OSError:
                continue
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
        for i, idx in enumerate(idxs):
            em.vector("siglip2", [float(x) for x in vecs[i]], shot_idx=idx)
            pooled = vecs[i] if pooled is None else pooled + vecs[i]
            n += 1

    if pooled is None:
        raise SkipPass("no readable keyframes")
    pooled = pooled / (float(np.linalg.norm(pooled)) + 1e-9)
    em.vector("siglip2", [float(x) for x in pooled])
    em.notes = {"vectors": n + 1, "dim": int(len(pooled))}
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
# aesthetic — classical measures, plus a zero-shot probe
# ══════════════════════════════════════════════════════════════════════════

_GOOD = ("a beautiful, well composed photograph with pleasing lighting",
         "a crisp professional video frame, sharp and well exposed")
_BAD = ("a blurry, badly lit amateur snapshot",
        "an underexposed noisy video frame with motion blur")


def aesthetic(job: Job) -> Emission:
    """Sharpness, exposure, noise and colourfulness — plus a relative quality
    probe against the embedding that already exists.

    The probe is the difference between two cosine similarities: how much more
    the frame looks like the "good" prompts than the "bad" ones. It is a
    *relative* score, useful for ranking frames within this archive, and it is
    not the LAION aesthetic MLP — that head is trained on CLIP ViT-L/14
    features and would be meaningless applied to SigLIP vectors. Saying so here
    matters more than having a number that looks official: a score whose
    provenance is wrong poisons every pattern found downstream.

    The classical measures need no model at all and are the ones to trust when
    the two disagree.
    """
    import cv2  # noqa: PLC0415
    np = _np()

    frames = job.frames()
    if not frames:
        raise SkipPass("no keyframes")

    em = Emission()
    sharps, exposures = [], []
    for n, f in enumerate(frames):
        if n % 20 == 0:
            job.heartbeat(f"frame {n}/{len(frames)}")
        img = _read(f["path"])
        if img is None:
            continue
        idx = int(f["shot_idx"])
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sharp = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        exposure = float(grey.mean() / 255.0)
        # Clipping, not brightness: the share of pixels pinned at either end.
        # A frame can average 0.5 and still have blown highlights, and it is
        # the clipping that makes footage unusable.
        clipped = float(((grey <= 2) | (grey >= 253)).mean())
        # Noise as the median absolute deviation of a high-pass residual —
        # cheap, and robust to the edges that would fool a plain variance.
        residual = grey.astype("float32") - cv2.GaussianBlur(
            grey.astype("float32"), (0, 0), 1.2)
        noise = float(np.median(np.abs(residual)) * 1.4826)
        # Hasler–Süsstrunk colourfulness, the standard closed-form metric.
        b, g, r = (img[:, :, i].astype("float32") for i in range(3))
        rg, yb = r - g, 0.5 * (r + g) - b
        colourful = float(math.hypot(rg.std(), yb.std())
                          + 0.3 * math.hypot(rg.mean(), yb.mean()))

        em.claim("style", "sharpness", num=round(sharp, 2), shot_idx=idx)
        em.claim("style", "exposure", num=round(exposure, 4), shot_idx=idx)
        em.claim("style", "clipping", num=round(clipped, 4), shot_idx=idx)
        em.claim("style", "noise", num=round(noise, 3), shot_idx=idx)
        em.claim("style", "colourfulness", num=round(colourful, 2),
                 shot_idx=idx)
        sharps.append(sharp)
        exposures.append(exposure)

    if not sharps:
        raise SkipPass("no readable keyframes")

    em.claim("style", "sharpness",
             "soft" if np.median(sharps) < 60 else
             "sharp" if np.median(sharps) > 250 else "normal",
             num=round(float(np.median(sharps)), 2))
    em.claim("style", "exposure",
             "underexposed" if np.mean(exposures) < 0.3 else
             "overexposed" if np.mean(exposures) > 0.72 else "well exposed",
             num=round(float(np.mean(exposures)), 4))

    rows = job.store.vectors_for(job.key, "siglip2")
    matrix, names = label_matrix(job, list(_GOOD) + list(_BAD))
    if rows and matrix is not None:
        good_n = len(_GOOD)
        for r in rows:
            v = np.asarray(r["values"], dtype="float32")
            v = v / (np.linalg.norm(v) + 1e-9)
            sims = matrix @ v
            score = float(sims[:good_n].mean() - sims[good_n:].mean())
            em.claim("style", "aesthetic", num=round(score, 5),
                     shot_idx=r["shot_idx"], confidence=0.5)
    em.notes = {"frames": len(sharps), "probe": bool(rows and matrix is not None),
                "labels": len(names)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# ocr — PP-OCRv5
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


def ocr(job: Job) -> Emission:
    """Read the burned-in text, then collapse it in time.

    Reels caption themselves, and those words never appear in the transcript.
    Detection runs per keyframe; identical strings on adjacent shots are then
    merged into one claim carrying how long the text was held, so an eight-second
    caption is one piece of evidence with a duration instead of forty
    near-duplicate rows that any later count would over-weight.
    """
    frames = job.frames()
    if not frames:
        raise SkipPass("no keyframes")

    langs = list(job.component.params.get("languages", ["en"]))
    floor = float(job.component.params.get("min_confidence", 0.6))

    def loader():
        from paddleocr import PaddleOCR  # noqa: PLC0415
        use_gpu = bool(job.resources.get("gpu_count"))
        engines = {}
        for lang in langs:
            try:
                engines[lang] = PaddleOCR(lang=lang, use_angle_cls=True,
                                          show_log=False, use_gpu=use_gpu)
            except TypeError:
                # 3.x dropped show_log/use_gpu and picks the device itself.
                engines[lang] = PaddleOCR(lang=lang)
        return engines

    try:
        engines = job.cache.get(job.component.load_key, loader)
    except ImportError:
        raise SkipPass("paddleocr is not installed") from None
    if not engines:
        raise SkipPass("no OCR language could be initialised")

    shots = job.shots()
    span = {int(s["idx"]): float(s["t1"]) - float(s["t0"]) for s in shots}
    seen: list = []                # [text, first_shot, last_shot, conf, rect]

    for n, f in enumerate(frames):
        job.heartbeat(f"frame {n}/{len(frames)}")
        img = _read(f["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        idx = int(f["shot_idx"])
        found: dict = {}
        for lang, engine in engines.items():
            try:
                result = (engine.predict(img) if hasattr(engine, "predict")
                          else engine.ocr(img, cls=True))
            except Exception as exc:
                job.note(f"{lang} OCR failed on shot {idx}: "
                         f"{type(exc).__name__}")
                continue
            for text, score, poly in _paddle_lines(result):
                text = text.strip()
                if not text or score < floor:
                    continue
                prev = found.get(text)
                if prev is None or score > prev[0]:
                    found[text] = (score, _rect(poly, w, h))

        for text, (score, rect) in found.items():
            for entry in seen:
                # Adjacent-shot merge only. The same words returning after a
                # gap is a repeat, not a hold, and flattening the two would
                # erase a real structural signal.
                if entry[0] == text and idx - entry[2] <= 1:
                    entry[2] = idx
                    entry[3] = max(entry[3], score)
                    break
            else:
                seen.append([text, idx, idx, score, rect])

    if not seen:
        raise SkipPass("no on-screen text above the confidence floor")

    em = Emission()
    for rank, (text, first, last, score, rect) in enumerate(seen):
        held = sum(span.get(i, 0.0) for i in range(first, last + 1))
        em.claim("ocr", "text", text, shot_idx=first, num=round(held, 3),
                 confidence=round(score, 4), ordinal=rank)
        if rect:
            em.claim("ocr", "text_region", {**rect, "text": text},
                     shot_idx=first, confidence=round(score, 4), ordinal=rank)
    em.claim("ocr", "text_density",
             num=round(len(seen) / max(job.duration or 1.0, 0.001), 3))
    em.notes = {"strings": len(seen), "frames": len(frames),
                "languages": langs}
    return em


# ══════════════════════════════════════════════════════════════════════════
# ocr-alt — Florence-2
# ══════════════════════════════════════════════════════════════════════════

def ocr_alt(job: Job) -> Emission:
    """The same frames, a different architecture, and both answers kept.

    Where this and PP-OCR agree, the words can be trusted; where they diverge,
    both are stored and the interface can show the disagreement. Agreement
    between two unrelated systems is a far stronger signal than either one's own
    confidence score, which is calibrated only against its own training set.
    """
    frames = job.frames()
    if not frames:
        raise SkipPass("no keyframes")
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

    em, found = Emission(), 0
    for n, f in enumerate(frames):
        job.heartbeat(f"frame {n}/{len(frames)}")
        try:
            image = Image.open(f["path"]).convert("RGB")
        except OSError:
            continue
        with torch.no_grad():
            inputs = proc(text="<OCR>", images=image, return_tensors="pt")
            if device == "cuda":
                inputs = {k: (v.to("cuda").to(model.dtype)
                              if k == "pixel_values" else v.to("cuda"))
                          for k, v in inputs.items()}
            out = model.generate(input_ids=inputs["input_ids"],
                                 pixel_values=inputs["pixel_values"],
                                 max_new_tokens=256, num_beams=1,
                                 do_sample=False)
        raw = proc.batch_decode(out, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(
            raw, task="<OCR>", image_size=image.size)
        text = str(parsed.get("<OCR>", "")).strip()
        if not text:
            continue
        found += 1
        em.claim("ocr", "text", text, shot_idx=int(f["shot_idx"]),
                 confidence=0.7, ordinal=n)

    if not found:
        raise SkipPass("Florence-2 read no text")
    em.notes = {"frames_with_text": found, "frames": len(frames)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# detect — YOLO11
# ══════════════════════════════════════════════════════════════════════════

def detect(job: Job) -> Emission:
    """Boxes, classes, counts and screen share, from a closed vocabulary.

    Eighty classes detected reliably beat a thousand guessed, because these
    outputs are inputs to later arithmetic. "The subject fills 40% of frame in
    the hook and 12% by the payoff" is a computable sentence only if screen
    share is a number attached to a box.
    """
    frames = job.frames()
    if not frames:
        raise SkipPass("no keyframes")

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

    em, totals, per_shot_max = Emission(), {}, {}
    paths = [f["path"] for f in frames]
    batch = int(job.params.get("batch", 8))

    for start in range(0, len(paths), batch):
        job.heartbeat(f"frame {start}/{len(paths)}")
        chunk = paths[start:start + batch]
        results = model.predict(chunk, conf=float(p.get("conf", 0.35)),
                                imgsz=int(p.get("imgsz", 960)), verbose=False)
        for offset, res in enumerate(results):
            f = frames[start + offset]
            idx = int(f["shot_idx"])
            names = getattr(res, "names", {}) or {}
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            h, w = res.orig_shape if hasattr(res, "orig_shape") else (1, 1)
            counts: dict = {}
            for b in boxes:
                cls = names.get(int(b.cls[0]), str(int(b.cls[0])))
                conf = float(b.conf[0])
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
                share = ((x1 - x0) * (y1 - y0)) / max(float(w) * float(h), 1.0)
                counts[cls] = counts.get(cls, 0) + 1
                totals[cls] = totals.get(cls, 0) + 1
                key = (idx, cls)
                per_shot_max[key] = max(per_shot_max.get(key, 0.0), share)
                em.claim("visual", "object", cls, shot_idx=idx,
                         num=round(share, 4), confidence=round(conf, 4),
                         ordinal=len(em.claims))
            for cls, count in counts.items():
                em.claim("visual", "object_count", cls, shot_idx=idx,
                         num=count, ordinal=len(em.claims))

    if not totals:
        raise SkipPass("nothing detected above the confidence threshold")

    for (idx, cls), share in per_shot_max.items():
        em.claim("visual", "screen_share", cls, shot_idx=idx,
                 num=round(share, 4), ordinal=len(em.claims))
    for rank, (cls, count) in enumerate(
            sorted(totals.items(), key=lambda kv: -kv[1])[:20]):
        em.claim("visual", "object", cls, num=count, ordinal=rank,
                 confidence=min(count / max(len(frames), 1), 1.0))
    em.notes = {"classes": len(totals), "detections": sum(totals.values())}
    return em


# ══════════════════════════════════════════════════════════════════════════
# faces
# ══════════════════════════════════════════════════════════════════════════

def faces(job: Job) -> Emission:
    """Count, scale and continuity — geometry, not identity.

    Face height relative to frame height is the most reliable proxy for shot
    scale on people-centred video. Embeddings are used only to link one face to
    the same face a few shots later, they are compared inside this function,
    and they are never written to the database. The archive should be able to
    answer "does the presenter return after the b-roll" without ever being able
    to answer "who is this".
    """
    np = _np()
    frames = job.frames()
    if not frames:
        raise SkipPass("no keyframes")

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

    em, tracks, any_face = Emission(), [], False
    for n, f in enumerate(frames):
        job.heartbeat(f"frame {n}/{len(frames)}")
        img = _read(f["path"])
        if img is None:
            continue
        h = img.shape[0]
        idx = int(f["shot_idx"])
        try:
            detected = app.get(img)
        except Exception:
            continue
        em.claim("visual", "face_count", num=len(detected), shot_idx=idx)
        if not detected:
            continue
        any_face = True
        for face in detected:
            x0, y0, x1, y1 = (float(v) for v in face.bbox)
            scale = (y1 - y0) / max(h, 1)
            em.claim("visual", "face_scale",
                     "close-up" if scale > 0.45 else
                     "medium" if scale > 0.18 else "wide",
                     shot_idx=idx, num=round(scale, 4),
                     confidence=round(float(getattr(face, "det_score", 0.9)), 4),
                     ordinal=len(em.claims))
            vec = getattr(face, "normed_embedding", None)
            if vec is None:
                continue
            vec = np.asarray(vec, dtype="float32")
            best, best_sim = -1, 0.0
            for t, ref in enumerate(tracks):
                sim = float(ref @ vec)
                if sim > best_sim:
                    best, best_sim = t, sim
            if best_sim < 0.45:
                tracks.append(vec)
                best = len(tracks) - 1
            em.claim("visual", "face_track", f"person {best + 1}",
                     shot_idx=idx, num=round(best_sim, 4),
                     ordinal=len(em.claims))

    if not any_face:
        raise SkipPass("no faces in any keyframe")
    em.claim("visual", "face_track", f"{len(tracks)} distinct people",
             num=len(tracks))
    em.notes = {"tracks": len(tracks), "frames": len(frames)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# depth — shot scale where there is no face to measure
# ══════════════════════════════════════════════════════════════════════════

def depth(job: Job) -> Emission:
    """Relative depth over the keyframe, reduced to shot scale.

    A close-up has a flat, near depth histogram in the centre of frame; a wide
    shot has a long tail. Depth Anything's output is *relative* — it has no
    metric units and comparing its values between two videos means nothing — so
    everything here is normalised within the frame before it becomes a claim.
    """
    np = _np()
    frames = job.frames()
    if not frames:
        raise SkipPass("no keyframes")
    try:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from transformers import (AutoImageProcessor,  # noqa: PLC0415
                                  AutoModelForDepthEstimation)
    except ImportError:
        raise SkipPass("transformers/torch are not installed") from None

    device, dtype = device_and_dtype(job.resources)

    def loader():
        model = AutoModelForDepthEstimation.from_pretrained(
            job.component.model,
            torch_dtype=torch_dtype(dtype) if device == "cuda"
            else torch.float32)
        proc = AutoImageProcessor.from_pretrained(job.component.model)
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
        return {"model": model, "processor": proc}

    bundle = job.cache.get(job.component.load_key, loader)
    model, proc = bundle["model"], bundle["processor"]

    em, done = Emission(), 0
    for n, f in enumerate(frames):
        job.heartbeat(f"frame {n}/{len(frames)}")
        try:
            image = Image.open(f["path"]).convert("RGB")
        except OSError:
            continue
        with torch.no_grad():
            inputs = proc(images=image, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to("cuda").to(model.dtype)
                          for k, v in inputs.items()}
            out = model(**inputs).predicted_depth
        d = out.squeeze().float().cpu().numpy()
        lo, hi = float(d.min()), float(d.max())
        d = (d - lo) / max(hi - lo, 1e-6)
        h, w = d.shape
        centre = d[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        spread = float(centre.std())
        near = float(centre.mean())

        idx = int(f["shot_idx"])
        em.claim("style", "depth_spread", num=round(spread, 4), shot_idx=idx)
        em.claim("style", "shot_scale",
                 "close-up" if spread < 0.12 and near > 0.55 else
                 "wide" if spread > 0.24 else "medium",
                 shot_idx=idx, num=round(near, 4),
                 confidence=0.6, ordinal=n)
        done += 1

    if not done:
        raise SkipPass("no readable keyframes")
    em.notes = {"frames": done}
    return em
