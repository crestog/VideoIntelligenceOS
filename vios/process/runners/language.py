"""
vios.process.runners.language — the only passes that are asked what it means.

Everything before this stage measures. This stage interprets, which is where a
database stops being trustworthy unless the interpretation is constrained. Three
constraints do that work here, and they are structural rather than
prompt-shaped:

**Shot indices, never timestamps.** A model is handed keyframes labelled `shot
4` and answers about shot 4. It is never shown a clock and never asked for one,
so it cannot produce `00:07` — and a returned index outside the range it was
given is dropped rather than stored. Every timestamp in the finished database
comes from the shot table.

**Grounding before interpretation.** `narrate` and `style-read` read the
transcript, the on-screen text, the shot descriptions and the numbers the signal
passes computed. They are asked to explain measurements that already exist,
not to invent observations.

**A control group.** `concepts` is scored against `keyphrase`, which is
statistical and cannot invent a phrase. A concept neither the extractor nor the
source text supports is still stored — it is evidence about the model — but it
is marked unsupported and does not become a graph node.

    "i dont trust one single llm, or model, or script, but if there are
     multiple systems and combination of things working for each elements ...
     then and only then it will be able to satisfy the quality requirements"
"""

from __future__ import annotations

import json
import re

from .base import Emission, Job, SkipPass, device_and_dtype, torch_dtype


# ══════════════════════════════════════════════════════════════════════════
# Parsing what a model returns
# ══════════════════════════════════════════════════════════════════════════

def _parse_json(text: str):
    """The first balanced JSON value in a model's reply, or None.

    Instruction-tuned models wrap JSON in fences, prefix it with "Here is",
    and append a summary paragraph. A regex for `\\[.*\\]` fails the moment a
    description contains a bracket, so this scans for a balanced value while
    tracking string state — the cheapest thing that is actually correct.
    """
    if not text:
        return None
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    start = min((i for i in (body.find("["), body.find("{")) if i >= 0),
                default=-1)
    if start < 0:
        return None
    opening = body[start]
    closing = "]" if opening == "[" else "}"
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(body)):
        ch = body[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("shots", "items", "results", "beats", "concepts", "data"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    return []


def _shot_index(entry, allowed: set):
    """A shot index from a model's reply, if it is one it was actually shown.

    This is the enforcement point for the temporal rule. A model that answers
    about shot 31 when it was handed shots 6 through 11 has hallucinated, and
    the claim is discarded — not clamped, not stored with a warning. Clamping
    would silently attach a real description to the wrong moment, which is the
    one failure this database cannot tolerate.
    """
    for key in ("shot", "shot_idx", "index", "id"):
        raw = entry.get(key) if isinstance(entry, dict) else None
        if raw is None:
            continue
        try:
            idx = int(str(raw).strip().lstrip("#"))
        except (TypeError, ValueError):
            continue
        if idx in allowed:
            return idx
    return None


def _shot_span(entry, allowed: set) -> tuple:
    """A [first, last] shot range from a reply, clipped to what was shown."""
    raw = None
    for key in ("shots", "range", "shot_range"):
        if isinstance(entry, dict) and entry.get(key) is not None:
            raw = entry[key]
            break
    if raw is None:
        one = _shot_index(entry, allowed)
        return (one, one) if one is not None else (None, None)
    if isinstance(raw, (int, str)):
        raw = [raw]
    try:
        nums = [int(str(v).strip().lstrip("#")) for v in list(raw)[:2]]
    except (TypeError, ValueError):
        return None, None
    nums = [n for n in nums if n in allowed]
    if not nums:
        return None, None
    return min(nums), max(nums)


# ══════════════════════════════════════════════════════════════════════════
# The shared vision-language model
# ══════════════════════════════════════════════════════════════════════════

# Every id below was checked to exist on the Hub before it was written here.
# The one that used to be in the registry — `Qwen/Qwen3-VL-8B-Instruct-AWQ` —
# never did, and could not: Qwen published AWQ for the 30B MoE and FP8 for the
# 8B dense model, and no AWQ of the 8B at all. So `describe`, `narrate` and
# `style-read` raised `OSError: ... is not a valid model identifier` on every
# video of every session, and the whole visual, narrative and style side of the
# database was empty while the run reported itself as finished. An id is not the
# kind of thing to guess at, and the second entry of each pair is the official
# full-precision repo — quantised on the way in — so a rename or a withdrawal
# upstream costs a longer download rather than an empty channel.
_MODEL_ALTERNATES = {
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit": ("Qwen/Qwen3-VL-8B-Instruct",),
    "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit": ("Qwen/Qwen3-4B-Instruct-2507",),
}

# Ids the Hub has already said it does not have, this process. A missing model
# is permanent and global: no retry, no later video and no second card will make
# an id nobody published resolve. Without this the engine paid a fresh round
# trip to huggingface.co to be told the same thing once per video per card —
# twenty times, in the session that prompted this — and each one read like a new
# problem in the log. Remembering it costs a set and turns the second failure
# onward into an instant one carrying the same sentence.
_MISSING_MODELS: set = set()


def _model_missing(exc: Exception) -> bool:
    """True when the fault is "there is no such repo", not "it would not load".

    Kept apart from every other load error because the two want opposite
    handling: a model that OOMs may well load once another lane releases its
    card, and a model that does not exist will not.
    """
    if type(exc).__name__ in ("RepositoryNotFoundError", "GatedRepoError"):
        return True
    text = str(exc)
    return ("is not a valid model identifier" in text
            or "Repository Not Found" in text
            or ("404 Client Error" in text and "huggingface.co" in text))


def _four_bit(compute_dtype):
    """NF4, with the compute dtype this card actually has.

    The pre-quantised repos carry their own `quantization_config` in
    `config.json`, and unsloth's declares `bnb_4bit_compute_dtype: bfloat16`.
    Inheriting that on a T4 is the exact bug ARCHITECTURE.md already documents:
    Turing has no bf16 tensor cores, and bitsandbytes selects its 4-bit kernel
    path off the compute dtype, so the load either errors or lands on an
    emulated path several times slower than the fp16 it should have used.
    Passing an explicit config overrides the file's, which is why this is built
    even for a repo that is already quantised.
    """
    try:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415
    except Exception:                                # noqa: BLE001
        return None
    try:
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype)
    except Exception:                                # noqa: BLE001
        return None


def _vlm(job: Job) -> dict:
    """Qwen3-VL 8B in 4-bit NF4, cached under the weights key three passes share.

    Three components — describe, narrate, style-read — declare the same
    `weights`, so the cohort packer counted 6.2 GB once and the cache loads it
    once. Loading it per component would need 18.6 GB on a card that has 15.

    Two nested chains, and they fail over for different reasons. The class chain
    exists because Qwen3-VL's transformers class name has moved with the library
    version and a Kaggle image is whatever it is on the day. The id chain exists
    because the registry's model id was wrong for an entire archive's worth of
    sessions and nothing in the engine was in a position to notice — a load
    fault is a per-video failure, so a configuration error arrived thirty times
    wearing thirty different videos' names.
    """
    comp = job.component
    device, dtype = device_and_dtype(job.resources)
    candidates = (comp.model, *_MODEL_ALTERNATES.get(comp.model, ()))

    def loader():
        import torch  # noqa: PLC0415
        import transformers  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415

        want = torch_dtype(dtype) if device == "cuda" else torch.float32
        kwargs = {"torch_dtype": want, "trust_remote_code": True}
        if device == "cuda":
            kwargs["device_map"] = "auto"
            quant = _four_bit(want)
            if quant is not None:
                kwargs["quantization_config"] = quant

        # Ordered oldest-first only in the sense that `AutoModelForVision2Seq`
        # is deprecated for removal in transformers v5 and warns on every use;
        # `AutoModelForImageTextToText` is the name that supersedes it, and sits
        # ahead of it here so the warning is only reached on an image too old to
        # have the replacement.
        classes = ("Qwen3VLForConditionalGeneration",
                   "AutoModelForImageTextToText",
                   "Qwen2_5_VLForConditionalGeneration",
                   "AutoModelForVision2Seq", "AutoModelForCausalLM")
        tried, missing = [], []
        for model_id in candidates:
            if model_id in _MISSING_MODELS:
                missing.append(f"{model_id} (already known absent)")
                continue
            gone = False
            for name in classes:
                cls = getattr(transformers, name, None)
                if cls is None:
                    continue
                try:
                    model = cls.from_pretrained(model_id, **kwargs)
                    model.eval()
                    proc = AutoProcessor.from_pretrained(
                        model_id, trust_remote_code=True)
                    return {"model": model, "processor": proc, "device": device,
                            "loaded_with": name, "model_id": model_id}
                except Exception as exc:  # noqa: BLE001 — try the next class
                    if _model_missing(exc):
                        # No class will conjure a repo that is not there, so
                        # stop walking them and move to the next id.
                        _MISSING_MODELS.add(model_id)
                        missing.append(f"{model_id} ({type(exc).__name__})")
                        gone = True
                        break
                    tried.append(f"{model_id} via {name}: "
                                 f"{type(exc).__name__}: {str(exc)[:160]}")
            if gone:
                continue

        if missing and not tried:
            # Every candidate came back "no such repo". That is a fault in this
            # file, not on the machine, and saying so is the difference between
            # an operator editing a model id and an operator buying a bigger GPU.
            raise RuntimeError(
                f"{comp.id}: no such model on the Hub — tried "
                + "; ".join(missing)
                + ". This is a configuration fault, not a hardware one: the id "
                  "in vios/process/registry.py needs correcting.")
        detail = "; ".join(tried + missing) or "no transformers class available"
        raise RuntimeError(f"{comp.id}: could not load a VLM — {detail}")

    return job.cache.get(comp.load_key, loader)


def _ask(job: Job, bundle: dict, prompt: str, images: list,
         max_new_tokens: int = 320, temperature: float = 0.2) -> str:
    """One turn with the VLM. Returns raw text."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    model, proc = bundle["model"], bundle["processor"]
    pil = []
    for path in images:
        try:
            pil.append(Image.open(path).convert("RGB"))
        except OSError:
            continue

    content = [{"type": "image"} for _ in pil] + [{"type": "text",
                                                   "text": prompt}]
    messages = [{"role": "user", "content": content}]
    try:
        text = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
    except Exception:
        text = prompt

    inputs = proc(text=[text], images=pil or None, return_tensors="pt",
                  padding=True)
    if bundle["device"] == "cuda":
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0.01, temperature=float(temperature),
            top_p=0.9)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ══════════════════════════════════════════════════════════════════════════
# describe
# ══════════════════════════════════════════════════════════════════════════

_DESCRIBE_PROMPT = """You are labelling frames from a short vertical video for \
a searchable archive. Each image below is the representative frame of one shot, \
in order. They are shots {ids}.

For each shot, report only what is physically visible. No interpretation, no \
guessing at intent, no adjectives about mood.

Return a JSON array, one object per shot, and nothing else:
[{{"shot": <the shot number>, "description": "<one or two literal sentences>", \
"subject": "<the main subject, 1-4 words>", "setting": "<where this is, 1-4 \
words>", "action": "<what is happening, 1-6 words>"}}]

Use only these shot numbers: {ids}. If you cannot see a shot clearly, omit it."""


def describe(job: Job) -> Emission:
    """Literal description, shot by shot, in batches the model can hold.

    Six frames per call is not arbitrary. Beyond that the model starts
    attributing what it saw in one frame to another — the failure is quiet and
    looks like a plausible description — and the batch is what bounds the
    damage. Each call is also self-contained, so one bad reply costs six shots
    rather than the video.
    """
    frames = job.frames()
    if not frames:
        raise SkipPass("no keyframes")

    bundle = _vlm(job)
    p = job.component.params
    size = max(int(p.get("max_shots_per_call", 6)), 1)
    em, ok, calls = Emission(), 0, 0

    for start in range(0, len(frames), size):
        chunk = frames[start:start + size]
        ids = [int(f["shot_idx"]) for f in chunk]
        allowed = set(ids)
        job.heartbeat(f"shots {ids[0]}-{ids[-1]} of {len(frames)}")
        calls += 1
        try:
            reply = _ask(job, bundle,
                         _DESCRIBE_PROMPT.format(ids=", ".join(map(str, ids))),
                         [f["path"] for f in chunk],
                         max_new_tokens=int(p.get("max_new_tokens", 320)),
                         temperature=float(p.get("temperature", 0.2)))
        except Exception as exc:                       # noqa: BLE001
            job.note(f"describe failed on shots {ids}: "
                     f"{type(exc).__name__}: {str(exc)[:200]}")
            continue

        for entry in _as_list(_parse_json(reply)):
            if not isinstance(entry, dict):
                continue
            idx = _shot_index(entry, allowed)
            if idx is None:
                continue
            text = str(entry.get("description", "")).strip()
            if text:
                em.claim("visual", "shot_description", text, shot_idx=idx,
                         confidence=0.75, ordinal=idx)
                ok += 1
            for key, kind in (("subject", "subject"), ("setting", "setting"),
                              ("action", "action")):
                val = str(entry.get(key, "")).strip()
                if val:
                    em.claim("visual", kind, val, shot_idx=idx,
                             confidence=0.7, ordinal=idx)

    if not ok:
        raise SkipPass("the model returned no usable shot descriptions")
    em.notes = {"described": ok, "shots": len(frames), "calls": calls,
                "loaded_with": bundle.get("loaded_with", "")}
    return em


# ══════════════════════════════════════════════════════════════════════════
# narrate
# ══════════════════════════════════════════════════════════════════════════

def _evidence(job: Job) -> dict:
    """Everything the database already knows, compacted for a prompt."""
    bundle = job.text_bundle()
    shots = job.shots()
    descriptions = {i: t for i, t in bundle["descriptions"] if i is not None}
    lines = []
    for s in shots:
        idx = int(s["idx"])
        length = float(s["t1"]) - float(s["t0"])
        lines.append(f"shot {idx} ({length:.1f}s): "
                     f"{descriptions.get(idx, 'no description')}")

    def one(channel, kind):
        rows = job.claims(channel, kind)
        return rows[0] if rows else {}

    return {
        "shots": "\n".join(lines),
        "shot_ids": {int(s["idx"]) for s in shots},
        "transcript": bundle["transcript"][:6000],
        "on_screen": " | ".join(bundle["on_screen"][:40]),
        "caption": bundle["caption"][:1000],
        "asl": one("style", "asl").get("num"),
        "cut_rate": one("style", "cut_rate").get("num"),
        "rhythm": one("style", "rhythm").get("value"),
        "silence": one("audio", "silence_ratio").get("num"),
        "tempo": one("audio", "tempo").get("num"),
        "duration": job.duration,
    }


_NARRATE_PROMPT = """You are analysing one short vertical video for a research \
archive that studies why short-form video holds attention.

Everything below was measured by other systems. Use it. Do not contradict it.

DURATION: {duration:.1f}s
SHOTS (index, length, what is visible):
{shots}

WHAT IS SAID: {transcript}
WHAT IS WRITTEN ON SCREEN: {on_screen}
THE CREATOR'S CAPTION: {caption}
EDITING: average shot {asl}s, {cut_rate} cuts per minute, rhythm {rhythm}

Answer in shot numbers only. Never write a timestamp — you cannot see a clock \
and any time you write would be invented. Only use shot numbers that appear \
above.

Return this JSON and nothing else:
{{"premise": "<what this video is, one sentence>",
 "hook": {{"shots": [first, last], "what": "<what the opening does>", \
"why": "<why it stops a scroll>"}},
 "beats": [{{"shots": [first, last], "what": "<what happens in this stretch>"}}],
 "turn": {{"shots": [first, last], "what": "<the moment the video changes \
direction, if there is one>"}},
 "payoff": {{"shots": [first, last], "what": "<what the viewer is left with>"}},
 "why_it_works": ["<a specific mechanism, tied to something above>"],
 "weakness": "<the weakest part, honestly>"}}"""


def narrate(job: Job) -> Emission:
    """Structure and mechanism, in shot ranges, grounded in the measurements.

    This is the pass whose output most directly serves script writing and
    pattern recognition, and it is deliberately last: it reads what every other
    pass produced. Handing it the numbers rather than the video is what stops
    it writing film-school vocabulary at random — each claim it makes has
    arithmetic sitting under it, and the arithmetic is in the same database.
    """
    ev = _evidence(job)
    if not ev["shots"]:
        raise SkipPass("no shots")
    if not (ev["transcript"] or ev["on_screen"] or ev["caption"]):
        raise SkipPass("nothing was said, written or captioned — there is no "
                       "narrative to read")

    bundle = _vlm(job)
    p = job.component.params
    frames = job.frames()
    # A handful of frames spread across the video, not all of them: the
    # narrative question is answered from the assembled evidence, and the
    # images are there to keep the reading honest about what it is looking at.
    picks = [frames[i] for i in
             sorted({0, len(frames) // 3, 2 * len(frames) // 3,
                     len(frames) - 1})] if frames else []

    job.heartbeat("narrating")
    prompt = _NARRATE_PROMPT.format(
        duration=ev["duration"] or 0.0, shots=ev["shots"],
        transcript=ev["transcript"] or "(nothing said)",
        on_screen=ev["on_screen"] or "(no on-screen text)",
        caption=ev["caption"] or "(no caption)",
        asl=round(ev["asl"] or 0, 2), cut_rate=round(ev["cut_rate"] or 0, 1),
        rhythm=ev["rhythm"] or "unknown")
    try:
        reply = _ask(job, bundle, prompt, [f["path"] for f in picks],
                     max_new_tokens=int(p.get("max_new_tokens", 700)),
                     temperature=float(p.get("temperature", 0.3)))
    except Exception as exc:                           # noqa: BLE001
        raise SkipPass(f"the model failed: {type(exc).__name__}: {exc}") from None

    data = _parse_json(reply)
    if not isinstance(data, dict):
        raise SkipPass("the model did not return usable JSON")

    allowed = ev["shot_ids"]
    em, wrote = Emission(), 0
    if data.get("premise"):
        em.claim("narrative", "premise", str(data["premise"]).strip(),
                 confidence=0.7)
        wrote += 1

    for key, kind in (("hook", "hook"), ("turn", "turn"),
                      ("payoff", "payoff")):
        entry = data.get(key)
        if not isinstance(entry, dict):
            continue
        first, _last = _shot_span(entry, allowed)
        what = str(entry.get("what", "")).strip()
        if not what:
            continue
        em.claim("narrative", kind, what, shot_idx=first, confidence=0.65)
        wrote += 1
        if entry.get("why"):
            em.claim("narrative", "why_it_works", str(entry["why"]).strip(),
                     shot_idx=first, confidence=0.6)

    for n, entry in enumerate(_as_list(data.get("beats"))):
        if not isinstance(entry, dict):
            continue
        first, _last = _shot_span(entry, allowed)
        what = str(entry.get("what", "")).strip()
        if what:
            em.claim("narrative", "beat", what, shot_idx=first,
                     confidence=0.6, ordinal=n)
            wrote += 1

    for n, why in enumerate(_as_list(data.get("why_it_works"))):
        if isinstance(why, str) and why.strip():
            em.claim("narrative", "why_it_works", why.strip(),
                     confidence=0.55, ordinal=100 + n)
            wrote += 1
    if data.get("weakness"):
        em.claim("narrative", "weakness", str(data["weakness"]).strip(),
                 confidence=0.5)

    if not wrote:
        raise SkipPass("the reply contained no usable claims")
    em.notes = {"claims": wrote}
    return em


# ══════════════════════════════════════════════════════════════════════════
# style-read
# ══════════════════════════════════════════════════════════════════════════

_STYLE_PROMPT = """You are describing the craft of one short vertical video for \
a research archive. Other systems measured the following. Your job is to name \
the technique these numbers add up to — not to re-measure them.

MEASURED:
- average shot length {asl}s, {cut_rate} cuts per minute, rhythm: {rhythm}
- camera movement across shots: {camera}
- shot scale across shots: {scale}
- colour: brightness {brightness}, saturation {saturation}, {temperature}
- dominant palette: {palette}

The images are representative frames.

Return this JSON and nothing else:
{{"technique": ["<a named technique this video uses, tied to a number above>"],
 "lighting": "<how it is lit>",
 "grade": "<how it is coloured>",
 "framing": "<how it is framed>",
 "edit_style": "<how it is cut>",
 "reference": "<what kind of work this resembles>"}}

Every entry in "technique" must be traceable to one of the measurements above."""


def style_read(job: Job) -> Emission:
    """Name the craft, with the arithmetic already on the table.

    The measurements go into the prompt and the model is told not to
    re-measure. Ungrounded, a vision model will report "fast, punchy cutting"
    for a reel with a four-second average shot length, because that sentence is
    what reels are usually described with. Grounded, it has to reconcile its
    vocabulary with a number it can see.
    """
    shots = job.shots()
    if not shots:
        raise SkipPass("no shots")

    def one(channel, kind, field="value"):
        rows = [r for r in job.claims(channel, kind) if r.get("shot_idx") is None]
        return rows[0].get(field) if rows else None

    def spread(channel, kind, limit=8):
        vals = [str(r.get("value")) for r in job.claims(channel, kind)
                if r.get("shot_idx") is not None and r.get("value")]
        if not vals:
            return "not measured"
        counts: dict = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        return ", ".join(f"{k} ({c})" for k, c in
                         sorted(counts.items(), key=lambda kv: -kv[1])[:limit])

    palette_rows = [r for r in job.claims("style", "palette") if r.get("value")]
    swatches: list = []
    for r in palette_rows[:6]:
        try:
            swatches += [s["hex"] for s in json.loads(r["value"])[:2]]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    bundle = _vlm(job)
    frames = job.frames()
    picks = [frames[i] for i in
             sorted({0, len(frames) // 2, len(frames) - 1})] if frames else []

    job.heartbeat("reading craft")
    prompt = _STYLE_PROMPT.format(
        asl=round(one("style", "asl", "num") or 0, 2),
        cut_rate=round(one("style", "cut_rate", "num") or 0, 1),
        rhythm=one("style", "rhythm") or "unknown",
        camera=spread("style", "camera_move"),
        scale=spread("style", "shot_scale") or spread("visual", "face_scale"),
        brightness=one("style", "brightness") or "unknown",
        saturation=one("style", "saturation") or "unknown",
        temperature=one("style", "temperature") or "unknown",
        palette=", ".join(dict.fromkeys(swatches)) or "not measured")
    try:
        reply = _ask(job, bundle, prompt, [f["path"] for f in picks],
                     max_new_tokens=420, temperature=0.25)
    except Exception as exc:                           # noqa: BLE001
        raise SkipPass(f"the model failed: "
                       f"{type(exc).__name__}: {str(exc)[:200]}") from None

    data = _parse_json(reply)
    if not isinstance(data, dict):
        raise SkipPass("the model did not return usable JSON")

    em, wrote = Emission(), 0
    for n, t in enumerate(_as_list(data.get("technique"))):
        if isinstance(t, dict):
            t = t.get("name") or t.get("technique") or ""
        if isinstance(t, str) and t.strip():
            em.claim("style", "technique", t.strip(), confidence=0.6,
                     ordinal=n)
            wrote += 1
    for key in ("lighting", "grade", "framing", "edit_style", "reference"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            em.claim("style", key, val.strip(), confidence=0.55)
            wrote += 1

    if not wrote:
        raise SkipPass("the reply contained no usable claims")
    em.notes = {"claims": wrote}
    return em


# ══════════════════════════════════════════════════════════════════════════
# concepts
# ══════════════════════════════════════════════════════════════════════════

_CONCEPT_PROMPT = """Extract the concepts from this short video's text for a \
knowledge graph.

TRANSCRIPT: {transcript}
ON SCREEN: {on_screen}
CAPTION: {caption}

Rules:
- Only concepts the text actually supports. Do not add what you know about the \
topic from elsewhere.
- "evidence" must be an exact quote copied from the text above. It will be \
checked against the source, and anything that is not found there will be marked \
unsupported.
- Normalise names: lowercase, singular, no hashtags.

Return this JSON and nothing else:
[{{"name": "<concept>", "type": "entity|topic|technique", "evidence": "<exact \
quote from above>"}}]"""


def concepts(job: Job) -> Emission:
    """Named concepts, each carrying the span of text that evidences it.

    The evidence quote is verified against the source before the claim is
    written. Not to filter the model — an unsupported concept is still stored,
    because what a model asserts without evidence is itself worth knowing — but
    to mark it, so the graph is built only from concepts that two independent
    systems can both point at. Auditable edges are the difference between a
    knowledge graph and a pile of associations.
    """
    bundle = job.text_bundle()
    source = " ".join(filter(None, [
        bundle["transcript"], bundle["caption"],
        " ".join(bundle["on_screen"])])).strip()
    if len(source) < 40:
        raise SkipPass("no transcript, caption or on-screen text for this "
                       f"video — {len(source)} characters between all three")

    comp = job.component
    device, dtype = device_and_dtype(job.resources)

    def loader():
        import torch  # noqa: PLC0415
        from transformers import (AutoModelForCausalLM,  # noqa: PLC0415
                                  AutoTokenizer)
        kwargs = {"torch_dtype": torch_dtype(dtype) if device == "cuda"
                  else torch.float32, "trust_remote_code": True}
        if device == "cuda":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(comp.model, **kwargs)
        model.eval()
        tok = AutoTokenizer.from_pretrained(comp.model, trust_remote_code=True)
        return {"model": model, "tokenizer": tok}

    try:
        pack = job.cache.get(comp.load_key, loader)
    except Exception as exc:                           # noqa: BLE001
        raise SkipPass(f"{comp.model} could not be loaded: "
                       f"{type(exc).__name__}: {str(exc)[:200]}") from None

    import torch  # noqa: PLC0415
    model, tok = pack["model"], pack["tokenizer"]
    prompt = _CONCEPT_PROMPT.format(
        transcript=bundle["transcript"][:5000] or "(nothing said)",
        on_screen=" | ".join(bundle["on_screen"][:40]) or "(none)",
        caption=bundle["caption"][:800] or "(none)")
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    job.heartbeat("extracting concepts")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=600, do_sample=False)
    reply = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)[0]

    data = _as_list(_parse_json(reply))
    if not data:
        raise SkipPass("the model did not return usable JSON")

    haystack = re.sub(r"\s+", " ", source.lower())
    control = {str(c.get("value", "")).lower()
               for c in job.claims("concept", "keyphrase")}
    em, supported = Emission(), 0

    for n, entry in enumerate(data[:40]):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip().lower().lstrip("#")
        if not name or len(name) > 80:
            continue
        kind = str(entry.get("type", "topic")).strip().lower()
        if kind not in ("entity", "topic", "technique"):
            kind = "topic"
        quote = re.sub(r"\s+", " ", str(entry.get("evidence", "")).strip().lower())
        in_text = bool(quote) and quote in haystack
        in_control = any(name in phrase or phrase in name
                         for phrase in control if phrase)
        # Two independent supports, weighted: a quote that is genuinely in the
        # source, and agreement with the statistical extractor. Both is a
        # graph node; one is a claim worth keeping; neither is recorded as the
        # model's assertion and nothing more.
        confidence = 0.35 + (0.35 if in_text else 0.0) + (0.3 if in_control
                                                          else 0.0)
        em.claim("concept", kind, name, num=round(confidence, 3),
                 confidence=round(confidence, 3), ordinal=n)
        if in_text:
            em.claim("concept", "claim_span", {"concept": name,
                                               "quote": str(entry.get("evidence"))},
                     confidence=round(confidence, 3), ordinal=n)
            supported += 1
        elif quote:
            em.claim("concept", "unsupported", name, num=round(confidence, 3),
                     confidence=round(confidence, 3), ordinal=n)

    if not em.claims:
        raise SkipPass("no usable concepts")
    em.notes = {"concepts": len(data), "quote_supported": supported,
                "control_phrases": len(control)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# text-embed
# ══════════════════════════════════════════════════════════════════════════

def text_embed(job: Job) -> Emission:
    """BGE-M3 over each passage separately, not over the video as a whole.

    Multilingual by construction, which is the requirement: the archive holds
    English, Hindi and Hinglish and the query may be in any of them, including
    romanised Hindi that no monolingual model handles. Embedding per passage
    rather than per video is what makes a semantic hit land on a moment instead
    of on a twenty-thousand-character blur — the vector carries the shot index
    of the passage it came from.
    """
    import numpy as np  # noqa: PLC0415
    bundle = job.text_bundle()

    passages: list = []
    for row in job.claims("speech", "segment"):
        if row.get("value"):
            passages.append((row.get("shot_idx"), str(row["value"])))
    if bundle["caption"]:
        passages.append((None, bundle["caption"][:2000]))
    for row in job.claims("ocr", "text"):
        if row.get("value"):
            passages.append((row.get("shot_idx"), str(row["value"])))
    for row in job.claims("narrative", "beat") + job.claims("narrative", "hook"):
        if row.get("value"):
            passages.append((row.get("shot_idx"), str(row["value"])))
    passages = [(i, t.strip()) for i, t in passages if t and t.strip()]
    if not passages:
        raise SkipPass("no transcript, caption, on-screen text or narrative "
                       "beat for this video — nothing to embed")

    comp = job.component
    device, dtype = device_and_dtype(job.resources)

    def loader():
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        model = AutoModel.from_pretrained(
            comp.model, torch_dtype=torch_dtype(dtype) if device == "cuda"
            else torch.float32)
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
        return {"model": model,
                "tokenizer": AutoTokenizer.from_pretrained(comp.model)}

    try:
        pack = job.cache.get(comp.load_key, loader)
    except Exception as exc:                           # noqa: BLE001
        raise SkipPass(f"{comp.model} could not be loaded: "
                       f"{type(exc).__name__}: {str(exc)[:200]}") from None

    import torch  # noqa: PLC0415
    model, tok = pack["model"], pack["tokenizer"]
    em, pooled, n = Emission(), None, 0
    batch = int(job.params.get("batch", 16))

    for start in range(0, len(passages), batch):
        chunk = passages[start:start + batch]
        job.heartbeat(f"passage {start}/{len(passages)}")
        inputs = tok([t for _i, t in chunk], padding=True, truncation=True,
                     max_length=512, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs).last_hidden_state[:, 0]
            # CLS pooling, then L2. BGE-M3's dense head is the CLS token — mean
            # pooling produces vectors that still look reasonable and score
            # measurably worse, which is the worst kind of wrong.
            out = (out / out.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        for i, (shot_idx, _t) in enumerate(chunk):
            em.vector("bge-m3", [float(x) for x in out[i]], shot_idx=shot_idx)
            pooled = out[i] if pooled is None else pooled + out[i]
            n += 1

    pooled = pooled / (float(np.linalg.norm(pooled)) + 1e-9)
    em.vector("bge-m3-video", [float(x) for x in pooled])
    em.notes = {"passages": n, "dim": int(len(pooled))}
    return em


# ══════════════════════════════════════════════════════════════════════════
# narrate-deep
# ══════════════════════════════════════════════════════════════════════════

def narrate_deep(job: Job) -> Emission:
    """InternVL3 38B across both cards, for the reels that earn it.

    Sharded with `device_map="auto"` over two T4s, which without NVLink means
    PCIe traffic at every layer boundary and roughly a ninety-second pass. Not
    worth it for five thousand reels; clearly worth it for the few hundred that
    outperformed, where the difference between a good reading and an excellent
    one compounds into everything written afterwards.

    It writes under its own observer id, so its claims sit *beside* the 8B
    model's rather than replacing them, and the two can be compared.
    """
    ev = _evidence(job)
    if not ev["shots"]:
        raise SkipPass("no shots")

    comp = job.component
    device, dtype = device_and_dtype(job.resources)
    if device != "cuda" or int(job.resources.get("gpu_count", 0)) < comp.cards:
        raise SkipPass(f"needs {comp.cards} GPUs; this machine has "
                       f"{job.resources.get('gpu_count', 0)}")

    def loader():
        import torch  # noqa: PLC0415
        from transformers import (AutoModel, AutoTokenizer)  # noqa: PLC0415
        model = AutoModel.from_pretrained(
            comp.model, torch_dtype=torch_dtype(dtype), device_map="auto",
            trust_remote_code=True, low_cpu_mem_usage=True).eval()
        tok = AutoTokenizer.from_pretrained(comp.model, trust_remote_code=True,
                                            use_fast=False)
        return {"model": model, "tokenizer": tok}

    try:
        pack = job.cache.get(comp.load_key, loader)
    except Exception as exc:                           # noqa: BLE001
        raise SkipPass(f"{comp.model} could not be loaded: "
                       f"{type(exc).__name__}: {str(exc)[:200]}") from None

    frames = job.frames()
    picks = [frames[i] for i in
             sorted({0, len(frames) // 3, 2 * len(frames) // 3,
                     len(frames) - 1})] if frames else []
    prompt = _NARRATE_PROMPT.format(
        duration=ev["duration"] or 0.0, shots=ev["shots"],
        transcript=ev["transcript"] or "(nothing said)",
        on_screen=ev["on_screen"] or "(no on-screen text)",
        caption=ev["caption"] or "(no caption)",
        asl=round(ev["asl"] or 0, 2), cut_rate=round(ev["cut_rate"] or 0, 1),
        rhythm=ev["rhythm"] or "unknown")

    job.heartbeat("deep narration")
    try:
        reply = _internvl_chat(pack, picks, prompt)
    except Exception as exc:                           # noqa: BLE001
        raise SkipPass(f"the model failed: {type(exc).__name__}: {exc}") from None

    data = _parse_json(reply)
    if not isinstance(data, dict):
        raise SkipPass("the model did not return usable JSON")

    allowed = ev["shot_ids"]
    em, wrote = Emission(), 0
    if data.get("premise"):
        em.claim("narrative", "premise", str(data["premise"]).strip(),
                 confidence=0.8)
        wrote += 1
    for key, kind in (("hook", "hook"), ("turn", "turn"), ("payoff", "payoff")):
        entry = data.get(key)
        if not isinstance(entry, dict):
            continue
        first, _last = _shot_span(entry, allowed)
        what = str(entry.get("what", "")).strip()
        if what:
            em.claim("narrative", kind, what, shot_idx=first, confidence=0.75)
            wrote += 1
    for n, entry in enumerate(_as_list(data.get("beats"))):
        if isinstance(entry, dict) and str(entry.get("what", "")).strip():
            first, _last = _shot_span(entry, allowed)
            em.claim("narrative", "beat", str(entry["what"]).strip(),
                     shot_idx=first, confidence=0.7, ordinal=n)
            wrote += 1
    for n, why in enumerate(_as_list(data.get("why_it_works"))):
        if isinstance(why, str) and why.strip():
            em.claim("narrative", "why_it_works", why.strip(),
                     confidence=0.7, ordinal=100 + n)
            wrote += 1
    if data.get("weakness"):
        em.claim("narrative", "critique", str(data["weakness"]).strip(),
                 confidence=0.65)
        wrote += 1

    if not wrote:
        raise SkipPass("the reply contained no usable claims")
    em.notes = {"claims": wrote, "images": len(picks)}
    return em


def _internvl_chat(pack: dict, frames: list, prompt: str) -> str:
    """InternVL's own chat API, which is not the transformers generate loop.

    The remote code exposes `model.chat(tokenizer, pixel_values, question,
    generation_config, num_patches_list=...)` and expects images preprocessed
    to 448×448 tiles with ImageNet normalisation. Written out rather than
    imported because the helper lives in the model card, not in a package.
    """
    import torch  # noqa: PLC0415
    import torchvision.transforms as T  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    model, tok = pack["model"], pack["tokenizer"]
    tf = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    tiles, counts = [], []
    for f in frames:
        try:
            tiles.append(tf(Image.open(f["path"]).convert("RGB")))
            counts.append(1)
        except OSError:
            continue
    if not tiles:
        raise RuntimeError("no readable frames")

    pixel_values = torch.stack(tiles).to(torch.float16).to(model.device)
    question = "".join(f"Image-{i + 1}: <image>\n"
                       for i in range(len(tiles))) + prompt
    return model.chat(tok, pixel_values, question,
                      {"max_new_tokens": 800, "do_sample": False},
                      num_patches_list=counts)
