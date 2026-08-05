"""
vios.process.runners.audio — speech, speakers, and the sound that is not speech.

Four passes over one waveform, and they disagree on purpose. Two transcribers
read the same audio with different decoders so that agreement between them can
be measured instead of assumed; a diarizer says how many voices there are
without saying whose; and a tagger reads everything the transcript throws away,
which on short-form video is about half the craft.

The archive is English, Hindi, Hinglish and a long southern tail, so language is
detected per video and never assumed. A pass that assumes English does not fail
on Telugu — it succeeds, fluently, at producing nonsense, and nothing downstream
can tell. Low-confidence detections are flagged as claims rather than silently
transcribed, which is the user's rule:

    "if its not able to process then it can report to me, we can improve later"
"""

from __future__ import annotations

import difflib
import math
import os
import re

from .. import media
from .base import Emission, Job, SkipPass, device_and_dtype

# Below this the language head is guessing. The transcript is still written —
# it is often right — but it carries a flag, and the engine tab can list every
# flagged video so the failure is visible rather than buried.
LANGUAGE_FLOOR = 0.55


def _hf_token(job: Job) -> str:
    """A Hugging Face token from the environment, never from a file.

    Same rule as the Telegram credentials: secrets are runtime values, and the
    repository is a place where they must not be able to appear.
    """
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return str(job.params.get("hf_token", "") or "")


def _norm(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()


# ══════════════════════════════════════════════════════════════════════════
# transcribe / transcribe-alt
# ══════════════════════════════════════════════════════════════════════════

def _whisper(job: Job):
    """A CTranslate2 Whisper, cached under the component's weights key.

    `int8_float16` only exists on CUDA. On CPU it is silently accepted by some
    builds and rejected by others, so the compute type is chosen from the
    device rather than copied from the registry unconditionally.
    """
    comp = job.component
    device, _dtype = device_and_dtype(job.resources)
    compute = comp.quant or "int8_float16"
    if device == "cpu":
        compute = "int8"

    def loader():
        from faster_whisper import WhisperModel  # noqa: PLC0415
        return WhisperModel(comp.model, device=device, compute_type=compute,
                            download_root=job.params.get("cache_dir") or None)

    return job.cache.get(comp.load_key, loader)


def _run_whisper(job: Job, *, alt: bool) -> Emission:
    wav_path = job.artifact("audio.wav")
    try:
        model = _whisper(job)
    except ImportError:
        raise SkipPass("faster-whisper is not installed") from None

    p = job.component.params
    job.heartbeat("decoding")
    segments, info = model.transcribe(
        wav_path,
        beam_size=int(p.get("beam_size", 5)),
        vad_filter=bool(p.get("vad_filter", True)),
        word_timestamps=bool(p.get("word_timestamps", False)),
        condition_on_previous_text=bool(p.get("condition_on_previous_text", False)),
        language=job.params.get("language") or None)

    em = Emission()
    lang = getattr(info, "language", "") or ""
    prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    em.claim("speech", "language", lang, num=round(prob, 4),
             confidence=round(prob, 4))
    if prob < LANGUAGE_FLOOR:
        em.claim("speech", "language_uncertain",
                 f"language detected as {lang!r} with {prob:.2f} confidence — "
                 f"this transcript may be wrong", num=round(prob, 4),
                 confidence=round(prob, 4))

    texts, words, n = [], [], 0
    for seg in segments:                    # a generator: decoding happens here
        text = (seg.text or "").strip()
        if not text:
            continue
        n += 1
        if n % 10 == 0:
            job.heartbeat(f"segment {n}")
        start = float(seg.start or 0.0)
        # `avg_logprob` is a mean log probability per token, so exponentiating
        # gives a per-token likelihood in 0..1 — a usable confidence, and the
        # only one Whisper offers that is not a hand-tuned threshold.
        conf = min(max(math.exp(float(getattr(seg, "avg_logprob", -0.5) or -0.5)),
                       0.0), 1.0)
        em.claim("speech", "segment", text, shot_idx=job.shot_at(start),
                 num=round(start, 3), confidence=round(conf, 4), ordinal=n)
        texts.append(text)
        for w in (getattr(seg, "words", None) or []):
            words.append({"w": (w.word or "").strip(),
                          "s": round(float(w.start or 0.0), 3),
                          "e": round(float(w.end or 0.0), 3)})

    if not texts:
        raise SkipPass("no speech detected")

    full = " ".join(texts)
    em.claim("speech", "transcript", full, num=len(full.split()))
    if words:
        # One claim holding every word time, rather than one claim per word.
        # Word-level timing is what makes moment search land on the syllable
        # instead of the shot, and 900 rows per reel to say so would quadruple
        # the claim table for data that is only ever read as a whole.
        em.claim("speech", "words", words, num=len(words))
    dur = job.duration or 1.0
    em.claim("speech", "speech_rate",
             f"{len(full.split()) / max(dur, 0.001) * 60:.0f} words per minute",
             num=round(len(full.split()) / max(dur, 0.001) * 60, 2))

    # The second reader earns its five seconds here: not by transcribing, but
    # by disagreeing measurably with the first.
    if alt:
        # Observer ids are `component@hash`, so the component prefix is how one
        # pass finds another pass's rows without caring which revision wrote
        # them.
        primary = " ".join(
            c["value"] for c in job.claims("speech", "segment")
            if c.get("value")
            and str(c.get("observer_id", "")).split("@")[0] == "transcribe")
        if primary:
            ratio = difflib.SequenceMatcher(
                None, _norm(primary), _norm(full)).ratio()
            em.claim("speech", "agreement",
                     f"{ratio * 100:.0f}% agreement with the primary transcript",
                     num=round(ratio, 4), confidence=round(ratio, 4))
            if ratio < 0.75:
                em.claim("speech", "contested",
                         "the two decoders disagree about this audio; both "
                         "readings are stored", num=round(ratio, 4))

    em.notes = {"language": lang, "language_probability": round(prob, 3),
                "segments": n, "words": len(words), "chars": len(full)}
    return em


def transcribe(job: Job) -> Emission:
    """Whisper large-v3 — the primary reading of speech."""
    return _run_whisper(job, alt=False)


def transcribe_alt(job: Job) -> Emission:
    """large-v3-turbo over the same waveform, as an independent observer."""
    return _run_whisper(job, alt=True)


# ══════════════════════════════════════════════════════════════════════════
# diarize
# ══════════════════════════════════════════════════════════════════════════

def diarize(job: Job) -> Emission:
    """How many voices, and where each one speaks. No identity, ever.

    Speaker labels are local to a video — SPEAKER_00 in one reel has nothing to
    do with SPEAKER_00 in another, and no embedding leaves this function. The
    useful output is structural: a monologue, an interview and a voiceover over
    b-roll are three different formats, and every other reading of the reel
    should be interpreted differently depending on which it is.
    """
    wav_path = job.artifact("audio.wav")
    token = _hf_token(job)
    if not token:
        raise SkipPass("no Hugging Face token in the environment — pyannote's "
                       "weights are gated (set HF_TOKEN)")

    def loader():
        import torch  # noqa: PLC0415
        from pyannote.audio import Pipeline  # noqa: PLC0415
        pipe = Pipeline.from_pretrained(job.component.model,
                                        use_auth_token=token)
        if job.resources.get("gpu_count"):
            pipe.to(torch.device("cuda"))
        return pipe

    try:
        pipe = job.cache.get(job.component.load_key, loader)
    except ImportError:
        raise SkipPass("pyannote.audio is not installed") from None

    job.heartbeat("diarizing")
    annotation = pipe(wav_path)

    em = Emission()
    speakers, spans, total = {}, [], 0.0
    for n, (turn, _track, label) in enumerate(
            annotation.itertracks(yield_label=True)):
        start, end = float(turn.start), float(turn.end)
        speakers[label] = speakers.get(label, 0.0) + (end - start)
        total += end - start
        spans.append((start, end))
        em.claim("speech", "speaker_turn", str(label),
                 shot_idx=job.shot_at(start), num=round(start, 3),
                 ordinal=n)

    if not speakers:
        raise SkipPass("no speech regions found")

    em.claim("speech", "speaker_count",
             "monologue" if len(speakers) == 1 else
             "two voices" if len(speakers) == 2 else
             f"{len(speakers)} voices", num=len(speakers))
    for label, seconds in sorted(speakers.items(), key=lambda kv: -kv[1]):
        em.claim("speech", "speaker_share", label,
                 num=round(seconds / max(total, 0.001), 4))

    # Crosstalk: the share of speaking time where two turns overlap. High
    # overlap is a conversation; near zero with two speakers is an edit.
    spans.sort()
    overlap = 0.0
    for i in range(1, len(spans)):
        overlap += max(0.0, min(spans[i - 1][1], spans[i][1]) - spans[i][0])
    em.claim("speech", "overlap", num=round(overlap / max(total, 0.001), 4))

    em.notes = {"speakers": len(speakers), "turns": len(spans),
                "overlap": round(overlap, 2)}
    return em


# ══════════════════════════════════════════════════════════════════════════
# audio-tag
# ══════════════════════════════════════════════════════════════════════════

# What a transcript cannot hear. Written as sentences because CLAP was trained
# on caption-like text, and "a whoosh transition sound effect" scores far more
# reliably than the bare word "whoosh".
SOUND_LABELS = (
    "music playing", "a person speaking", "singing", "laughter", "applause",
    "a whoosh transition sound effect", "a notification or alert sound",
    "a bass drop", "silence", "background room tone", "traffic noise",
    "wind or outdoor ambience", "typing on a keyboard", "footsteps",
    "kitchen sounds, chopping or sizzling", "a crowd or party", "a car engine",
    "water running", "a phone ringing", "an animal sound",
    "a cinematic riser or build-up", "a record scratch", "a click or tap",
)


def audio_tag(job: Job) -> Emission:
    """CLAP over windows of the track: sound events, music presence, a vector.

    Zero-shot again, and for the same reason as the visual tagger — the answer
    is a cosine between two embeddings, so it can be low but it cannot be
    invented. The pooled embedding is stored as a vector, which is what makes
    "find reels that sound like this one" a query rather than a listening
    session.
    """
    import numpy as np  # noqa: PLC0415

    job.artifact("audio.wav")                # presence check: no audio, no pass
    try:
        import torch  # noqa: PLC0415
        from transformers import ClapModel, ClapProcessor  # noqa: PLC0415
    except ImportError:
        raise SkipPass("transformers/torch are not installed") from None

    # CLAP is trained at 48 kHz and the pipeline's working wav is 16 kHz mono.
    # Upsampling 16→48 invents no detail but does put the spectrum where the
    # model expects it; re-extracting from the source is better and costs one
    # ffmpeg call.
    wav48 = job.path("audio48.wav")
    if not os.path.exists(wav48):
        try:
            media.wav(job.source, wav48, rate=48000)
        except media.MediaError:
            wav48 = job.artifact("audio.wav")

    device, _ = device_and_dtype(job.resources)

    def loader():
        model = ClapModel.from_pretrained(job.component.model)
        proc = ClapProcessor.from_pretrained(job.component.model)
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
        return {"model": model, "processor": proc}

    bundle = job.cache.get(job.component.load_key, loader)
    model, proc = bundle["model"], bundle["processor"]

    try:
        import soundfile as sf  # noqa: PLC0415
        data, rate = sf.read(wav48, dtype="float32", always_2d=False)
    except Exception:
        import librosa  # noqa: PLC0415
        data, rate = librosa.load(wav48, sr=48000, mono=True)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if len(data) < rate:
        raise SkipPass("audio shorter than one second")

    with torch.no_grad():
        text_in = proc(text=list(SOUND_LABELS), return_tensors="pt",
                       padding=True)
        if device == "cuda":
            text_in = {k: v.to("cuda") for k, v in text_in.items()}
        tvec = model.get_text_features(**text_in)
        tvec = (tvec / tvec.norm(dim=-1, keepdim=True)).cpu().numpy()

    window = int(rate * float(job.params.get("window_seconds", 10.0)))
    starts = list(range(0, max(len(data) - window // 2, 1), window)) or [0]
    em = Emission()
    pooled, per_window = None, 0

    for n, start in enumerate(starts):
        chunk = data[start:start + window]
        if len(chunk) < rate // 2:
            continue
        job.heartbeat(f"window {n + 1}/{len(starts)}")
        with torch.no_grad():
            audio_in = proc(audios=chunk, sampling_rate=rate,
                            return_tensors="pt")
            if device == "cuda":
                audio_in = {k: v.to("cuda") for k, v in audio_in.items()}
            avec = model.get_audio_features(**audio_in)
            avec = (avec / avec.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

        pooled = avec if pooled is None else pooled + avec
        per_window += 1
        sims = tvec @ avec
        t = start / rate
        idx = job.shot_at(t)
        for rank, i in enumerate(np.argsort(-sims)[:3]):
            if float(sims[i]) < float(job.params.get("min_similarity", 0.05)):
                break
            em.claim("audio", "sound_event", SOUND_LABELS[i], shot_idx=idx,
                     num=round(float(sims[i]), 4),
                     confidence=round(float(sims[i]), 4),
                     ordinal=n * 3 + rank)

    if pooled is None:
        raise SkipPass("no usable audio windows")
    pooled = pooled / per_window
    pooled = pooled / (float(np.linalg.norm(pooled)) + 1e-9)
    em.vector("clap", [float(x) for x in pooled])

    overall = tvec @ pooled
    music_i = SOUND_LABELS.index("music playing")
    speech_i = SOUND_LABELS.index("a person speaking")
    em.claim("audio", "music_presence",
             "music-led" if overall[music_i] > overall[speech_i] else
             "speech-led", num=round(float(overall[music_i]), 4))
    for rank, i in enumerate(np.argsort(-overall)[:6]):
        em.claim("audio", "sound_event", SOUND_LABELS[i],
                 num=round(float(overall[i]), 4),
                 confidence=round(float(overall[i]), 4), ordinal=rank)

    em.notes = {"windows": per_window, "labels": len(SOUND_LABELS)}
    return em
