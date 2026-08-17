"""
vios.process.registry — every model and script declared as data.

This file is the answer to "what is actually running?". Nothing here executes.
Each component is a row: what it needs, what it costs, what it produces, what it
depends on. The rotation loop reads these rows to decide what to load; the
interface reads the same rows to draw the list of systems. There is exactly one
description of the pipeline and both the machine and the person read it.

Why data and not code:

  A pipeline written as a function is a pipeline you cannot look at. You cannot
  ask it "what would fit in 14 GB?", you cannot show it in a table, and you
  cannot add a model to it without editing control flow. A pipeline written as a
  list of rows answers all three by inspection.

Four rules govern what goes in this list.

**Mathematics before models.** Cut rate, loudness, colour, motion, tempo and
shot scale are computed, not asked. A model that is asked "how fast is this
cut?" produces a plausible sentence; `numpy.diff` over the shot table produces
the number. Nine of the components below never load a neural network at all, and
they carry the parts of the database that must be exactly right.

**Time is never claimed.** Every component that speaks about *when* speaks in
shot indices. The shot table maps an index to seconds by arithmetic we control.
No prompt anywhere in this system asks a model for a timestamp, because models
produce timestamps that look correct and are not.

**Two observers where it matters.** Speech, on-screen text and meaning each get
a second, architecturally different reader. Agreement is a confidence signal;
disagreement is a flag worth surfacing. This is the schema-level form of "i dont
trust one single llm, or model, or script".

**Sized for the machine that exists.** Every VRAM figure assumes Kaggle's 2 × T4:
Turing, SM 75, 16 GB per card, no NVLink. That means FP16 or INT4 (AWQ/GPTQ),
never BF16, never FP8, never FlashAttention-2 — see `resources.capabilities`.
Anything needing more than one card says so and is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from . import CHANNELS

# What a component can write. Used by the tab to show, per pass, whether it
# contributed evidence, geometry, files, or vectors.
# `metrics` is per-frame numbers written as packed columns (`frame_metric`)
# rather than a row per reading. It is listed separately from `claims` because a
# pass that writes only metrics writes no evidence rows at all, and the tab
# would otherwise report it as having contributed nothing.
OUTPUTS = ("video", "shots", "claims", "vectors", "artifacts", "metrics")

# Coarse ordering. Components sort by stage first, so a cohort never contains a
# perception pass scheduled ahead of the shot detection it reads.
STAGE_STRUCTURE, STAGE_SIGNAL, STAGE_PERCEPTION, STAGE_LANGUAGE = 0, 1, 2, 3
STAGE_NAMES = {
    STAGE_STRUCTURE: "structure",
    STAGE_SIGNAL: "signal",
    STAGE_PERCEPTION: "perception",
    STAGE_LANGUAGE: "language",
}

# Waves — the cut a stage cannot make.
#
# Stage says what a pass depends on. It does not say whether the pass is what
# makes the archive searchable, and stage 2 mixes the 40-second SigLIP that does
# with the 110-second Florence-2 that refines it. Running the whole registry over
# video 1 before video 2 begins means nothing is searchable until everything is
# done; running the spine over every video first means the archive is searchable
# — by text and by image — in the first hour, and every wave after that improves
# the same database in place.
#
# Wave 1 is need-closed: every hard need of a wave-1 pass is itself wave 1. That
# is what makes the wave safe rather than merely fast, and `wave_leaks` asserts
# it instead of trusting it.
WAVE_SPINE, WAVE_FULL, WAVE_DEEP = 1, 2, 3
WAVE_NAMES = {
    WAVE_SPINE: "spine",
    WAVE_FULL: "refine",
    WAVE_DEEP: "deep",
}


@dataclass(frozen=True)
class Component:
    """One pass over one video.

    `weights` is the field that makes cohorts cheap: three components can share
    one 6 GB VLM, and the packer must count those 6 GB once, not three times.
    Components with the same `weights` string load one model between them.
    """

    id: str
    title: str
    stage: int
    family: str                      # ffmpeg | signal | audio | vision | vlm | text
    summary: str                     # one line, shown in the tab's list
    detail: str = ""                 # the paragraph shown when a row is opened

    channel: str = ""                # "" for structural passes that emit no claims
    model: str = ""                  # the HF repo, or the library doing the work
    weights: str = ""                # VRAM-sharing key; defaults to `model`
    quant: str = ""                  # awq | int8_float16 | fp16 | ""
    revision: str = "1"              # bump to re-run a pass under a new observer

    device: str = "cpu"              # cpu | gpu
    cards: int = 1                   # 2 means it must be sharded across both T4s
    wave: int = WAVE_FULL            # 1 = the searchable spine, run first
    # Only wave 1 is hand-set, and only on the thirteen passes that produce
    # something a search box can reach: shots, transcript, caption, description,
    # and the three vector spaces. Wave 3 is derived rather than declared
    # (`wave_of`), so a pass that needs both cards or is off by default cannot be
    # left in an early wave by a field nobody remembered to change.
    vram_mb: int = 0
    ram_mb: int = 512
    disk_mb: int = 0                 # weights to download, for the disk check
    seconds: float = 1.0             # per video, on one T4, ~30 s reel

    needs: tuple = ()                # component ids that must finish first
    soft: tuple = ()                 # subset of `needs` whose absence is survivable
    # A soft need still orders the work — it stays in `needs`, so `topo_sort` and
    # `plan_cohorts` keep the edge — but a soft need that failed or skipped no
    # longer vetoes this pass. It is the difference between "OCR found no text"
    # and "OCR could not run": the runners were written for the first case
    # (`Job.text_bundle` returns an empty list when OCR wrote nothing) and the
    # engine's gate was turning the second into a deletion of every consumer.
    produces: tuple = ("claims",)
    kinds: tuple = ()                # claim kinds, so the tab can say what it wrote
    requires: tuple = ()             # importable module names, for preflight
    params: dict = field(default_factory=dict)

    tier: str = "core"               # core | deep | optional
    default_on: bool = True
    # False means: this pass is known not to run on Kaggle's 2×T4. It is data,
    # not a deletion — the component stays in the registry, the engine reports
    # it as `deferred` rather than `failed`, and a machine that can run it
    # picks it up later. Nothing is fought into working where it does not.
    kaggle_ok: bool = True
    notes: str = ""

    @property
    def load_key(self) -> str:
        return self.weights or self.model or self.id

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES.get(self.stage, str(self.stage))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["stage_name"] = self.stage_name
        d["load_key"] = self.load_key
        return d


# ══════════════════════════════════════════════════════════════════════════
# Stage 0 — structure. No model, no GPU, and everything else depends on it.
# ══════════════════════════════════════════════════════════════════════════

_STRUCTURE = [
    Component(
        id="probe", title="Probe", stage=STAGE_STRUCTURE, family="ffmpeg",
        wave=WAVE_SPINE,
        summary="Container truth: duration, resolution, fps, codecs, bitrate.",
        detail=(
            "ffprobe on the original mp4. Fills the video row rather than "
            "writing claims — a container's declared duration is not an "
            "observation anyone can disagree with. Runs first because every "
            "later pass needs to know how long the video is before it can "
            "decide how to sample it."),
        produces=("video",), seconds=0.8, ram_mb=128,
        requires=("subprocess",),
    ),
    Component(
        id="artifacts", title="Artifacts", stage=STAGE_STRUCTURE, family="ffmpeg",
        wave=WAVE_SPINE,
        summary="Poster, sprite sheet, 4-second loop, 480p proxy, waveform, wav.",
        detail=(
            "One ffmpeg invocation per artifact, all derived and all "
            "disposable. The 480p faststart proxy is what the site actually "
            "streams; the sprite sheet is what makes scrubbing instant; the "
            "16 kHz mono wav is what every audio pass reads, so the original "
            "is decoded once instead of six times."),
        produces=("artifacts",), seconds=14.0, ram_mb=1024,
        requires=("subprocess",),
        params={"proxy_height": 480, "sprite_cols": 10, "loop_seconds": 4},
    ),
    Component(
        id="shots", title="Shot detection", stage=STAGE_STRUCTURE, family="vision",
        wave=WAVE_SPINE,
        summary="TransNetV2 over a PySceneDetect prefilter — the atomic unit.",
        detail=(
            "Shots are the spine of the whole database. Every claim about "
            "'when' is a shot index, and this pass is the only thing allowed "
            "to decide where a shot begins. PySceneDetect's content detector "
            "runs first and cheaply on the proxy; TransNetV2 then resolves the "
            "soft cuts and dissolves that a threshold detector misses, which "
            "on edited reels is most of them."),
        model="Soulter/TransNetV2", weights="transnetv2", device="gpu",
        vram_mb=700, disk_mb=90, seconds=5.0, ram_mb=1536,
        needs=("artifacts",), produces=("shots",),
        requires=("torch", "scenedetect"),
        params={"threshold": 0.5, "min_shot_frames": 8},
    ),
    Component(
        id="keyframes", title="Keyframes", stage=STAGE_STRUCTURE, family="ffmpeg",
        wave=WAVE_SPINE,
        summary="One sharp representative frame per shot, plus its atlas tile.",
        detail=(
            "Picks the least-blurred frame in each shot by variance of the "
            "Laplacian rather than taking the midpoint, because the midpoint of "
            "a shot with camera movement is frequently the blurriest frame in "
            "it. Every vision pass downstream reads these frames, so a bad "
            "choice here degrades six components at once."),
        needs=("shots",), produces=("artifacts",), seconds=3.0, ram_mb=1024,
        requires=("cv2",), params={"candidates_per_shot": 5},
    ),
    Component(
        id="allframes", title="All frames", stage=STAGE_STRUCTURE, family="ffmpeg",
        wave=WAVE_SPINE,
        summary="Every frame of the video, with a per-frame timestamp and a "
                "coverage proof.",
        detail=(
            "Not a sample. One ffmpeg pass with -fps_mode passthrough writes "
            "exactly the frames the file contains — the default mode "
            "manufactures duplicates to force a constant rate, and on a "
            "110-frame test file it emitted 150, forty of them invented. Each "
            "frame is dated by the presentation timestamp read out of the "
            "stream rather than by index/fps, which drifts seconds off on the "
            "variable-rate video that most reels are. A coverage.json is "
            "written beside the frames recording four independent checks: "
            "every file present, the count against the stream's own "
            "declaration, every timestamp read rather than interpolated, and "
            "every wall-clock second holding at least one frame. That last "
            "check is the one a frame count cannot substitute for.\n\n"
            "Two tiers come out of the one decode, because the models want "
            "different things and neither compromise is acceptable. The 384 px "
            "analysis tier is at or above what SigLIP, CLIP and Depth-Anything "
            "ingest. The full tier keeps source resolution for OCR, detection "
            "and faces, where a twelve-pixel caption or a face at the back of "
            "a room is legible at 1080x1920 and simply absent at 384. The "
            "filter graph splits after decoding once, so the second tier costs "
            "an encode rather than another pass over the file."),
        needs=("probe",), produces=("artifacts",), seconds=45.0, ram_mb=1024,
        requires=("subprocess",),
        # `full_tier` ships on. It is roughly 200-400 MB per reel across both
        # tiers, in a Kaggle working directory that dies with the session
        # anyway, and the engine's cache budget evicts it once the last
        # consumer in the cohort is done. Resolution here is chosen for output
        # quality; scratch space is not the thing being optimised.
        params={"analysis_width": 384, "full_tier": True, "full_width": 0},
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# Stage 1 — signal. Arithmetic over pixels and samples. No model can be wrong
# about these numbers because no model is involved.
# ══════════════════════════════════════════════════════════════════════════

_SIGNAL = [
    Component(
        id="caption", title="Caption and comments", stage=STAGE_SIGNAL,
        wave=WAVE_SPINE,
        family="signal", channel="caption",
        summary="The creator's text, hashtags, mentions, and the captured comments.",
        detail=(
            "Reads the info JSON the capture plane stored beside the mp4. "
            "Hashtags and mentions are extracted structurally, not by a model, "
            "and comments are kept verbatim with their like counts — the "
            "audience's own words are the only direct evidence of what landed."),
        seconds=0.2, needs=(), produces=("claims",),
        kinds=("caption", "hashtag", "mention", "comment", "creator_reply",
               "likes", "views", "comments", "uploader"),
    ),
    Component(
        id="cuts", title="Cut rhythm", stage=STAGE_SIGNAL, family="signal",
        wave=WAVE_SPINE,
        channel="style",
        summary="Average shot length, cut rate, and how regular the rhythm is.",
        detail=(
            "Pure arithmetic over the shot table: ASL, cuts per minute, the "
            "coefficient of variation of shot lengths, and where the cut rate "
            "accelerates. Editing tempo is one of the strongest correlates of "
            "retention and it is a number, not an opinion."),
        seconds=0.2, needs=("shots",),
        kinds=("asl", "cut_rate", "rhythm", "acceleration"),
    ),
    Component(
        id="perframe", title="Per-frame measurement", stage=STAGE_SIGNAL,
        family="signal", channel="style",
        summary="Brightness, contrast, saturation, sharpness, motion and a "
                "perceptual hash for every frame in the video.",
        detail=(
            "The other pixel passes read the keyframe set, one image per shot. "
            "This one reads the complete frame set the allframes pass wrote "
            "and measures all of it, so the archive can answer what was "
            "happening at any moment rather than at forty moments. The "
            "per-frame numbers go to the store as packed columnar arrays — one "
            "row per metric per video, not one row per frame. A 900-frame reel "
            "on six metrics would be 5,400 rows and twenty-seven million "
            "across the archive, none of which anyone queries one at a time; "
            "as arrays it is six rows that load in one read. Only the summary "
            "becomes claims. The perceptual hash earns its place by "
            "finding freeze frames - a run of identical hashes is the "
            "difference between a video that ended and one that broke - and "
            "by making near-duplicate reels findable without a model."),
        seconds=10.0, ram_mb=1024, needs=("allframes",),
        revision="2",
        requires=("cv2", "numpy"),
        produces=("claims", "metrics"),
        kinds=("brightness_mean", "brightness_min", "brightness_max",
               "contrast_mean", "saturation_mean", "temperature_mean",
               "sharpness_mean", "motion_mean", "black_frames",
               "freeze_spans", "longest_freeze", "unique_frames"),
        params={"tier": "analysis", "stride": 1},
    ),
    Component(
        id="colour", title="Colour", stage=STAGE_SIGNAL, family="signal",
        channel="style",
        summary="Per-shot palette in CIELAB, plus per-frame colour statistics.",
        detail=(
            "k-means in CIELAB rather than RGB, because perceptual distance is "
            "the thing being measured and RGB distance is not it. Produces a "
            "five-swatch palette per shot plus the statistics that let 'high "
            "contrast, cool, desaturated' become a searchable query. The "
            "palette stays shot-level — a palette is a property of a look, and "
            "clustering one frame's pixels tells you less than clustering the "
            "shot's. The statistics are per-frame, because a grade that shifts "
            "mid-shot is exactly the kind of thing worth being able to find."),
        seconds=12.0, needs=("keyframes", "allframes"),
        revision="2",
        requires=("cv2", "numpy"),
        produces=("claims", "metrics"),
        kinds=("palette", "brightness", "contrast", "saturation", "temperature"),
        params={"swatches": 5, "tier": "analysis", "stride": 1},
    ),
    Component(
        id="motion", title="Motion and camera", stage=STAGE_SIGNAL, family="signal",
        channel="style",
        summary="Frame-difference energy and an affine fit for camera movement.",
        detail=(
            "Estimates a global affine transform between consecutive frames and "
            "reads the camera move out of its parameters — translation is a "
            "pan, uniform scale is a zoom, high residual is handheld. "
            "Classifying camera movement by fitting a transform is exact where "
            "asking a vision model the same question is a guess dressed as an "
            "answer. Fitted between every consecutive pair from the extracted "
            "frame set rather than by re-decoding, so the motion curve is "
            "sampled at the video's own frame rate and the fit is against the "
            "frames every other pass saw."),
        seconds=25.0, needs=("shots", "allframes"),
        revision="2",
        requires=("cv2", "numpy"),
        produces=("claims", "metrics"),
        kinds=("camera_move", "motion_energy", "stability"),
        params={"tier": "analysis", "stride": 1},
    ),
    Component(
        id="loudness", title="Loudness", stage=STAGE_SIGNAL, family="signal",
        channel="audio",
        summary="EBU R128 LUFS, true peak, dynamic range, silence ratio.",
        detail=(
            "The broadcast standard, computed the standard way. Loudness "
            "profile over time is what separates a reel that opens on a beat "
            "drop from one that opens on a whisper, and the silence ratio is a "
            "reliable structural marker for pauses used as emphasis."),
        seconds=1.5, revision=2,
        needs=("artifacts",), requires=("numpy", "soundfile"),
        produces=("claims", "metrics"),
        kinds=("lufs", "true_peak", "dynamic_range", "silence_ratio",
               "loudness_curve", "shot_level", "silence"),
    ),
    Component(
        id="music", title="Tempo and key", stage=STAGE_SIGNAL, family="signal",
        channel="audio",
        summary="Beat tracking, BPM, key estimate, harmonic/percussive split.",
        detail=(
            "librosa's beat tracker for tempo and beat grid, Krumhansl–Schmuckler "
            "for key, HPSS for the harmonic/percussive ratio. Knowing where the "
            "beats are makes 'the cut lands on the downbeat' a checkable "
            "statement rather than a stylistic impression."),
        seconds=4.0, needs=("artifacts",), requires=("librosa", "numpy"),
        kinds=("tempo", "key", "beat_grid", "harmonic_ratio"),
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# Stage 2 — perception. What is in the audio and on the screen.
# ══════════════════════════════════════════════════════════════════════════

_PERCEPTION = [
    # ── speech ──────────────────────────────────────────────────────────
    Component(
        id="transcribe", title="Transcribe", stage=STAGE_PERCEPTION, family="audio",
        wave=WAVE_SPINE,
        channel="speech",
        summary="Whisper large-v3 via CTranslate2, word timestamps, 99 languages.",
        detail=(
            "The primary reader of speech. INT8/FP16 on CTranslate2 rather than "
            "the reference implementation because it is roughly four times "
            "faster on a T4 for the same weights. Language is detected per "
            "video, not assumed: the archive is English, Hindi and a long tail "
            "of others, and a pass that assumes English silently produces "
            "confident nonsense on the tail. When detection confidence is low "
            "the video is flagged rather than transcribed badly."),
        model="Systran/faster-whisper-large-v3", weights="whisper-large-v3",
        quant="int8_float16", device="gpu", vram_mb=3100, disk_mb=3100,
        seconds=9.0, ram_mb=2048, revision=2,
        needs=("artifacts",), produces=("claims",),
        requires=("faster_whisper",),
        kinds=("transcript", "segment", "words", "language",
               "language_uncertain", "speech_rate"),
        params={"beam_size": 5, "vad_filter": True, "word_timestamps": True,
                "condition_on_previous_text": False},
        notes="condition_on_previous_text off: it causes repetition loops on "
              "music-heavy reels, which is most of them. Segments keep "
              "Whisper's own boundaries and additionally carry the frame span "
              "they cover, so speech shares one index with everything the "
              "vision passes wrote.",
    ),
    Component(
        id="transcribe-alt", title="Transcribe (second reader)",
        stage=STAGE_PERCEPTION, family="audio", channel="speech",
        summary="large-v3-turbo as an independent observer of the same audio.",
        detail=(
            "A different decoder over the same waveform. Where the two agree, "
            "confidence is high and search can trust the words; where they "
            "diverge, the segment is marked contested and the interface shows "
            "both. This costs five seconds a reel and is the cheapest quality "
            "signal in the entire pipeline."),
        model="mobiuslabsgmbh/faster-whisper-large-v3-turbo", weights="whisper-turbo",
        quant="int8_float16", device="gpu", vram_mb=1600, disk_mb=1600,
        seconds=5.0, revision=2,
        needs=("artifacts",), requires=("faster_whisper",),
        kinds=("transcript", "segment", "language", "agreement", "contested"),
        params={"beam_size": 1, "vad_filter": True},
    ),
    Component(
        id="diarize", title="Speakers", stage=STAGE_PERCEPTION, family="audio",
        channel="speech",
        summary="pyannote 3.1: how many voices, and who speaks when.",
        detail=(
            "Speaker turns, not speaker names. Distinguishing a monologue from "
            "an interview from a voiceover-over-b-roll is structural "
            "information that changes how every other reading of the reel "
            "should be interpreted. No identity is inferred or stored."),
        model="pyannote/speaker-diarization-3.1", weights="pyannote-3.1",
        device="gpu", vram_mb=1400, disk_mb=100, seconds=7.0, revision=2,
        needs=("artifacts",), requires=("pyannote.audio",),
        kinds=("speaker_turn", "speaker_count", "speaker_share", "overlap"),
        notes="Needs a Hugging Face token and gated-model acceptance. A turn "
              "is written as the frame span it covers, not the shot it began "
              "in — the speaker changes mid-shot far more often than at a cut.",
    ),
    Component(
        id="audio-tag", title="Sound events", stage=STAGE_PERCEPTION, family="audio",
        channel="audio",
        summary="CLAP over a gapless 0.96 s grid: every sound, at full resolution.",
        detail=(
            "Music, laughter, applause, whooshes, room tone, silence. A "
            "transcript records the words and misses everything else, and on "
            "short-form video the everything else is half the craft. "
            "Scoring runs on a fixed grid of 0.96 s windows at a 0.48 s hop "
            "covering the whole track with no gaps — the audio equivalent of "
            "'every frame' — because a sound attributed to a shot answers "
            "'there was applause somewhere in these eleven seconds', which "
            "cannot place a moment. 0.96 s is also what CLAP was trained to "
            "ingest, so a window is one whole model input rather than a slice "
            "of one. Every label keeps its full score curve as a packed metric "
            "array, so a later feature can pick its own threshold without "
            "re-running the model over the archive, and each label that stays "
            "above threshold across consecutive windows collapses into one "
            "run-length claim carrying the frame span it covers. CLAP also "
            "yields per-window embeddings, which makes 'find the moment that "
            "sounds like this' a query over one timeline rather than a "
            "comparison of pooled averages."),
        model="laion/clap-htsat-fused", weights="clap-panns", device="gpu",
        vram_mb=1300, disk_mb=1800, seconds=14.0, revision=2,
        needs=("artifacts",),
        produces=("claims", "vectors", "metrics"),
        requires=("torch", "transformers"),
        kinds=("sound_event", "music_presence"),
        params={"batch": 16, "window_seconds": 0.96, "hop_seconds": 0.48,
                "top_k": 3, "min_similarity": 0.05},
    ),

    # ── on-screen text ──────────────────────────────────────────────────
    Component(
        id="ocr", title="On-screen text", stage=STAGE_PERCEPTION, family="vision",
        channel="ocr",
        summary="PP-OCRv5 (EasyOCR fallback), with temporal deduplication.",
        detail=(
            "Reels caption themselves; the words burned into the frame are "
            "often the whole message and never appear in the transcript. "
            "Detection runs on every frame at source resolution, then identical "
            "consecutive readings collapse into one claim spanning a frame run, "
            "so a caption held for eight seconds is one row that still answers "
            "exactly what was on screen at any frame inside it. Reading "
            "keyframes instead would have meant one frame per shot: a caption "
            "that appears and leaves inside a single shot was invisible, and "
            "that is most captions.\n\n"
            "PP-OCR is preferred and EasyOCR is the fallback, because "
            "paddlepaddle-gpu is a ~700 MB wheel that must match the CUDA "
            "build and is deliberately not installed here — without a backend "
            "every PaddleOCR constructor fails. EasyOCR is torch, is already "
            "installed, and the emission records which engine answered."),
        model="PaddlePaddle/PP-OCRv5", weights="ppocr-v5", device="gpu",
        revision="2",
        vram_mb=1500, disk_mb=400, seconds=90.0,
        needs=("allframes",), requires=("paddleocr",),
        kinds=("text", "text_region", "text_density", "text_style"),
        params={"languages": ["en", "devanagari"], "min_confidence": 0.6,
                "tier": "full", "batch": 8, "stride": 1},
    ),
    Component(
        id="ocr-alt", title="On-screen text (second reader)",
        stage=STAGE_PERCEPTION, family="vision", channel="ocr",
        summary="Florence-2 reading the same frames with a different architecture.",
        detail=(
            "A detector-plus-recogniser and an end-to-end vision-language model "
            "fail in different ways: the first drops stylised type, the second "
            "hallucinates plausible words. Running both and keeping both means "
            "the failures do not overlap, and agreement between them is a much "
            "stronger signal than either one's own confidence score.\n\n"
            "Its remote code needs timm, and its flash-attention guard has to "
            "be stripped from the import check because flash-attn needs Ampere "
            "and these are T4s. `hf_revision` pins the remote code to one "
            "commit: it is executed in this process, so 'whatever main says "
            "today' is not a version."),
        model="microsoft/Florence-2-large", weights="florence-2-large",
        quant="fp16", device="gpu", vram_mb=1700, disk_mb=1600, seconds=110.0,
        revision="3",
        needs=("allframes",), requires=("transformers", "torch", "timm"),
        kinds=("text",),
        params={"tier": "full", "batch": 4, "stride": 1,
                "hf_revision": "21a599d414c4d928c9032694c424fb94458e3594"},
    ),

    # ── what is in frame ────────────────────────────────────────────────
    Component(
        id="detect", title="Objects", stage=STAGE_PERCEPTION, family="vision",
        channel="visual",
        summary="YOLO11 over every frame: boxes, classes, counts, screen share.",
        detail=(
            "A closed vocabulary, deliberately. Eighty classes detected "
            "reliably with boxes and counts are worth more than a thousand "
            "classes guessed, because counts and positions are the inputs to "
            "later arithmetic — screen share over time, entrance and exit, "
            "how much of the frame the subject occupies. Every one of those "
            "is a per-frame quantity: a keyframe told you a person was in the "
            "shot, and every frame tells you when they entered it, where they "
            "moved and when they left."),
        model="Ultralytics/YOLO11", weights="yolo11x", quant="fp16",
        device="gpu", vram_mb=1200, disk_mb=120, seconds=70.0,
        revision="2",
        needs=("allframes",), requires=("ultralytics",),
        kinds=("object", "object_count", "screen_share", "object_track"),
        params={"conf": 0.35, "imgsz": 960, "tier": "full", "batch": 16,
                "stride": 1},
    ),
    Component(
        id="faces", title="Faces and framing", stage=STAGE_PERCEPTION,
        family="vision", channel="visual",
        summary="SCRFD detection and tracking: how many people, how close.",
        detail=(
            "Face count, box size relative to frame, and track continuity "
            "across every frame. Face size is the most reliable proxy for shot "
            "scale on people-centred video, and 'talking head cut to b-roll cut "
            "back' is a structure that can be detected geometrically. Per-frame "
            "coverage is what makes the tracks real rather than inferred: a "
            "face present in two keyframes might be one person throughout or "
            "two people either side of a cut, and only the frames between them "
            "answer that. Embeddings are used only to link tracks within one "
            "video and are not stored as identity."),
        model="deepinsight/insightface-scrfd", weights="insightface",
        device="gpu", vram_mb=900, disk_mb=350, seconds=60.0,
        revision="2",
        needs=("allframes",), requires=("insightface",),
        kinds=("face_count", "face_scale", "face_track"),
        params={"tier": "full", "batch": 8, "stride": 1},
    ),
    Component(
        id="depth", title="Shot scale", stage=STAGE_PERCEPTION, family="vision",
        channel="style",
        summary="Depth Anything V2 small — close-up, medium, or wide, per frame.",
        detail=(
            "Relative depth over every frame, reduced to the depth spread of "
            "the central region. A close-up has a flat, near histogram; a wide "
            "shot has a long tail. This gives shot scale for footage with no "
            "faces in it, where the face-size proxy has nothing to measure. The "
            "per-frame spread is stored as a packed metric array rather than as "
            "claims, because it is a curve to be read as a curve — a push-in is "
            "a slope in it, and that is invisible in one number per shot.\n\n"
            "The repo id carries the `-hf` suffix on purpose. "
            "`depth-anything/Depth-Anything-V2-Small` is the authors' original "
            "checkpoint: its config.json has no `model_type`, so "
            "`AutoModelForDepthEstimation` cannot dispatch and every load raised "
            "'Unrecognized model'. The `-hf` mirror is the same weights in "
            "transformers layout."),
        model="depth-anything/Depth-Anything-V2-Small-hf",
        weights="depth-anything-v2-s",
        quant="fp16", device="gpu", vram_mb=500, disk_mb=100, seconds=45.0,
        revision="3",
        needs=("allframes",), requires=("transformers", "torch"),
        produces=("claims", "metrics"),
        kinds=("shot_scale", "depth_spread"),
        params={"tier": "analysis", "batch": 16, "stride": 1},
        tier="optional",
    ),
    Component(
        id="visual-embed", title="Visual embedding", stage=STAGE_PERCEPTION,
        wave=WAVE_SPINE,
        family="vision", channel="visual",
        summary="SigLIP-2 per frame — the vectors behind visual search.",
        detail=(
            "One vector per frame, one per shot and one pooled per video. This "
            "is what makes 'find the moment that looks like this' work, what "
            "powers near-duplicate detection across five thousand reels, and "
            "what the aesthetic head and the zero-shot tagger both read. "
            "Per-frame is the difference between finding the reel and finding "
            "the second: a shot-level vector is an average of everything in the "
            "shot, and the one frame the query is actually about is averaged "
            "away. Computed once, used by four things."),
        model="google/siglip2-so400m-patch14-384", weights="siglip2-so400m",
        quant="fp16", device="gpu", vram_mb=1800, disk_mb=1700, seconds=40.0,
        revision="2",
        needs=("allframes",), produces=("vectors",),
        requires=("transformers", "torch"),
        kinds=(),
        params={"tier": "analysis", "batch": 32, "stride": 1},
    ),
    Component(
        id="clip-embed", title="Visual embedding (second space)",
        wave=WAVE_SPINE,
        stage=STAGE_PERCEPTION, family="vision", channel="visual",
        summary="CLIP ViT-L/14 over the same frames — a second, independent "
                "retrieval space.",
        detail=(
            "Not redundancy. SigLIP and CLIP were trained on different data "
            "with different objectives and they fail on different queries: "
            "CLIP is stronger on proper nouns, logos and the kind of "
            "internet-native phrasing a search box actually receives, SigLIP "
            "on dense scene description. A moment retrievable by either is "
            "retrievable, which is what 'accurate' has to mean when the query "
            "is written by a person rather than chosen from a list. Where the "
            "two disagree that is evidence about the frame too, and both are "
            "kept. fp16 because Kaggle's T4s are sm_75 — no BF16, no FP8. The "
            "load key matches the v1 loader's so the two generations share one "
            "copy of the weights instead of each paying 1.7 GB."),
        model="openai/clip-vit-large-patch14",
        weights="openai/clip-vit-large-patch14",
        quant="fp16", device="gpu", vram_mb=1700, disk_mb=1700, seconds=35.0,
        needs=("allframes",), produces=("vectors",),
        requires=("transformers", "torch"),
        kinds=(),
        params={"tier": "analysis", "batch": 32, "stride": 1},
    ),
    Component(
        id="tag", title="Zero-shot tags", stage=STAGE_PERCEPTION, family="vision",
        wave=WAVE_SPINE,
        channel="visual",
        summary="Cosine similarity against a curated label set. No prompting.",
        detail=(
            "The label vocabulary — settings, activities, formats, objects "
            "specific to this archive — is embedded once, and tagging is then a "
            "matrix multiply against vectors that already exist. Free, "
            "deterministic, and extensible: adding forty new labels next month "
            "re-tags five thousand reels in seconds without touching a GPU. "
            "Now run against the per-frame vectors, so a tag carries the frame "
            "run it was true over instead of being a property of the whole "
            "video — which is what makes a tag a way to find a moment."),
        needs=("visual-embed",), seconds=2.0,
        revision="2",
        requires=("numpy",), kinds=("tag",),
        params={"top_k": 12, "min_similarity": 0.22, "min_run": 2},
    ),
    Component(
        id="aesthetic", title="Frame quality", stage=STAGE_PERCEPTION,
        family="vision", channel="style",
        summary="Sharpness, exposure, clipping, noise, colourfulness, quality probe.",
        detail=(
            "Five classical no-reference measures — variance of the Laplacian, "
            "mean luminance, clipped-pixel share, a high-pass MAD noise "
            "estimate and Hasler–Süsstrunk colourfulness — plus a relative "
            "quality probe taken as the margin between 'well shot' and 'badly "
            "shot' text embeddings against the SigLIP vector that already "
            "exists. Deliberately not the LAION aesthetic MLP: that head is "
            "trained on CLIP ViT-L/14 features and would be meaningless over "
            "SigLIP ones. Measured on every frame and stored as packed metric "
            "arrays, because the useful question is which *moments* are badly "
            "shot, not whether the reel on average was. Useful less as a "
            "verdict than as a filter — when a pattern only appears in "
            "badly-exposed frames, the pattern is about the exposure."),
        weights="aesthetic-probe", device="gpu", vram_mb=200, disk_mb=0,
        seconds=25.0,
        revision="2",
        needs=("visual-embed", "allframes"), requires=("cv2", "numpy"),
        produces=("claims", "metrics"),
        kinds=("aesthetic", "sharpness", "exposure", "clipping", "noise",
               "colourfulness"),
        params={"tier": "analysis", "batch": 32, "stride": 1},
        tier="optional",
    ),
]

# ══════════════════════════════════════════════════════════════════════════
# Stage 3 — language. The only place a model is asked what something means,
# and it is asked in shot indices.
# ══════════════════════════════════════════════════════════════════════════

_VLM = "qwen3-vl-8b-awq"

_LANGUAGE = [
    Component(
        id="describe", title="Describe shots", stage=STAGE_LANGUAGE, family="vlm",
        wave=WAVE_SPINE,
        channel="visual",
        summary="Qwen3-VL 8B AWQ: what is physically present, shot by shot.",
        detail=(
            "Literal description only — objects, people, setting, action, text "
            "position. No interpretation, no intent, no timestamps. The model "
            "receives the keyframes of a bounded range of shots and answers "
            "about shot 3, not about 00:07, because it is not able to know what "
            "00:07 is and will invent it if asked."),
        model="Qwen/Qwen3-VL-8B-Instruct-AWQ", weights=_VLM, quant="awq",
        device="gpu", vram_mb=6200, disk_mb=6000, seconds=18.0, ram_mb=3072,
        needs=("keyframes", "shots"), requires=("transformers", "torch"),
        kinds=("shot_description", "subject", "setting", "action"),
        params={"max_shots_per_call": 6, "max_new_tokens": 320,
                "temperature": 0.2},
    ),
    Component(
        id="narrate", title="Narrative", stage=STAGE_LANGUAGE, family="vlm",
        channel="narrative",
        summary="The hook, the turn, the payoff — as shot ranges, with reasons.",
        detail=(
            "Same weights as `describe`, a different question, and the "
            "expensive one. It is given the transcript, the on-screen text, the "
            "shot descriptions and the loudness curve, and asked what the video "
            "is doing and why it holds attention — answering in shot ranges. "
            "This is the pass whose output most directly serves script writing "
            "and pattern recognition, and it is deliberately last, because it "
            "reads everything the earlier passes produced."),
        model="Qwen/Qwen3-VL-8B-Instruct-AWQ", weights=_VLM, quant="awq",
        device="gpu", vram_mb=6200, disk_mb=0, seconds=20.0, ram_mb=3072,
        needs=("describe", "transcribe", "ocr", "loudness", "cuts"),
        # `describe` is the subject; the rest enrich the prompt. narrate carries
        # its own truthful guard — "nothing was said, written or captioned" —
        # so an OCR engine that could not start must not delete this pass.
        soft=("transcribe", "ocr", "loudness", "cuts"),
        requires=("transformers", "torch"),
        kinds=("hook", "beat", "turn", "payoff", "premise", "why_it_works",
               "weakness"),
        params={"max_new_tokens": 700, "temperature": 0.3},
    ),
    Component(
        id="style-read", title="Craft", stage=STAGE_LANGUAGE, family="vlm",
        channel="style",
        summary="Framing, lighting, grade and edit, read against the measurements.",
        detail=(
            "The model is handed the numbers the signal passes computed — cut "
            "rate, palette, camera moves, shot scale — and asked to name the "
            "technique they add up to. Grounding the reading in measurements is "
            "what stops it becoming film-school vocabulary applied at random; "
            "every stylistic claim it makes has arithmetic sitting under it."),
        model="Qwen/Qwen3-VL-8B-Instruct-AWQ", weights=_VLM, quant="awq",
        device="gpu", vram_mb=6200, disk_mb=0, seconds=12.0, ram_mb=3072,
        needs=("describe", "colour", "motion", "cuts"),
        # The measurements are what ground the vocabulary, and the pass already
        # prints "not measured" for any of them that is absent.
        soft=("colour", "motion", "cuts"),
        requires=("transformers", "torch"),
        kinds=("technique", "lighting", "grade", "framing", "edit_style",
               "reference"),
    ),
    Component(
        id="keyphrase", title="Keyphrases", stage=STAGE_LANGUAGE, family="text",
        channel="concept",
        summary="YAKE and KeyBERT over the assembled text. Deterministic.",
        detail=(
            "A statistical extractor and an embedding-based one, both "
            "reproducible, both incapable of inventing a phrase that is not in "
            "the text. They are the control group against which the language "
            "model's concept extraction is judged: a concept the model reports "
            "and neither extractor can find is a concept worth doubting."),
        needs=("transcribe", "ocr", "caption"), seconds=0.6,
        # Three sources of text, any one of which is enough. The pass raises its
        # own skip naming all three when every one of them is empty.
        soft=("transcribe", "ocr", "caption"),
        requires=("yake",), kinds=("keyphrase",),
    ),
    Component(
        id="concepts", title="Concepts and entities", stage=STAGE_LANGUAGE,
        family="text", channel="concept",
        summary="Qwen3 4B AWQ over the assembled text: entities, topics, techniques.",
        detail=(
            "Normalised concept names with the span of text that evidences "
            "each, which is what makes the knowledge graph traversable and its "
            "edges auditable. Only concepts that both this pass and the "
            "keyphrase extractor support become graph nodes; the rest are kept "
            "as claims but do not get to shape the structure."),
        model="Qwen/Qwen3-4B-Instruct-AWQ", weights="qwen3-4b-awq", quant="awq",
        device="gpu", vram_mb=3400, disk_mb=3000, seconds=8.0,
        needs=("transcribe", "ocr", "caption", "keyphrase"),
        # Same three text sources, plus the statistical control it is scored
        # against. Without the control every concept simply lands unsupported,
        # which is a weaker claim rather than a missing one.
        soft=("transcribe", "ocr", "caption", "keyphrase"),
        requires=("transformers", "torch"),
        kinds=("entity", "topic", "technique", "claim_span", "unsupported"),
    ),
    Component(
        id="text-embed", title="Text embedding", stage=STAGE_LANGUAGE,
        wave=WAVE_SPINE,
        family="text", channel="concept",
        summary="BGE-M3 over transcript, caption, OCR and narration.",
        detail=(
            "Multilingual by construction, which matters when the same archive "
            "holds English, Hindi and Hinglish and the query may be in any of "
            "them. Embeds each passage separately rather than the video as a "
            "whole, so a semantic hit lands on a moment instead of on a "
            "twenty-thousand-character blur."),
        model="BAAI/bge-m3", weights="bge-m3", quant="fp16", device="gpu",
        vram_mb=2400, disk_mb=2300, seconds=2.0,
        needs=("transcribe", "caption"), produces=("vectors",),
        # It embeds whatever passages exist — speech, caption, on-screen text,
        # narrative beats — and skips itself when there are none.
        soft=("transcribe", "caption"),
        requires=("transformers", "torch"), kinds=(),
    ),
    Component(
        id="hook", title="Hook analysis", stage=STAGE_LANGUAGE, family="signal",
        channel="narrative",
        summary="The first three seconds, measured: words, cuts, loudness, text.",
        detail=(
            "Assembly, not inference. What is said, what is written, how many "
            "cuts, how loud, and what is in frame during the window where "
            "retention is decided — all of it read out of claims that already "
            "exist. Cheap to compute and re-computable the instant the "
            "definition of 'the hook' changes."),
        needs=("transcribe", "ocr", "cuts", "loudness", "describe"),
        # Pure assembly: every input is optional and each one absent costs one
        # claim, not the pass. `shots` is the only thing it cannot do without,
        # and that arrives through `cuts`.
        soft=("transcribe", "ocr", "cuts", "loudness"),
        seconds=0.3, kinds=("hook_words", "hook_cuts", "hook_loudness",
                            "hook_text", "hook_visual"),
        params={"window_seconds": 3.0},
    ),

    # ── the deep pass, off by default ───────────────────────────────────
    Component(
        id="narrate-deep", title="Narrative (38B)", stage=STAGE_LANGUAGE,
        family="vlm", channel="narrative",
        summary="InternVL3 38B AWQ across both cards, for the reels worth it.",
        detail=(
            "Sharded over two T4s with device_map, which without NVLink means "
            "PCIe traffic on every layer boundary and roughly a ninety-second "
            "pass. Not worth it for five thousand reels; clearly worth it for "
            "the few hundred that outperformed, where the difference between a "
            "good reading and an excellent one compounds into everything "
            "written afterwards. Off by default, pointed at a subset."),
        model="OpenGVLab/InternVL3-38B-AWQ", weights="internvl3-38b-awq",
        quant="awq", device="gpu", cards=2, vram_mb=21500, disk_mb=21000,
        seconds=95.0, ram_mb=6144,
        needs=("describe", "transcribe", "ocr", "narrate"),
        soft=("transcribe", "ocr"),
        requires=("transformers", "torch"),
        kinds=("hook", "beat", "turn", "payoff", "why_it_works", "critique"),
        tier="deep", default_on=False,
        notes="Two cards. Nothing else can be resident while this runs.",
    ),

    # ── the same quality tier, on someone else's hardware ───────────────
    Component(
        id="narrate-cloud", title="Narrative (cloud)", stage=STAGE_LANGUAGE,
        family="cloud", channel="narrative",
        summary="The deep reading through NVIDIA's hosted API — one call per video.",
        detail=(
            "`narrate-deep` is the one pass that genuinely does not belong on "
            "a T4: a 38B model sharded over PCIe, ninety-five seconds a video, "
            "nothing else resident. Rather than fight that, this runs the same "
            "reading against a hosted frontier model. It costs no VRAM at all, "
            "so the packer schedules it *alongside* a full GPU cohort instead "
            "of displacing one — the frontier-quality reading happens in the "
            "gaps between local passes. "
            "One call per video, never one per chunk: the account is limited "
            "to roughly forty requests a minute and metered in credits, and "
            "calling per chunk against a 550B model is exactly what produced "
            "`ResourceExhausted (33/32)` in the v1 log. A rate-limited video "
            "is deferred with a clock on it, never failed, because coverage is "
            "idempotent and resuming tomorrow costs nothing but time."),
        model="", weights="", device="cpu", vram_mb=0, disk_mb=0,
        seconds=25.0, ram_mb=256,
        needs=("describe", "transcribe", "ocr", "narrate", "detect"),
        # The cloud model is handed the local narrative to second-guess, so
        # `describe` and `narrate` are its subject. The measurement channels it
        # quotes are not: `_signal_summary` prints what is there.
        soft=("transcribe", "ocr", "detect"),
        requires=("openai",),
        produces=("claims",),
        kinds=("premise", "hook", "hook_why", "beat", "turn", "payoff",
               "why_it_works", "assertion", "answers", "audience", "subject",
               "critique"),
        params={"temperature": 0.2, "max_tokens": 2048},
        tier="deep", default_on=True,
        notes=("Needs VIOS_NIM_API_KEY in Kaggle Secrets. VIOS_NIM_MODEL "
               "picks the model; VIOS_NIM_RPM caps the request rate. "
               "On by default: it is the only pass that costs no VRAM, so "
               "leaving it off meant the key sat idle through an entire run "
               "while the GPU passes queued behind each other. With no key it "
               "declines on the first video and says so, which is a cheaper "
               "way to find out than a run that never used it."),
    ),
]

CATALOGUE = tuple(_STRUCTURE + _SIGNAL + _PERCEPTION + _LANGUAGE)
BY_ID = {c.id: c for c in CATALOGUE}


# ══════════════════════════════════════════════════════════════════════════
# Reading the catalogue
# ══════════════════════════════════════════════════════════════════════════

def get(component_id: str) -> Component:
    try:
        return BY_ID[component_id]
    except KeyError:
        raise KeyError(f"no such component: {component_id!r}") from None


def all_ids() -> list:
    return [c.id for c in CATALOGUE]


def defaults() -> list:
    return [c.id for c in CATALOGUE if c.default_on]


def stage_of(component_id: str) -> str:
    """The stage name a component belongs to, or "" if it is not in the
    catalogue at all."""
    c = BY_ID.get(component_id)
    return c.stage_name if c else ""


def stages_of(ids) -> list:
    """The stage names present in `ids`, in stage order.

    Used to cut the channel bundles: a cohort is a VRAM grouping and can hold
    two stages at once, so "which stages did this cohort touch" is a question
    the engine has to ask rather than assume.
    """
    have = {BY_ID[i].stage for i in ids if i in BY_ID}
    return [STAGE_NAMES[s] for s in sorted(have) if s in STAGE_NAMES]


def components_in_stage(stage: str, ids=None) -> list:
    """Component ids in one stage, restricted to `ids` when given.

    The restriction matters: a stage report for a session that deselected half
    of `perception` must describe what was actually planned, not what the
    catalogue could have run. Otherwise every bundle would look permanently
    incomplete.
    """
    pool = list(ids) if ids is not None else all_ids()
    return [i for i in pool
            if i in BY_ID and BY_ID[i].stage_name == stage]


def group_members(label: str, ids=None, spine=None) -> list:
    """Resolve a bundle label back to the components it covered.

    A snapshot bundle is named after the group it was cut on, and since the
    rotation cuts on wave boundaries that label is a wave name (`spine`,
    `refine`, `deep`) as often as it is a stage name. A retry after a failed
    upload has only the label to work from, so it needs one function that
    answers both — otherwise a wave-labelled bundle looks like a stage that no
    longer exists and its pending flag is silently cleared.

    `spine` must be the same override the rotation planned with, or a wave name
    resolves to a different set of components than the bundle carried.

    Stage names win a collision, because they are the older meaning and
    `stages_of` still produces them; no stage and no wave share a name today
    and `validate` has no reason to allow it.
    """
    pool = list(ids) if ids is not None else all_ids()
    hit = [i for i in pool if i in BY_ID and BY_ID[i].stage_name == label]
    if hit:
        return hit
    for wave, name in WAVE_NAMES.items():
        if name == label or label == str(wave):
            return [i for w, group in waves(pool, spine)
                    if w == wave for i in group]
    return []


def validate() -> list:
    """Structural checks over the catalogue itself.

    Called at import time by the engine and surfaced in the tab. A typo in a
    `needs` tuple or a channel that does not exist would otherwise become a
    runtime failure three cohorts into a twelve-hour session, which is the worst
    possible time to discover it.
    """
    problems = []
    seen = set()
    for c in CATALOGUE:
        if c.id in seen:
            problems.append(f"{c.id}: duplicate id")
        seen.add(c.id)
        if c.channel and c.channel not in CHANNELS:
            problems.append(f"{c.id}: unknown channel {c.channel!r}")
        for n in c.needs:
            if n not in BY_ID:
                problems.append(f"{c.id}: needs unknown component {n!r}")
            elif BY_ID[n].stage > c.stage:
                problems.append(
                    f"{c.id} (stage {c.stage}) needs {n} (stage {BY_ID[n].stage})")
        for p in c.produces:
            if p not in OUTPUTS:
                problems.append(f"{c.id}: unknown output {p!r}")
        if "claims" in c.produces and not c.channel:
            problems.append(f"{c.id}: writes claims but declares no channel")
        if c.device == "gpu" and not c.vram_mb:
            problems.append(f"{c.id}: gpu component with no vram_mb")
    # cycles
    try:
        topo_sort(all_ids())
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def topo_sort(ids) -> list:
    """Dependency order, stable and deterministic.

    Ties break on (stage, id) rather than insertion order so two workers on two
    accounts derive the identical plan from the identical catalogue — which is
    what lets them share a coverage table without ever talking.
    """
    want = list(dict.fromkeys(ids))
    unknown = [i for i in want if i not in BY_ID]
    if unknown:
        raise KeyError(f"unknown components: {unknown}")
    wanted = set(want)
    ready, out = [], []
    remaining = {i: {n for n in BY_ID[i].needs if n in wanted} for i in want}
    ready = sorted((i for i, d in remaining.items() if not d),
                   key=lambda i: (BY_ID[i].stage, i))
    while ready:
        cur = ready.pop(0)
        out.append(cur)
        newly = []
        for i, deps in remaining.items():
            if cur in deps:
                deps.discard(cur)
                if not deps and i not in out and i not in ready:
                    newly.append(i)
        ready = sorted(ready + newly, key=lambda i: (BY_ID[i].stage, i))
        remaining.pop(cur, None)
    if remaining:
        raise ValueError(f"dependency cycle among {sorted(remaining)}")
    return out


def wave_of(component) -> int:
    """Which wave a component belongs to, derivation included.

    Wave 3 is derived rather than declared: a pass that needs both cards cannot
    share a cohort with anything, and a pass that is off by default is not part
    of the archive's baseline. Either way it belongs after the passes that are.
    Deriving it means a new component lands in the right wave with no field set.
    """
    c = component if isinstance(component, Component) else BY_ID.get(component)
    if c is None:
        return WAVE_FULL
    if c.cards > 1 or not c.default_on:
        return WAVE_DEEP
    return int(c.wave)


def waves(ids, spine=None) -> list:
    """`[(wave, [component ids])]`, ascending, over the ids given.

    `spine` overrides which components are wave 1 (the `VIOS_WAVE1` env var).
    An override is closed over its own needs by the caller — `close_needs` — so
    a hand-written spine cannot leave a pass waiting for an input that is
    scheduled two waves later.
    """
    out: dict = {}
    over = set(spine) if spine else None
    for cid in ids:
        c = BY_ID.get(cid)
        if c is None:
            continue
        w = wave_of(c)
        if over is not None:
            w = WAVE_SPINE if cid in over else max(w, WAVE_FULL)
        out.setdefault(w, []).append(cid)
    return [(w, out[w]) for w in sorted(out)]


def close_needs(ids, universe=None) -> list:
    """`ids` plus every component they transitively need, within `universe`.

    This is what makes a wave safe. A wave whose members need something from a
    later wave is a wave that runs passes with nothing to read — and because
    `soft` needs still order the work, the closure covers both kinds.
    """
    pool = set(universe) if universe is not None else set(BY_ID)
    out, stack = set(), list(ids)
    while stack:
        cid = stack.pop()
        if cid in out or cid not in BY_ID:
            continue
        out.add(cid)
        stack.extend(n for n in BY_ID[cid].needs if n in pool)
    return [c for c in out if c in pool]


def wave_leaks(ids, spine=None) -> list:
    """`[(component, need, need_wave)]` for every need scheduled after its user.

    Empty is the invariant. Asserted here rather than trusted, because a wave
    that is not need-closed does not fail — it quietly produces a thinner
    database, which is the one failure mode this archive cannot notice by
    looking at it.
    """
    by_id = {cid: w for w, group in waves(ids, spine) for cid in group}
    return [(cid, n, by_id[n])
            for cid, w in by_id.items()
            for n in BY_ID[cid].needs
            if n in by_id and by_id[n] > w]


def missing_dependencies(ids) -> dict:
    """Components required by the selection that are not in it.

    Selecting `narrate` without `describe` is a legitimate thing to want — the
    describe pass may already be done from a previous session — so this reports
    rather than refuses. The engine checks coverage for the real answer.
    """
    wanted = set(ids)
    out = {}
    for i in ids:
        gap = [n for n in BY_ID[i].needs if n not in wanted]
        if gap:
            out[i] = gap
    return out


# ══════════════════════════════════════════════════════════════════════════
# Cohorts — the rotation loop's plan
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Cohort:
    """One load of the GPU: a set of components that are resident together.

    The loop runs each component in the cohort over every pending video, uploads
    the shard, unloads everything, and packs the next cohort. Grouping matters
    because loading a 6 GB model takes forty seconds and doing it once per video
    instead of once per cohort would cost more time than the inference.
    """
    index: int = 0
    components: list = field(default_factory=list)
    vram_mb: int = 0
    cards: int = 1
    loads: list = field(default_factory=list)   # distinct weights to fetch

    def as_dict(self, videos: int = 0) -> dict:
        seconds = sum(BY_ID[c].seconds for c in self.components)
        return {
            "index": self.index,
            "components": list(self.components),
            "titles": [BY_ID[c].title for c in self.components],
            "vram_mb": self.vram_mb,
            "cards": self.cards,
            "loads": list(self.loads),
            "seconds_per_video": round(seconds, 1),
            "eta_seconds": round(seconds * videos, 1) if videos else None,
        }


def plan_cohorts(ids, vram_budget_mb: int, gpu_count: int = 1,
                 disk_budget_mb: int = 0) -> list:
    """Pack components into cohorts that fit the measured VRAM.

    First-fit over the dependency order, not best-fit, and the difference is
    deliberate. Best-fit would pack denser and reorder passes to do it, which
    breaks the property that matters more than density: a cohort's outputs are
    complete and uploaded before the next cohort starts, so a session killed at
    the twelve-hour wall loses at most one cohort of work.

    Components sharing `weights` cost VRAM once — three VLM passes on the same
    8B model are one 6 GB load, and packing them apart would triple both the
    download and the load time for no benefit at all.

    A component needing two cards gets a cohort to itself. Sharding a 38B model
    across both T4s leaves nothing to share with.
    """
    order = topo_sort(ids)
    cohorts, cur = [], Cohort(index=0)
    cur_loads: dict = {}

    def close():
        nonlocal cur, cur_loads
        if cur.components:
            cur.loads = list(cur_loads)
            cohorts.append(cur)
        cur = Cohort(index=len(cohorts))
        cur_loads = {}

    for cid in order:
        c = BY_ID[cid]

        if c.cards > gpu_count and c.device == "gpu":
            # Cannot run here at all. Skipped rather than dropped silently —
            # the caller reports it.
            continue

        if c.cards > 1:
            close()
            cur.components.append(cid)
            cur_loads[c.load_key] = c.vram_mb
            cur.vram_mb = c.vram_mb
            cur.cards = c.cards
            close()
            continue

        cost = 0 if c.load_key in cur_loads else c.vram_mb
        if c.device == "gpu" and cur.vram_mb + cost > vram_budget_mb:
            close()
            cost = c.vram_mb

        cur.components.append(cid)
        if c.device == "gpu":
            cur_loads[c.load_key] = c.vram_mb
            cur.vram_mb += cost

    close()
    return [c for c in cohorts if c.components]


def unrunnable(ids, res: dict) -> dict:
    """Components the measured machine cannot host, with the reason.

    Shown in the tab beside the plan rather than raised, because "this session
    got one GPU instead of two" is a normal Kaggle Tuesday and the answer is to
    run the other twenty-six passes, not to stop.

    `kaggle_ok=False` is reported here too, and before the GPU filter, because
    that flag is a statement about the whole machine rather than about VRAM — a
    CPU pass can be flagged as well. It is the plan's way of saying the same
    thing the coverage row will say per video: not broken, not here.
    """
    out = {}
    per_card = res.get("usable_vram_mb", 0)
    total = res.get("usable_vram_total_mb", 0)
    gpus = res.get("gpu_count", 0)
    kaggle = res.get("host") == "kaggle"
    for cid in ids:
        c = BY_ID.get(cid)
        if c is None:
            continue
        if kaggle and not c.kaggle_ok:
            out[cid] = "flagged as not runnable on Kaggle — held for another machine"
            continue
        if c.device != "gpu":
            continue
        if not gpus:
            out[cid] = "no GPU in this session"
        elif c.cards > gpus:
            out[cid] = f"needs {c.cards} cards, session has {gpus}"
        elif c.cards > 1 and c.vram_mb > total:
            out[cid] = f"needs {c.vram_mb} MB across cards, {total} MB usable"
        elif c.cards == 1 and c.vram_mb > per_card:
            out[cid] = f"needs {c.vram_mb} MB on one card, {per_card} MB usable"
    return out


def rows(selected=None) -> list:
    """The catalogue as the tab renders it: one dict per component."""
    sel = set(selected) if selected is not None else set(defaults())
    out = []
    for c in CATALOGUE:
        d = c.as_dict()
        d["selected"] = c.id in sel
        out.append(d)
    return out


def estimate(ids, videos: int, gpu_count: int = 1) -> dict:
    """Wall-clock estimate for a full sweep, before anything has been measured.

    Replaced by the coverage table's measured averages the moment real runs
    exist — this is only for the "what am I about to start" screen.
    """
    order = topo_sort(ids)
    per_video = sum(BY_ID[i].seconds for i in order)
    parallel = max(gpu_count, 1)
    return {
        "components": len(order),
        "videos": videos,
        "seconds_per_video": round(per_video, 1),
        "gpu_hours": round(per_video * videos / 3600.0, 1),
        "wall_hours": round(per_video * videos / 3600.0 / parallel, 1),
        "download_mb": sum({BY_ID[i].load_key: BY_ID[i].disk_mb
                            for i in order}.values()),
    }
