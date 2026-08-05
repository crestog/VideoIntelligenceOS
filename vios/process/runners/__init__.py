"""
vios.process.runners — component id → the function that runs it.

The table below is the only place a registry entry is connected to code. Keeping
it explicit rather than clever — no naming convention, no decorator scanning —
means the answer to "what actually runs when I tick `narrate`?" is one line you
can read, and a component with no runner is caught by `missing()` instead of by
an AttributeError twelve hours into a session.

Resolution is lazy on purpose. The web process imports this module to render the
engine tab and has no torch, no cv2 and no ffmpeg; it needs the names and the
descriptions, not the implementations. Nothing heavy is imported until a pass is
about to run.
"""

from __future__ import annotations

import importlib

from .base import Emission, Job, ModelCache, SkipPass  # noqa: F401 — re-export

# component id → (module in this package, function name)
RUNNERS: dict = {
    # stage 0 — structure
    "probe":          ("structure", "probe"),
    "artifacts":      ("structure", "artifacts"),
    "shots":          ("structure", "shots"),
    "keyframes":      ("structure", "keyframes"),

    # stage 1 — signal: arithmetic, no model
    "caption":        ("signal", "caption"),
    "cuts":           ("signal", "cuts"),
    "colour":         ("signal", "colour"),
    "motion":         ("signal", "motion"),
    "loudness":       ("signal", "loudness"),
    "music":          ("signal", "music"),

    # stage 2 — perception
    "transcribe":     ("audio", "transcribe"),
    "transcribe-alt": ("audio", "transcribe_alt"),
    "diarize":        ("audio", "diarize"),
    "audio-tag":      ("audio", "audio_tag"),
    "ocr":            ("vision", "ocr"),
    "ocr-alt":        ("vision", "ocr_alt"),
    "detect":         ("vision", "detect"),
    "faces":          ("vision", "faces"),
    "depth":          ("vision", "depth"),
    "visual-embed":   ("vision", "visual_embed"),
    "tag":            ("signal", "tag"),
    "aesthetic":      ("vision", "aesthetic"),

    # stage 3 — language
    "describe":       ("language", "describe"),
    "narrate":        ("language", "narrate"),
    "style-read":     ("language", "style_read"),
    "keyphrase":      ("signal", "keyphrase"),
    "concepts":       ("language", "concepts"),
    "text-embed":     ("language", "text_embed"),
    "hook":           ("signal", "hook"),
    "narrate-deep":   ("language", "narrate_deep"),
}

_cache: dict = {}


def get(component_id: str):
    """The runner for a component, imported on first use.

    Raises KeyError for an unknown id, which is the engine's signal that the
    registry and this table have drifted apart — a condition worth stopping
    for, not working around.
    """
    if component_id in _cache:
        return _cache[component_id]
    module_name, func_name = RUNNERS[component_id]
    module = importlib.import_module(f"{__name__}.{module_name}")
    fn = getattr(module, func_name)
    _cache[component_id] = fn
    return fn


def available() -> set:
    return set(RUNNERS)


def missing() -> list:
    """Registry components with no runner. Should always be empty."""
    from .. import registry  # noqa: PLC0415 — avoids a circular import
    return sorted(c.id for c in registry.CATALOGUE if c.id not in RUNNERS)


def orphaned() -> list:
    """Runners with no registry component. Should always be empty."""
    from .. import registry  # noqa: PLC0415
    return sorted(cid for cid in RUNNERS if cid not in registry.BY_ID)
