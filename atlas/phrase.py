"""
The written form of a measured fact.

`claim` is Atlas's own evidence table, and every row in it is a *measurement*:
a kind, a value, a number, a confidence, the shot it belongs to. Search does
not read measurements — it reads language. This module is the translation, and
it exists because the naive translation is what broke colour search.

The naive translation is to hand `claim.value` to the text index as though it
were prose. Do that and a style passage comes out reading

    handheld freeze pan right yellow freeze pan left grey orange green freeze
    orange green grey orange pull out grey #000100, #596135, #84826E …

which is thirty-six unrelated measurements glued into one string under one
coarse timestamp. Three separate failures live in that line.

**A word that is not the fact.** `orange` alone never says *what* is orange.
Somebody searching "orange colour" is asking about the dominant colour of a
shot, and the passage does not contain the word `colour`, so the second half of
the query can only match noise. Written out — `the dominant colour is orange` —
question and answer share vocabulary, which is the entire mechanism of both BM25
and a dense encoder.

**Tokens that are not words.** `#596135`, `7.699999809265137`, `person 2`. None
of these is ever typed into a search box: they cannot match, they cannot fail to
match, and they take up room. Every one of them lowers the share of the passage
the real fact occupies, which is precisely what BM25's length normalisation
punishes.

**Facts that say nothing.** `freeze` is 5,334 of this archive's 27,346 claims
and has exactly one distinct value. Indexing it adds one token to a fifth of the
corpus: it discriminates nothing and dilutes everything.

So every kind is sorted into one of three fates — refused, passed through as the
language it already is, or written into a statement. Refusing is refusal *from
the text index only*; the measurement stays in `claim`, where the Data tab, the
graph and the numeric filters read it. Whether a number is worth keeping and
whether it is worth matching on are different questions.

A kind Atlas has never seen is decided by rule rather than by name: structure
and bare numbers are refused, a short label is written together with its kind,
and a long value is prose. That is the same bargain `reflect` makes with tables
it has never seen — a claim kind added upstream next month becomes searchable on
the next build with no code change here.
"""
import re

# ── Refused ───────────────────────────────────────────────────────────────
REFUSED = frozenset({
    # Structure. A JSON array of beat times is a fact about the audio, not a
    # sentence about it; flattened into a passage it is a hundred digits.
    "palette", "text_region", "beat_grid", "loudness_curve", "words",
    # A constant. One distinct value across thousands of rows, so its presence
    # in a passage is indistinguishable from its absence.
    "freeze",
    # A number wearing a value column. `likes: 328095` is worth ranking on and
    # worthless to match on.
    "likes", "comments", "comments_captured", "declared_duration",
    "caption_length", "caption_words",
    # Said twice. `object_count` and `screen_share` carry an object's *name* as
    # their value and the count or share in `num`; the name is already indexed
    # by `object`, and a second copy only inflates its term frequency.
    "object_count", "screen_share",
    # An identity local to one video. `person 2` is the second face this tracker
    # happened to see, not a person anybody can search for. What a viewer
    # remembers about a face is its size on screen, which `face_scale` carries.
    "face_track",
})

# ── Already language ──────────────────────────────────────────────────────
# Indexed verbatim and merged as prose, because a transcript line's neighbours
# belong with it: half a sentence in one row and half in the next is one thing
# somebody said.
PROSE = frozenset({
    "transcript", "segment", "caption", "hashtag", "mention", "uploader",
    "screen_text", "text", "keyphrase",
    "hook_opening_line", "hook_words",
    # Provenance, already written as sentences and one per video.
    "agreement", "contested", "language_uncertain",
})

# ── Written ───────────────────────────────────────────────────────────────
# `{v}` is the measured value. The template's whole job is to put the question
# into the passage beside the answer: `medium` is unsearchable, `medium shot` is
# what a person types.
WRITTEN = {
    "dominant_colour": "the dominant colour is {v}",
    "camera_move":     "camera {v}",
    "shot_scale":      "{v} shot",
    "face_scale":      "a {v} view of a face",
    "brightness":      "{v} lighting",
    "saturation":      "{v} colours",
    "temperature":     "{v} tones",
    "sharpness":       "{v} focus",
    "rhythm":          "{v} cutting rhythm",
    "acceleration":    "the pace {v}",
    "key":             "in the key of {v}",
    "hook_form":       "the opening is a {v}",
    "language":        "spoken in {v}",

    # Values that are already a phrase — `near-silent`, `music-led`,
    # `well exposed`. A template here would only add a word nobody's query
    # contains.
    "object":          "{v}",
    "sound_event":     "{v}",
    "silence":         "{v}",
    "exposure":        "{v}",
    "music_presence":  "{v}",
}

# A transcript's language arrives as ISO 639-1 because that is what the decoder
# emits. Nobody searches for `hi`; they search for Hindi.
LANGUAGES = {
    "en": "English", "hi": "Hindi", "ja": "Japanese", "ko": "Korean",
    "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic", "zh": "Chinese", "it": "Italian",
    "bn": "Bengali", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "ur": "Urdu", "id": "Indonesian",
    "tr": "Turkish", "vi": "Vietnamese", "th": "Thai", "nl": "Dutch",
    "pl": "Polish", "uk": "Ukrainian", "fa": "Persian", "ml": "Malayalam",
    "kn": "Kannada", "ne": "Nepali", "he": "Hebrew", "sv": "Swedish",
}

# The widest thing in this archive that is still a label rather than a sentence
# is four words (`well exposed`, `0 of ? held`, `2.5 cuts per minute`). Longer
# than that and an unknown kind is treated as prose.
_LABEL_WORDS = 4

# Below this confidence the observer is not making a claim, it is declining to.
# `object` bottoms out at 0.0003 in this archive — 204 rows under the floor, and
# each one would otherwise become an indexed fact reading `bicycle` about a reel
# with no bicycle in it. A false answer is worse than a missing one, because the
# person reading it has no way to tell.
#
# Deliberately a floor and not a threshold: everything above it is kept and
# ranked by how sure it was, because the alternative — a hard cut at a defensible
# 0.5 — throws away the whole audio channel, which never exceeds 0.6.
FLOOR = 0.05


def _is_structure(value: str) -> bool:
    """A machine's punctuation at the front — JSON, or a hex colour."""
    return value[:1] in ("{", "[", "#", "<")


def _is_number(value: str) -> bool:
    """No word in it at all: `328095`, `7.6999998`, `-18.3`, `41%`, `1,024`."""
    try:
        float(value.replace(",", "").rstrip("%").strip())
    except ValueError:
        return False
    return True


def _leads_with_number(value: str) -> bool:
    """A reading, not a description: `103 BPM`, `45.0 cuts per minute`, `-18.3 LUFS`.

    These read like English and are not. Nobody types "24.08s average shot
    length" into a search box, so the sentence can never be matched — but its
    words can, and `shots`, `cuts`, `per`, `minute`, `of`, `the` then appear on
    every video in the archive. That is how "close-up shot" came to match every
    reel with a shot count: one useless passage per video, each donating the word
    the query needed.

    Refused from the text index, kept in `claim` where `num` makes them
    filterable and sortable — which is the only way anybody would use them.
    """
    head = value.lstrip("-+~<>≈ ")[:1]
    return head.isdigit()


# A statement that nothing was found. The frame captioner writes "no salient
# objects or text detected" when it has nothing to say, and it says it on 1,076
# of this archive's 2,399 frame notes — so nearly half the visual channel would
# otherwise be a negation. Indexed, it is worse than absent: it puts the words
# `objects` and `text` on every reel that has neither, so searching for text on
# screen returns the reels proven to have none.
#
# Deliberately narrow, in two shapes and no third. Either the cell *names the
# act of looking* — `no salient objects or text detected` — or it is one of the
# handful of words a field carries when it is standing in for empty. A bare
# negation is neither: this archive's transcripts contain the line `NO!`, and
# somebody shouting is content.
_ABSENT = re.compile(
    r"^\W*(?:"
    r"(?:no|none|nothing|not|zero)\b[\w\s,'-]{0,60}?"
    r"\b(?:detect|found|present|visible|identif|salient|available|recognis|"
    r"recogniz|applicable)\w*"
    r"|(?:none|n/?a|null|nil|unknown|undefined|empty|unavailable)"
    r")\s*[.!]?\s*$",
    re.I)


def is_absence(text) -> bool:
    """Is this cell a detector reporting that it found nothing?

    True for `no salient objects or text detected`, `none`, `n/a`, `unknown`.
    False for `no way that's real` — a negation with no looking in it is speech.
    """
    s = "" if text is None else str(text).strip()
    if not s or len(s) > 120:
        return False
    return bool(_ABSENT.match(s))


def written(kind, value, confidence=None):
    """`(form, text)` for one claim, or `None` to keep it out of the text index.

    `form` is `'fact'` for a written statement and `'prose'` for language that
    arrived as language. The caller shapes the two differently and that is the
    point of returning it: prose merges with its neighbours into a passage, a
    fact stands alone and collapses across the run of shots that repeat it.
    """
    kind = (kind or "").strip().lower()
    value = "" if value is None else str(value).strip()
    if not value or kind in REFUSED:
        return None
    if confidence is not None and confidence < FLOOR:
        return None
    if kind in PROSE:
        # A named prose kind is trusted with its own shape: a transcript line is
        # allowed to open with a number, because somebody said the number.
        return None if is_absence(value) else ("prose", value)

    if kind == "language":
        value = LANGUAGES.get(value.lower(), value)

    # Before the prose fallback, not after it. `cut_on_beat` holds
    # `50% of cuts land on a beat` — six words, so length alone would send it
    # down the prose path and straight past the guard that exists to stop it.
    # Every kind that is not declared prose gets its shape checked first.
    if (_is_structure(value) or _is_number(value) or _leads_with_number(value)
            or is_absence(value)):
        return None

    template = WRITTEN.get(kind)
    if template is None:
        # An unrecognised kind, decided by shape. Refusing it outright would
        # make every future pass invisible to search until someone edited this
        # file; indexing it raw is what this module exists to stop.
        if len(value.split()) > _LABEL_WORDS:
            return "prose", value
        template = kind.replace("_", " ").strip() + " {v}"
    return "fact", " ".join(template.format(v=value).split())
