"""What the claim encoder must do, pinned.

Search accuracy is the thing this project is for, and it is decided by four
functions: `phrase.written` (what a measurement says in words), `phrase.is_absence`
(what a detector's shrug looks like), `index._labels` (which cells are label lists
rather than language) and `index.build_facts` (how long a fact held for). Every one
of them was written in response to a measured failure — `orange colour` matching
all thirty reels in the archive — so the failures are what the assertions describe.

Run with `python _t_phrase.py` from the repo root. No database and no network.
"""
import sqlite3
import sys

from atlas import index, phrase, reflect

ok = 0


def eq(got, want, what):
    global ok
    assert got == want, f"{what}: got {got!r}, want {want!r}"
    ok += 1


def yes(cond, what):
    global ok
    assert cond, what
    ok += 1


# ── The fact is named, not just its value ────────────────────────────────
# `orange` alone cannot answer "orange colour" — the query's second word has
# nothing to match. This is the whole reason the module exists.
eq(phrase.written("dominant_colour", "orange"),
   ("fact", "the dominant colour is orange"), "colour names itself")
eq(phrase.written("shot_scale", "medium"), ("fact", "medium shot"),
   "a bare `medium` is unsearchable")
eq(phrase.written("camera_move", "pan right"), ("fact", "camera pan right"),
   "camera move")
eq(phrase.written("face_scale", "close-up"),
   ("fact", "a close-up view of a face"), "face scale reads as English")
eq(phrase.written("brightness", "dark"), ("fact", "dark lighting"), "brightness")
eq(phrase.written("temperature", "warm"), ("fact", "warm tones"), "temperature")

# An ISO code is not a word anybody types.
eq(phrase.written("language", "hi"), ("fact", "spoken in Hindi"), "language code")
eq(phrase.written("language", "xx"), ("fact", "spoken in xx"), "unknown code")

# ── Refused from the text index ──────────────────────────────────────────
# Refused means refused from *text*. Every one of these stays in `claim`.
eq(phrase.written("freeze", "freeze"), None, "a constant discriminates nothing")
eq(phrase.written("palette", '[{"hex": "#000002", "share": 0.4}]'), None,
   "hex is never typed into a search box")
eq(phrase.written("likes", "328095"), None, "a number wearing a value column")
eq(phrase.written("declared_duration", "7.699999809265137"), None, "raw float")
eq(phrase.written("face_track", "person 2"), None, "an identity local to a video")
eq(phrase.written("object_count", "person"), None, "already indexed by `object`")
eq(phrase.written("dominant_colour", "#EBD6BF"), None,
   "a known kind holding a blob is still refused")
eq(phrase.written("dominant_colour", ""), None, "empty")
eq(phrase.written("dominant_colour", None), None, "null")

# ── A reading is not a description ───────────────────────────────────────
# These read like English and cannot be typed. Indexed, each donates the word a
# query needed to every video in the archive — one useless passage per reel is
# how "close-up shot" came to match every reel that had a shot count.
eq(phrase.written("tempo", "103 BPM"), None, "a reading leads with its number")
eq(phrase.written("asl", "24.08s average shot length"), None, "shot length")
eq(phrase.written("shot_count", "38 shots"), None, "shot count")
eq(phrase.written("speech_rate", "233 words per minute"), None, "speech rate")
eq(phrase.written("hook_cuts", "1 shots in the first 3s"), None, "hook cuts")
# The guard has to run *before* the prose fallback, not after it: `cut_on_beat`
# is six words long, so length alone would send it straight past the check that
# exists to stop it. This is the assertion that pins the ordering.
eq(phrase.written("cut_on_beat", "50% of cuts land on a beat"), None,
   "a long reading is still a reading")
yes(len("50% of cuts land on a beat".split()) > phrase._LABEL_WORDS,
    "and it is long enough to have taken the prose path")

# ── Language stays language ──────────────────────────────────────────────
eq(phrase.written("transcript", "और doctors इन्हें replace कर देंगे"),
   ("prose", "और doctors इन्हें replace कर देंगे"), "a transcript is prose")
eq(phrase.written("object", "potted plant"), ("fact", "potted plant"),
   "an object is already a noun")
# A declared prose kind is trusted with its own shape. Somebody said the number.
eq(phrase.written("transcript", "10 out of 10 uncomfortable things to do alone"),
   ("prose", "10 out of 10 uncomfortable things to do alone"),
   "speech may open with a number")
eq(phrase.written("agreement", "95% agreement with the primary transcript"),
   ("prose", "95% agreement with the primary transcript"),
   "provenance is written as a sentence already")

# ── A guess below the floor is not a claim ───────────────────────────────
eq(phrase.written("object", "bicycle", 0.0003), None,
   "a 0.03% object is a false answer, and a false answer beats no answer only "
   "for the person who cannot tell")
eq(phrase.written("object", "bicycle", 0.51), ("fact", "bicycle"), "kept above")
eq(phrase.written("sound_event", "laughter", 0.57), ("fact", "laughter"),
   "the audio channel never exceeds 0.6, so the floor cannot be 0.5")

# ── An unknown kind is decided by shape, not refused ─────────────────────
# A pass added next month has to be searchable without editing phrase.py.
eq(phrase.written("mood", "tense"), ("fact", "mood tense"), "unknown short label")
eq(phrase.written("mood", "a long sentence about the mood of this shot"),
   ("prose", "a long sentence about the mood of this shot"), "unknown long value")
eq(phrase.written("whatever", "42"), None, "unknown numeric")
eq(phrase.written("whatever", '{"a": 1}'), None, "unknown structure")

# ── A detector's shrug is not evidence ──────────────────────────────────
# 1,076 of this archive's 2,399 frame notes say nothing was found. Indexed, they
# put the words `objects` and `text` on every reel proven to have neither.
for s in ("no salient objects or text detected", "No salient objects or text "
          "detected.", "none", "N/A", "unknown", "null", "nothing detected",
          "no faces found", "not applicable", "zero objects detected"):
    yes(phrase.is_absence(s), f"absence: {s!r}")
# Narrow on purpose. A negation with no looking in it is somebody talking, and
# this archive's transcripts really do contain the line `NO!`.
for s in ("NO!", "no", "nothing", "no way that's real", "no, I didn't do that",
          "none of us knew what to say", "nothing was found in the drawer but a "
          "key", "notebook on a desk", "unknown caller keeps ringing me"):
    yes(not phrase.is_absence(s), f"not absence: {s!r}")
eq(phrase.written("screen_text", "no text detected"), None,
   "a sentinel is refused even from a prose kind")

# ── A label list is not a paragraph ─────────────────────────────────────
# The CV worker wrote one row per frame. Merged as prose they become
# `person person person 3× cow, person 3× cow, person 5× cow` — and a search for
# `dog` then ranks a reel of cows first, on term frequency alone.
eq(index._labels('[{"label": "dog", "conf": 0.75}, {"label": "bed", "conf": 0.7}]'),
   [("dog", 0.75), ("bed", 0.7)], "json labels keep their confidence")
eq(index._labels('["person","bicycle"]'), [("person", None), ("bicycle", None)],
   "a bare json list")
eq(index._labels("3× cow, person"), [("cow", None), ("person", None)],
   "a count marker is the detector counting, not a word")
eq(index._labels("2× chair, person, dining table, knife"),
   [("chair", None), ("person", None), ("dining table", None), ("knife", None)],
   "four labels, four facts")
eq(index._labels("cow"), [("cow", None)], "one label is still a label")
# Function words are the one thing a label list never has, and the only cheap way
# to tell `3× cow, person` from a sentence about a cow.
eq(index._labels("The video starts with a cow lying on a bed at 11:59 pm."), [],
   "a sentence is not a label list")
eq(index._labels("yeah, exactly, that"), [], "nor is hesitant speech")
eq(index._labels('[{"label": "a whole sentence pretending to be a label here"}]'),
   [], "nor a sentence inside a json list")
eq(index._labels(None), [], "nor a null")

# ── …but the column decides, because the cell cannot ────────────────────
# `조용히` is a word somebody said and `cow` is a thing a model saw, and they are
# the same shape. What separates them is the company they keep.
c = sqlite3.connect(":memory:")
c.execute("CREATE TABLE claim (video_key TEXT, t0 REAL, t1 REAL, "
          "channel TEXT, kind TEXT, value TEXT, confidence REAL)")
c.executemany("INSERT INTO claim VALUES (?,?,?,?,?,?,?)",
              [("k1", 0.0, 2.0, "style", "dominant_colour", "orange", 0.8),
               ("k1", 0.0, 2.0, "style", "camera_move", "pan right", 0.9),
               ("k1", 0.0, 2.0, "speech", "transcript", "a whole sentence", 1.0)])
c.execute("CREATE TABLE frame_notes (video_key TEXT, t0 REAL, description TEXT)")
c.executemany("INSERT INTO frame_notes VALUES ('k1', 0.0, ?)",
              [(v,) for v in ["3× cow, person"] * 15 + ["cow"] * 15])
c.execute("CREATE TABLE transcripts (video_key TEXT, t0 REAL, text TEXT)")
c.executemany("INSERT INTO transcripts VALUES ('k1', 0.0, ?)",
              [(v,) for v in ["a whole spoken sentence with function words in it"]
               * 25 + ["조용히", "滝くん", "すみません"]])
specs = {(s["table"], s["source"]): s for s in reflect.text_sources(c)}
yes(index._label_column(c, specs[("frame_notes", "visual")]),
    "a column of labels is a label column")
yes(not index._label_column(c, specs[("transcripts", "speech")]),
    "three short foreign words do not make a transcript into labels")

# ── reflect finds the kind and confidence columns, and only where they are ──
eq(specs[("claim", "style")]["kind"], "kind", "claim names its measurements")
eq(specs[("claim", "style")]["conf"], "confidence", "and how sure it was")
eq(specs[("transcripts", "speech")]["kind"], None, "a prose table does not")
# The spec must actually yield the six columns it promised, in that order.
row = c.execute(specs[("claim", "style")]["sql"]).fetchone()
eq(len(row), 6, "a kind spec yields six columns")
yes(isinstance(row[5], float) and 0.0 <= row[5] <= 1.0,
    "the sixth is a confidence")
eq(len(c.execute(specs[("transcripts", "speech")]["sql"]).fetchone()), 4,
   "everything else still yields four")

# ── A fact holds for as long as it holds, and no longer ──────────────────
# Per-shot claims tile the timeline, so consecutive shots showing the same thing
# have a gap near zero. Two orange shots at opposite ends of a reel do not.
runs = index.build_facts([
    (0.0, 2.0, "orange", 0.8), (2.0, 4.0, "orange", 0.8),   # two shots in a row
    (30.0, 32.0, "orange", 0.8),                            # and one much later
])
eq(len(runs), 1, "one moment per distinct fact")
t0, t1, text, span, conf = runs[0]
eq((t0, t1), (0.0, 4.0), "the longest run is the one emitted")
eq(round(span, 3), 6.0, "span counts the seconds it was true, not the gap")

# The prose gap would have glued these into one 32-second run and called the
# whole reel orange. That is the defect this constant fixes.
yes(index.FACT_GAP_S < index.MERGE_GAP_S, "a fact is not a subtitle line")

# Overlapping observations of the same fact are one span, not two.
span = index.build_facts([(0.0, 5.0, "person", None),
                          (2.0, 7.0, "person", None)])[0][3]
eq(round(span, 3), 7.0, "the union, not the sum")

# The best sighting settles it. The question a rank answers is whether this reel
# is a real answer, and averaging would punish a fact for also having been
# guessed at weakly somewhere else in the same reel.
conf = index.build_facts([(0.0, 2.0, "laughter", 0.12),
                          (4.0, 6.0, "laughter", 0.59)])[0][4]
eq(conf, 0.59, "the clearest sighting, not the average")

# Two different facts at the same second never merge — the string that made
# `orange colour` match everything was thirty-six of these in a row.
facts = index.build_facts([(0.0, 2.0, "camera pan right", 0.9),
                           (0.0, 2.0, "the dominant colour is orange", 0.8)])
eq(len(facts), 2, "one fact per moment")
eq(sorted(f[2] for f in facts),
   ["camera pan right", "the dominant colour is orange"], "kept apart")

# A rollup with no timing keeps none rather than being given a false position.
eq(index.build_facts([(None, None, "music-led", 1.0)]),
   [(None, None, "music-led", 0.0, 1.0)],
   "an untimed fact is about the whole reel")

# ── Prominence separates a reel that is orange from one with an orange shot ──
whole = index._prominence(16.0, 16.2)
flicker = index._prominence(2.0, 60.0)
yes(whole > flicker * 1.3, f"prominence too flat: {whole} vs {flicker}")
yes(1.0 <= flicker < whole <= 1.6, f"out of range: {flicker}, {whole}")
eq(index._prominence(0.0, 60.0), 1.0, "no span, no boost")

# ── Certainty separates a measurement from a guess ──────────────────────
eq(index._certainty(None), 1.0, "a pass that reports no confidence is not unsure")
eq(index._certainty(1.0), 1.0, "certain")
eq(index._certainty(0.0), 0.5, "never zero: a stored claim that ranks at zero "
                               "may as well not have been stored")
yes(index._certainty(0.8) > index._certainty(0.57),
    "counted pixels outrank a zero-shot tag")
yes(index._certainty(0.6) > index._certainty(0.0412),
    "and within the audio channel, the confident tag outranks the desperate one")

print(f"ok — {ok} assertions", file=sys.stderr)
