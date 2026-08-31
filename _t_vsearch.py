"""What frame search must do, pinned.

Reverse frame search — "find me more frames that look like this one" — needs no
model at all: the query vector is already in the database, so it is a cosine
against a matrix. That makes it the one visual mode that works on a laptop with
no torch installed, and the one most worth protecting.

The failure these assertions describe was measured, not imagined. `_coarse` used
to take the best 1,536 frames overall and group them by reel, which sounds like
"rank the reels" and is not: frames inside one reel are near-duplicates of each
other, so a matching reel contributes hundreds of nearly identical high scores
and crowds the list. On a thirty-reel archive of 32,302 frames the shortlist came
back with **nine** reels. The other twenty-one were never scored frame-by-frame,
so they could not be returned at any rank however well they matched.

Run with `python _t_vsearch.py` from the repo root. No database and no network —
the resident index is injected directly, because that is the state the ranking
functions actually read.
"""
import sys

import numpy as np

from atlas import vsearch

ok = 0


def eq(got, want, what):
    global ok
    assert got == want, f"{what}: got {got!r}, want {want!r}"
    ok += 1


def yes(cond, what):
    global ok
    assert cond, what
    ok += 1


def resident(clusters, stride=1, space="t"):
    """Install a resident index built from `{video_key: (n_frames, cosine)}`.

    `cosine` is how close that reel's centre sits to the query direction `e0`,
    and each reel's frames are jittered tightly around its own centre. That is
    what real footage looks like in an embedding space: a dense cluster per reel,
    not a scatter.

    The shape matters more than the numbers. The old shortlist failed precisely
    when one reel was both **large** and **closer than every other reel**, so its
    frames filled the global top-N and no other reel got in. A fixture whose
    reels all point the same way would let the small ones through and prove
    nothing, which is the mistake this docstring exists to stop someone
    repeating.
    """
    rng = np.random.default_rng(4)
    dim = 8
    vecs, ids, videos = [], [], []
    for ordv, (key, (n, cos)) in enumerate(clusters.items()):
        videos.append(key)
        centre = np.zeros(dim, dtype=np.float32)
        centre[0] = float(cos)
        centre[1] = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
        block = centre + rng.normal(0, 0.005, size=(n, dim)).astype(np.float32)
        vecs.append(block / np.linalg.norm(block, axis=1, keepdims=True))
        ids.append((np.int64(ordv) << 32) + np.arange(n, dtype=np.int64))
    with vsearch._LOCK:
        vsearch._RESIDENT[space] = {
            "vecs": np.concatenate(vecs).astype(np.float32),
            "ids": np.concatenate(ids), "dim": dim,
            "stride": stride, "videos": videos}
    return space, dim


def old_coarse(space, q, limit):
    """The shortlist as it was, so the assertions below can prove it was wrong.

    A regression test that passes against the code it replaces is not a test.
    """
    with vsearch._LOCK:
        res = vsearch._RESIDENT[space]
    sims = res["vecs"] @ q.astype(np.float32)
    take = min(len(sims), max(limit * 64, 512))
    top = (np.argpartition(-sims, take - 1)[:take] if len(sims) > take
           else np.arange(len(sims)))
    best, videos = {}, res["videos"]
    for i in top:
        ordv = int(res["ids"][i]) >> 32
        if ordv >= len(videos):
            continue
        key, s = videos[ordv], float(sims[i])
        if s > best.get(key, -2.0):
            best[key] = s
    return sorted(best.items(), key=lambda kv: -kv[1])[:limit]


# ── Every reel is scored, not only the ones owning a globally-top frame ──────
# One reel is 3,000 frames sitting nearest the query; eleven others are small and
# a little further away. Every one of the 3,000 outscores every frame of every
# small reel, so the best 1,536 frames overall are all the same reel's.
BIG = {"loud": (3000, 0.99)}
BIG.update({f"quiet{i}": (30, 0.90 - 0.02 * i) for i in range(1, 12)})
space, dim = resident(BIG)

q = np.zeros(dim, dtype=np.float32)
q[0] = 1.0

# First, the defect — pinned, so nobody restores the old shortlist believing it
# equivalent. This is the nine-reels-of-thirty failure at fixture scale.
was = old_coarse(space, q, 24)
eq(len(was), 1, "the old shortlist saw one reel of twelve")
eq(was[0][0], "loud", "the crowded one, which was the only one it could see")

ranked = vsearch._coarse(space, q, np, 24)
eq(len(ranked), 12, "every reel in the index is ranked, not just the crowded one")
eq(ranked[0][0], "loud", "and the reel with the closest frame still leads")
eq([k for k, _s in ranked][1:4], ["quiet1", "quiet2", "quiet3"],
   "the rest follow in order of their own best frame")
yes(all(a[1] >= b[1] for a, b in zip(ranked, ranked[1:])),
    "scores descend, so a caller may truncate the list and keep the best")

# The cap is still a cap.
eq(len(vsearch._coarse(space, q, np, 5)), 5, "limit is respected")

# A reel whose frames all point away is ranked last, not omitted — omission is a
# decision to make from a score, not one for the shortlist to make silently.
MIXED = {"near": (40, 1.0), "far": (40, 0.10)}
space2, dim2 = resident(MIXED, space="t2")
q2 = np.zeros(dim2, dtype=np.float32)
q2[0] = 1.0
r2 = vsearch._coarse(space2, q2, np, 24)
eq([k for k, _s in r2], ["near", "far"], "both reels ranked, best first")
yes(r2[0][1] > 0.9 > r2[1][1], "and the ordering is by cosine")

# ── The unstrided fast path ─────────────────────────────────────────────────
# `_exact` exists to make a strided index give an unstrided answer. With stride
# 1 the resident matrix *is* the full-rate data, so re-reading vec_payload
# fetches vectors already in RAM. That cost was 700 ms of a 700 ms query.
hits = vsearch._resident_hits(space2, q2, np, 5, "")
yes(hits is not None, "stride 1 answers from the resident matrix")
eq(len(hits), 5, "and returns the frames asked for")
eq(hits[0][0], "near", "best frame first")
yes(all(hits[i][2] >= hits[i + 1][2] for i in range(len(hits) - 1)),
    "hits come back sorted by score, not merely partitioned")
eq(sorted({k for k, _i, _s in hits}), ["near"],
   "the top five frames all come from the reel that matches")

# `None` and `[]` must stay distinguishable: one means "not my job", the other
# means "nothing matched". A caller that conflated them would fall through to
# `_exact` on an empty result and search twice.
space3, _d3 = resident(MIXED, stride=4, space="t3")
yes(vsearch._resident_hits(space3, q2, np, 5, "") is None,
    "a strided index declines, so _exact keeps the job it was written for")
yes(vsearch._resident_hits("no-such-space", q2, np, 5, "") is None,
    "an absent space declines too")

# Excluding a reel drops its frames before the top-k is taken, so asking for
# five still returns five — the old path over-fetched and filtered after.
ex = vsearch._resident_hits(space2, q2, np, 5, "near")
eq(sorted({k for k, _i, _s in ex}), ["far"], "the excluded reel is gone")
eq(len(ex), 5, "and the limit is still filled from what remains")
eq(vsearch._resident_hits(space2, q2, np, 5, "nobody") and True, True,
   "excluding a reel that is not indexed is not an error")

# A frame is its own best match. If this breaks, the vectors are not normalised
# and every cosine below it is a dot product of arbitrary magnitudes.
with vsearch._LOCK:
    res = vsearch._RESIDENT[space2]
own = res["vecs"][0].copy()
best = vsearch._resident_hits(space2, own, np, 1, "")
eq((best[0][0], best[0][1]), ("near", 0), "a frame retrieves itself first")
yes(best[0][2] > 0.999, "at a cosine of one")

# ── One hit per moment ──────────────────────────────────────────────────────
# Correct ranking and useful ranking are not the same list. Frames next to each
# other in one reel are the same photograph, so the honest top twenty-four is
# twenty-four views of one second. Measured on the live archive before this:
# asking for twenty-four frames like a given frame returned twenty-four frames
# from *one* reel spanning three to ten distinct seconds, and because the poster
# cache is keyed per second, one file was served for ten of the tiles.
#
# `fps` is 30 here, so the 1.5 s default gap is 45 frames.
FPS = {"a": 30.0, "b": 30.0}
run = [("a", i, 0.9 - i * 0.0001) for i in range(200)]

eq(len({i for _k, i, _s in run}), 200, "the fixture is one unbroken run")
kept = vsearch._spread(run, FPS, 24)
eq(len(kept), 5, "a 200-frame run at 30 fps yields five moments, not two dozen")
eq([i for _k, i, _s in kept], [0, 45, 90, 135, 180],
   "each kept frame is a gap clear of the last one kept")
yes(all(a[2] >= b[2] for a, b in zip(kept, kept[1:])),
    "and they stay in score order, so the page is still ranked")

# Greedy from the top means the frame kept for a moment is that moment's best,
# never a worse neighbour that happened to come first by index.
humped = [("a", 60, 0.99), ("a", 61, 0.98), ("a", 62, 0.97)]
eq(vsearch._spread(humped, FPS, 24), [("a", 60, 0.99)],
   "three views of one instant collapse to the best of them")

# The per-reel cap is what stops one reel owning the page even when its moments
# are genuinely distinct — the same principle as `_coarse`, one level down.
wide = [("a", i * 100, 0.9 - i * 0.01) for i in range(20)]
eq(len(vsearch._spread(wide, FPS, 24)), 6,
   "one reel contributes at most its share, however many moments it has")
eq(len(vsearch._spread(wide, FPS, 24, per_video=0)), 20,
   "and the cap can be lifted")

# Two reels fill the page independently: a gap is a fact about one reel's own
# timeline, so identical frame indices in different reels are different moments.
mixed = []
for i in range(200):
    mixed.append(("a", i, 0.9 - i * 0.0001))
    mixed.append(("b", i, 0.8 - i * 0.0001))
both = vsearch._spread(mixed, FPS, 24)
eq(sorted({k for k, _i, _s in both}), ["a", "b"], "both reels are represented")
eq(len(both), 10, "five moments each, the cap not yet reached")

# A reel with no frame rate on record still gets spread, on the assumed rate.
# `_fps` refuses to guess 30 for a *timestamp* because a wrong `t` seeks the
# player to the wrong moment; a wrong gap only spaces results differently.
nofps = vsearch._spread(run, {}, 24)
eq([i for _k, i, _s in nofps], [0, 45, 90, 135, 180],
   "an unknown frame rate falls back to the assumed one rather than giving up")

# Turning both rules off must return the ranking untouched, so that a caller
# wanting raw frame order — a diagnostic, a test — can still get it.
eq(vsearch._spread(run, FPS, 7, gap_s=0, per_video=0), run[:7],
   "with the spread disabled the list passes straight through")
eq(vsearch._spread([], FPS, 24), [], "an empty ranking spreads to nothing")

with vsearch._LOCK:
    for s in ("t", "t2", "t3"):
        vsearch._RESIDENT.pop(s, None)

print(f"ok — {ok} assertions")
sys.exit(0)
