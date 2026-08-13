"""
Moment search.

*"I simply type natural language query and it finds most relevant and accurate
videos for me... make it as most accurate as possible... and I want the search
to be very very extremely fast."*

Those two goals pull in opposite directions everywhere except here, and the
design is the resolution.

## Why hybrid, and not just one of them

A dense encoder understands that "someone cooking pasta" and "a man boiling
spaghetti" are the same thing, and is hopeless at "@nikocado" or "iPhone 15
Pro" — rare tokens get averaged into nothing. BM25 is the reverse: exact on
names, numbers and jargon, blind to paraphrase. Published comparisons on this
exact trade-off put dense-only around 78% recall@10 and the fusion of both
around 91% on the same corpus. Running both and fusing is the single largest
accuracy win available, so both always run.

## Why Reciprocal Rank Fusion and not score blending

BM25 scores are unbounded and corpus-dependent; cosine similarities sit in
[-1, 1]. Normalising them onto a common scale requires knowing each retriever's
score distribution, which changes per query. RRF sidesteps it by discarding the
scores and keeping only the ranks:

    score(d) = Σ  1 / (k + rank_r(d))       k = 60

A document ranked 1st by one retriever and 40th by the other beats one ranked
15th by both — which is the behaviour you want, because a strong signal from
either kind of evidence is worth more than being vaguely plausible to both.
k=60 is from the original paper and is not sensitive; it exists to stop rank 1
from dominating everything below it.

## Why exhaustive vector search, and no vector database

The corpus is a few hundred thousand passages at 384 dimensions. As one float32
matrix that is well under a gigabyte, resident in RAM, and a query is a single
`(N,384) @ (384,)` matmul — memory-bandwidth-bound, a few milliseconds, and
*exact*. An ANN index would add a service, a build step, a tuning knob and an
approximation, to make a fast thing slightly faster. At ten million passages
that trade flips; at this size it plainly does not.

## Why results are videos, not passages

A reel with six matching moments should appear once, with six moments attached,
not six times. Passages are scored individually, then grouped by video, and the
video's score is its best moment plus a damped contribution from the rest —
so corroborating evidence helps, but a video cannot win on volume alone.
"""

import math
import re
import sqlite3
import threading
import time
from collections import OrderedDict

from . import config, index, media
from .tgchannel import log

_VEC_LOCK = threading.RLock()
_VECTORS = None          # (N, D) float32, L2-normalised
_VEC_IDS = None          # (N,) int64, moment ids aligned to _VECTORS
_VEC_POS = None          # {moment_id: row} for the reverse lookup

_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
# VECTOR RESIDENCY
# ══════════════════════════════════════════════════════════════════════════
def reload_vectors(expect: str = "") -> bool:
    """Load the flat vector file into RAM. Called after the indexer writes it.

    np.fromfile rather than np.load: the file is a bare float32 dump with its
    shape recorded in a sidecar, which avoids pickle and lets the indexer write
    it with `.tofile()` in one call.

    `expect` is the index generation the caller believes `moments` is on. The
    vectors are keyed by `moments.id`, which every rebuild reassigns, so a file
    from an older generation would load cleanly and then point each hit at the
    wrong passage. A definite disagreement — both sides naming a generation, and
    naming different ones — is refused. Either side being silent is accepted:
    an archive built before this stamp existed should not lose dense search on
    upgrade, and its next rebuild stamps both.
    """
    global _VECTORS, _VEC_IDS, _VEC_POS
    try:
        import numpy as np
    except ImportError:
        return False

    meta = index.vector_state()
    if not meta or not meta.get("count"):
        return False
    have_id = str(meta.get("build_id") or "")
    if expect and have_id and have_id != expect:
        log(f"dense index ignored — built for index {have_id}, this one is "
            f"{expect}; it reloads after the next build")
        return False
    try:
        dim = int(meta.get("dim") or config.EMBED_DIM)
        vecs = np.fromfile(config.VECTOR_PATH, dtype=np.float32)
        ids = np.fromfile(config.VECTOR_PATH + ".ids", dtype=np.int64)
        if dim <= 0 or vecs.size % dim:
            log(f"vector file is not a multiple of {dim} floats — ignoring it")
            return False
        vecs = vecs.reshape(-1, dim)
        if len(ids) != len(vecs):
            log(f"vector/id length mismatch ({len(vecs)} vs {len(ids)}) — "
                f"ignoring the dense index")
            return False
    except (OSError, ValueError) as e:
        log(f"could not load vectors — {type(e).__name__}: {e}")
        return False

    with _VEC_LOCK:
        _VECTORS = vecs
        _VEC_IDS = ids
        _VEC_POS = {int(m): i for i, m in enumerate(ids)}
    log(f"dense index resident — {len(ids)} vectors x {dim}d "
        f"({vecs.nbytes / 1048576:.0f} MB)")
    clear_cache()
    return True


def dense_ready() -> bool:
    with _VEC_LOCK:
        return _VECTORS is not None and len(_VECTORS) > 0


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# ══════════════════════════════════════════════════════════════════════════
# QUERY PREPARATION
# ══════════════════════════════════════════════════════════════════════════
# FTS5 treats these as operators. A person typing `iPhone 15 "Pro Max"` or
# `nike-air` means them literally, and an unescaped one is a syntax error that
# would surface as "search is broken".
_FTS_SPECIAL = re.compile(r'["():^*\-+,]')
_TOKEN = re.compile(r"[A-Za-z0-9_\']+")

# Words that carry no retrieval signal in a phrase-shaped query. Only used to
# build the *relaxed* fallback query, never to alter the strict one.
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "and",
    "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "show", "find", "me", "get", "search", "video", "videos", "clip",
    "clips", "moment", "moments", "where", "when", "what", "who", "which",
    "someone", "something", "anyone", "anything", "some", "any", "all",
}


def _fts_query(text: str, mode: str = "and") -> str:
    """Turn a person's sentence into a valid FTS5 query.

    Three shapes are produced from one input, and all three are tried in order
    until one returns enough:

      and     every token must appear — precise, can return nothing
      or      any token may appear — recall, ranked by bm25 so the documents
              containing more of them still come first
      prefix  the last token becomes a prefix match, so a half-typed word
              still finds things
    """
    tokens = _TOKEN.findall(text or "")
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
    if not tokens:
        return ""
    safe = [f'"{_FTS_SPECIAL.sub(" ", t)}"' for t in tokens]

    if mode == "and":
        return " AND ".join(safe)
    if mode == "prefix":
        body = safe[:-1]
        last = _FTS_SPECIAL.sub(" ", tokens[-1])
        tail = f'"{last}"*'
        return " OR ".join(body + [tail]) if body else tail

    content = [t for t in tokens if t.lower() not in _STOP]
    use = content if content else tokens
    return " OR ".join(f'"{_FTS_SPECIAL.sub(" ", t)}"' for t in use)


# ══════════════════════════════════════════════════════════════════════════
# RETRIEVERS
# ══════════════════════════════════════════════════════════════════════════
def _lexical(conn: sqlite3.Connection, query: str, limit: int) -> list:
    """BM25 over the moment text. Returns [(moment_id, rank)] best first.

    Escalates through and → or → prefix, stopping as soon as a shape returns a
    reasonable number of hits. A strict AND is the most precise thing available
    and usually enough; falling back only when it is not means a specific query
    stays specific.
    """
    if not query.strip():
        return []
    for mode in ("and", "or", "prefix"):
        fts = _fts_query(query, mode)
        if not fts:
            return []
        try:
            rows = conn.execute(
                "SELECT m.id FROM moments_fts f "
                "JOIN moments m ON m.id = f.rowid "
                "WHERE moments_fts MATCH ? "
                "ORDER BY bm25(moments_fts) LIMIT ?",
                (fts, limit)).fetchall()
        except sqlite3.Error as e:
            # No fts5 in this build, or a query shape it dislikes.
            if mode == "prefix":
                log(f"lexical search failed ({e}) — falling back to LIKE")
                return _like_fallback(conn, query, limit)
            continue
        if len(rows) >= 8 or mode == "prefix":
            return [(int(r[0]), i + 1) for i, r in enumerate(rows)]
        if rows:
            keep = [(int(r[0]), i + 1) for i, r in enumerate(rows)]
            # Too few for a confident answer, but real: widen and merge, with
            # the strict hits keeping their better ranks.
            wider = _lexical_mode(conn, query, "or", limit)
            seen = {m for m, _ in keep}
            for mid, _ in wider:
                if mid not in seen:
                    keep.append((mid, len(keep) + 1))
                    seen.add(mid)
            return keep
    return []


def _lexical_mode(conn: sqlite3.Connection, query: str, mode: str,
                  limit: int) -> list:
    fts = _fts_query(query, mode)
    if not fts:
        return []
    try:
        rows = conn.execute(
            "SELECT f.rowid FROM moments_fts f WHERE moments_fts MATCH ? "
            "ORDER BY bm25(moments_fts) LIMIT ?", (fts, limit)).fetchall()
    except sqlite3.Error:
        return []
    return [(int(r[0]), i + 1) for i, r in enumerate(rows)]


def _like_fallback(conn: sqlite3.Connection, query: str, limit: int) -> list:
    """Last resort when fts5 is absent. Slow and unranked, but not nothing."""
    tokens = [t for t in _TOKEN.findall(query) if t.lower() not in _STOP][:4]
    if not tokens:
        return []
    where = " OR ".join("text LIKE ?" for _ in tokens)
    args = [f"%{t}%" for t in tokens] + [limit]
    try:
        rows = conn.execute(
            f"SELECT id FROM moments WHERE {where} LIMIT ?", args).fetchall()
    except sqlite3.Error:
        return []
    return [(int(r[0]), i + 1) for i, r in enumerate(rows)]


def _dense(query: str, limit: int) -> list:
    """Exhaustive cosine search. Returns [(moment_id, rank)] best first.

    `argpartition` rather than a full sort: it selects the top-k in O(N) and
    only the k survivors are sorted. On 200k rows that is the difference
    between ~1 ms and ~20 ms, for an identical result.
    """
    with _VEC_LOCK:
        vecs, ids = _VECTORS, _VEC_IDS
    if vecs is None or not len(vecs):
        return []

    from .encoder import get_encoder
    enc = get_encoder()
    if enc is None:
        return []
    try:
        import numpy as np
        q = enc.encode_query(query).astype("float32")
    except Exception as e:
        log(f"query encode failed — {type(e).__name__}: {e}")
        return []

    # Both sides are unit length, so the dot product is the cosine.
    sims = vecs @ q
    k = min(limit, len(sims))
    if k <= 0:
        return []
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(ids[i]), rank + 1) for rank, i in enumerate(top)]


# ══════════════════════════════════════════════════════════════════════════
# FUSION
# ══════════════════════════════════════════════════════════════════════════
def _rrf(rank_lists: list, k: int = None) -> dict:
    """Reciprocal Rank Fusion. Returns {moment_id: fused_score}."""
    k = k or config.RRF_K
    fused = {}
    for weight, ranks in rank_lists:
        if not ranks:
            continue
        for mid, rank in ranks:
            fused[mid] = fused.get(mid, 0.0) + weight / (k + rank)
    return fused


def _phrase_bonus(text: str, query: str) -> float:
    """A small, honest boost for containing the query as written.

    Neither retriever rewards word order — BM25 is a bag of words and a dense
    vector is a blur. When somebody types "red bicycle" and a passage literally
    says "red bicycle", that is worth something over one that says "bicycle …
    red door". Kept small so it tunes the top rather than reordering it.
    """
    if not text or not query:
        return 0.0
    t, q = text.lower(), query.lower().strip()
    if len(q) < 4:
        return 0.0
    if q in t:
        return 0.30
    tokens = [w for w in _TOKEN.findall(q) if w.lower() not in _STOP]
    if len(tokens) < 2:
        return 0.0
    # Adjacent pairs from the query appearing intact in the passage.
    hits = sum(1 for a, b in zip(tokens, tokens[1:])
               if f"{a.lower()} {b.lower()}" in t)
    return min(0.20, 0.07 * hits)


# ══════════════════════════════════════════════════════════════════════════
# THE SEARCH
# ══════════════════════════════════════════════════════════════════════════
# Every ordering the grouped results understand. Unlike the library's sorts,
# none of these becomes SQL — they run over the already-ranked pool in
# `_order_key`, so the set lives here next to the code that reads it and the
# HTTP layer validates against it rather than inventing its own list.
SORTS = ("relevance", "recent", "oldest", "longest", "shortest",
         "liked", "matches")


def search(conn: sqlite3.Connection, query: str, limit: int = 24,
           offset: int = 0, sources: list = None, video_key: str = None,
           candidates: int = None, sort: str = "relevance",
           creator: str = None, category: str = None,
           min_dur: float = None, max_dur: float = None,
           min_hits: int = None) -> dict:
    """Run a hybrid search and return grouped video results.

    The whole pipeline, in order:
      1. BM25 over the passage text                  → ranks
      2. Exhaustive cosine over the passage vectors  → ranks
      3. RRF fuse the two rank lists                 → per-moment score
      4. Weight by evidence type and phrase match    → per-moment score
      5. Group by video, best moment plus damped rest → per-video score
      6. Filter and order the grouped videos          → the page
      7. Attach every matching moment, sorted by time, for the ribbon

    Filtering and sorting happen *after* grouping, never by narrowing the
    candidate pool first: a `WHERE creator = …` in front of the ranking would
    change which moments compete, so "the best matches, from this creator"
    and "the best matches from this creator" would quietly differ. Every
    result carries the same score it would have had unfiltered.
    """
    t0 = time.perf_counter()
    query = (query or "").strip()
    if not query:
        return {"ok": True, "query": "", "results": [], "total": 0,
                "took_ms": 0, "mode": "empty"}

    cache_key = (query, limit, offset, tuple(sources or ()), video_key,
                 sort, creator, category, min_dur, max_dur, min_hits)
    with _CACHE_LOCK:
        hit = _CACHE.get(cache_key)
        if hit is not None:
            _CACHE.move_to_end(cache_key)
            out = dict(hit)
            out["cached"] = True
            return out

    depth = candidates or config.CANDIDATES
    lex = _lexical(conn, query, depth)
    den = _dense(query, depth) if dense_ready() else []

    mode = ("hybrid" if lex and den else
            "lexical" if lex else "dense" if den else "none")
    fused = _rrf([(1.0, lex), (1.0, den)])
    if not fused:
        return {"ok": True, "query": query, "results": [], "total": 0,
                "took_ms": round((time.perf_counter() - t0) * 1000, 1),
                "mode": mode, "dense": dense_ready()}

    # One round trip for every candidate's row.
    ids = list(fused)
    rows = {}
    CHUNK = 900                      # under SQLITE_MAX_VARIABLE_NUMBER
    for i in range(0, len(ids), CHUNK):
        part = ids[i:i + CHUNK]
        q = ("SELECT id, video_key, t_start, t_end, source, weight, text "
             f"FROM moments WHERE id IN ({','.join('?' * len(part))})")
        for r in conn.execute(q, part):
            rows[int(r[0])] = r

    lex_rank = dict(lex)
    den_rank = dict(den)

    per_video = {}
    for mid, base in fused.items():
        row = rows.get(mid)
        if not row:
            continue
        _id, vkey, t_start, t_end, source, weight, text = row
        if sources and source not in sources:
            continue
        if video_key and vkey != video_key:
            continue

        score = base * float(weight or 1.0) * (1.0 + _phrase_bonus(text, query))
        slot = per_video.setdefault(vkey, {"moments": [], "score": 0.0})
        slot["moments"].append({
            "id": mid, "t_start": t_start, "t_end": t_end, "source": source,
            "text": text, "score": round(score, 6),
            "lex_rank": lex_rank.get(mid), "dense_rank": den_rank.get(mid),
        })

    if not per_video:
        return {"ok": True, "query": query, "results": [], "total": 0,
                "took_ms": round((time.perf_counter() - t0) * 1000, 1),
                "mode": mode, "dense": dense_ready()}

    for vkey, slot in per_video.items():
        ms = sorted(slot["moments"], key=lambda m: -m["score"])
        best = ms[0]["score"]
        # Corroboration, with sharply diminishing returns: the 2nd moment adds
        # a third of its score, the 3rd a quarter, and so on. A video with one
        # excellent match still beats a video with ten weak ones.
        extra = sum(m["score"] / (i + 2) for i, m in enumerate(ms[1:6]))
        distinct = len({m["source"] for m in ms})
        # Two different KINDS of evidence agreeing is a stronger signal than
        # two hits of the same kind, which are often the same sentence twice.
        slot["score"] = (best + 0.5 * extra) * (1.0 + 0.08 * (distinct - 1))
        slot["best"] = ms[0]
        slot["moments"] = sorted(
            ms, key=lambda m: (m["t_start"] if m["t_start"] is not None
                               else -1.0))

    # Meta for every matched video, not just the page: the filters and the
    # non-relevance sorts read columns that live here, so they cannot be
    # applied until it is loaded. The pool is bounded by the candidate depth,
    # so this is one extra query, not one per result.
    meta = _video_meta(conn, list(per_video))

    def _keep(vkey, slot):
        m = meta.get(vkey, {})
        if creator and (m.get("creator") or "") != creator:
            return False
        if category and (m.get("category") or "") != category:
            return False
        dur = float(m.get("duration") or 0.0)
        # A video whose duration was never probed has 0.0, which is not the
        # same as "shorter than the floor" — excluding it would hide real
        # matches for a metadata gap, so an unknown duration passes.
        if min_dur is not None and dur and dur < float(min_dur):
            return False
        if max_dur is not None and dur and dur > float(max_dur):
            return False
        if min_hits is not None and len(slot["moments"]) < int(min_hits):
            return False
        return True

    kept = [(k, s) for k, s in per_video.items() if _keep(k, s)]

    def _num(vkey, col):
        v = meta.get(vkey, {}).get(col)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Every ordering falls back to the relevance score, so a tie — or a column
    # the bundle never filled — degrades to "best match first" rather than to
    # an arbitrary order that changes between pages.
    def _order_key(kv):
        vkey, slot = kv
        rel = -slot["score"]
        if sort == "recent" or sort == "oldest":
            when = _num(vkey, "created_at")
            if when is None:
                return (1, 0.0, rel)
            return (0, -when if sort == "recent" else when, rel)
        if sort == "longest" or sort == "shortest":
            dur = _num(vkey, "duration")
            if not dur:
                return (1, 0.0, rel)
            return (0, -dur if sort == "longest" else dur, rel)
        if sort == "liked":
            likes = _num(vkey, "likes")
            if likes is None:
                return (1, 0.0, rel)
            return (0, -likes, rel)
        if sort == "matches":
            return (0, -len(slot["moments"]), rel)
        return (0, rel, rel)

    # Which creators and categories this query actually reached, counted over
    # the matched pool rather than the whole archive: offering a filter that
    # would return nothing is worse than offering no filter. Counted before
    # `_keep` so the chip that is currently active still shows its own count
    # and can be switched off.
    facets = {"creators": {}, "categories": {}}
    for vkey in per_video:
        m = meta.get(vkey, {})
        for field, col in (("creators", "creator"), ("categories", "category")):
            val = (m.get(col) or "").strip()
            if val:
                facets[field][val] = facets[field].get(val, 0) + 1
    facets = {
        field: [{"value": v, "count": c}
                for v, c in sorted(vals.items(), key=lambda kv: (-kv[1], kv[0]))[:14]]
        for field, vals in facets.items()
    }

    order = sorted(kept, key=_order_key)
    total = len(order)
    page = order[offset:offset + limit]

    results = []
    for rank, (vkey, slot) in enumerate(page, start=offset + 1):
        m = meta.get(vkey, {})
        results.append({
            "rank": rank,
            "video_key": vkey,
            "score": round(slot["score"], 6),
            "title": m.get("title") or m.get("caption") or f"Video {vkey}",
            "caption": m.get("caption"),
            "creator": m.get("creator"),
            "category": m.get("category"),
            "duration": m.get("duration") or 0.0,
            "width": m.get("width"),
            "height": m.get("height"),
            "likes": m.get("likes"),
            "created_at": m.get("created_at"),
            "msg_id": m.get("msg_id"),
            "poster": m.get("poster"),
            "has_file": media.resident(m.get("local_path"), vkey),
            "moment_count": m.get("moment_count") or len(slot["moments"]),
            "hit_count": len(slot["moments"]),
            "best": slot["best"],
            "moments": slot["moments"][:24],
        })

    out = {
        "ok": True, "query": query, "results": results, "total": total,
        "offset": offset, "limit": limit, "mode": mode,
        "dense": dense_ready(),
        "sort": sort,
        # `matched` is before filtering and `total` after, so the UI can say
        # "18 of 340, narrowed by your filters" instead of implying the query
        # itself only found 18.
        "matched": len(per_video),
        "facets": facets,
        "filters": {"creator": creator, "category": category,
                    "min_dur": min_dur, "max_dur": max_dur,
                    "min_hits": min_hits,
                    "sources": list(sources or ())},
        "candidates": {"lexical": len(lex), "dense": len(den),
                       "fused": len(fused)},
        "took_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    with _CACHE_LOCK:
        _CACHE[cache_key] = out
        while len(_CACHE) > config.QUERY_CACHE:
            _CACHE.popitem(last=False)
    return out


def _video_meta(conn: sqlite3.Connection, keys: list) -> dict:
    """Pull the precomputed card data for a page of results, in one query."""
    if not keys:
        return {}
    out = {}
    CHUNK = 900
    for i in range(0, len(keys), CHUNK):
        part = keys[i:i + CHUNK]
        try:
            cur = conn.execute(
                "SELECT * FROM video_index WHERE video_key IN "
                f"({','.join('?' * len(part))})", part)
        except sqlite3.Error:
            return out
        names = [d[0] for d in cur.description]
        for row in cur.fetchall():
            r = dict(zip(names, row))
            out[r["video_key"]] = r
    return out


# ══════════════════════════════════════════════════════════════════════════
# SUPPORTING QUERIES
# ══════════════════════════════════════════════════════════════════════════
def suggestions(conn: sqlite3.Connection, prefix: str, limit: int = 8) -> list:
    """Type-ahead from real corpus terms, not a canned list.

    Suggestions come from creators, categories and the most common content
    words actually present, so every suggestion is guaranteed to return
    results — the failure mode of a static list is suggesting a search that
    finds nothing.
    """
    prefix = (prefix or "").strip().lower()
    if len(prefix) < 2:
        return []
    out, seen = [], set()

    for sql, args in (
        ("SELECT DISTINCT creator FROM video_index WHERE creator IS NOT NULL "
         "AND LOWER(creator) LIKE ? LIMIT ?", (prefix + "%", limit)),
        ("SELECT DISTINCT category FROM video_index WHERE category IS NOT NULL "
         "AND LOWER(category) LIKE ? LIMIT ?", (prefix + "%", limit)),
    ):
        try:
            for (val,) in conn.execute(sql, args):
                if val and val.lower() not in seen:
                    seen.add(val.lower())
                    out.append({"text": val, "kind": "name"})
        except sqlite3.Error:
            pass

    if len(out) < limit:
        try:
            rows = conn.execute(
                "SELECT text FROM moments_fts WHERE moments_fts MATCH ? "
                "LIMIT ?", (f'"{_FTS_SPECIAL.sub(" ", prefix)}"*',
                            limit * 6)).fetchall()
            for (text,) in rows:
                for word in _TOKEN.findall((text or "").lower()):
                    if word.startswith(prefix) and word not in seen \
                            and len(word) > len(prefix):
                        seen.add(word)
                        out.append({"text": word, "kind": "term"})
                        break
                if len(out) >= limit:
                    break
        except sqlite3.Error:
            pass
    return out[:limit]


def similar(conn: sqlite3.Connection, video_key: str, limit: int = 12) -> list:
    """Videos that look like this one, by averaging its moment vectors.

    The centroid of a video's passages is a serviceable description of the
    video, and comparing centroids finds reels about the same thing without
    anybody typing a query.
    """
    if not dense_ready():
        return []
    try:
        import numpy as np
    except ImportError:
        return []

    ids = [int(r[0]) for r in conn.execute(
        "SELECT id FROM moments WHERE video_key = ?", (video_key,))]
    if not ids:
        return []

    with _VEC_LOCK:
        vecs, all_ids, pos = _VECTORS, _VEC_IDS, _VEC_POS
    if vecs is None:
        return []
    rows = [pos[i] for i in ids if i in pos]
    if not rows:
        return []

    centroid = vecs[rows].mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return []
    sims = vecs @ (centroid / norm)

    k = min(len(sims), limit * 40)
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]

    by_video = {}
    for i in top:
        mid = int(all_ids[i])
        row = conn.execute(
            "SELECT video_key FROM moments WHERE id = ?", (mid,)).fetchone()
        if not row or row[0] == video_key:
            continue
        vk = row[0]
        if vk not in by_video:
            by_video[vk] = float(sims[i])
        if len(by_video) >= limit:
            break

    keys = list(by_video)
    meta = _video_meta(conn, keys)
    out = []
    for vk in sorted(keys, key=lambda k: -by_video[k]):
        m = meta.get(vk, {})
        out.append({"video_key": vk, "similarity": round(by_video[vk], 4),
                    "title": m.get("title") or f"Video {vk}",
                    "duration": m.get("duration"), "poster": m.get("poster"),
                    "creator": m.get("creator"),
                    "moment_count": m.get("moment_count")})
    return out


def stats(conn: sqlite3.Connection) -> dict:
    """What search knows right now — shown in the UI so the state is visible
    rather than guessed at."""
    def one(sql, default=0):
        try:
            row = conn.execute(sql).fetchone()
            return row[0] if row and row[0] is not None else default
        except sqlite3.Error:
            return default

    by_source = {}
    try:
        for src, n in conn.execute(
                "SELECT source, COUNT(*) FROM moments GROUP BY source"):
            by_source[src or "unknown"] = n
    except sqlite3.Error:
        pass

    vec = index.vector_state()
    return {
        "moments": one("SELECT COUNT(*) FROM moments"),
        "videos": one("SELECT COUNT(*) FROM video_index"),
        "playable": len(media.resident_keys(conn)),
        "by_source": by_source,
        "dense_ready": dense_ready(),
        "dense_count": vec.get("count", 0),
        "dense_model": vec.get("model"),
        "cache_size": len(_CACHE),
        # Two aggregates the landing page states as facts about the archive.
        # Both are one indexed scan over a table that has one row per video, so
        # they cost nothing next to the counts above, and a total that has to be
        # assembled in the browser out of a paged library call would be wrong
        # for every page but the last.
        "seconds": one("SELECT COALESCE(SUM(duration), 0) FROM video_index"),
        "creators": one("SELECT COUNT(DISTINCT creator) FROM video_index "
                        "WHERE creator IS NOT NULL AND creator <> ''"),
    }
