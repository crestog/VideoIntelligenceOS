"""
atlas.roadmap — the archive read as a curriculum.

The Graph tab answers "what is connected to what". This module asks a different
question of the same two derived tables: *in what order should somebody watch
this to end up knowing something*. No new extraction, no model call, no
hand-written syllabus — the order is inferred from which concepts co-occur
across videos and, crucially, from how one-sided that co-occurrence is.

The inference is subsumption. If nearly every video that mentions "compound
interest" also mentions "interest", while plenty of videos about interest never
mention compound interest, then interest is the broader idea and the one to
watch first. As probabilities over the two video sets:

    P(A|B) is high             A turns up almost whenever B does
    P(A|B) − P(B|A) is large   but not the other way round
    |A| > |B|                  and A covers more of the scope

All three together make A a prerequisite of B. The third is not decoration: it
is a strict order on support size, which is what makes the result acyclic by
construction rather than by a cycle-breaking heuristic.

What comes out is a layered DAG. Stage one is everything with no prerequisite
among the kept concepts; a concept sits one level past its deepest
prerequisite. Each step carries the moments worth watching for it, with
timecodes, so a stage is a playlist rather than a reading list.

Two scopes. With no goal the plan covers the whole archive: the concepts that
connect the most of it, ordered. With a goal the scope is whatever a hybrid
search returns for it, so "learn to edit hooks" plans over the reels about that
and nothing else — and support, prerequisites and stages are all recomputed
inside that scope, because a concept that is foundational across an archive can
be advanced inside one corner of it.

Progress is the only thing stored: one row per step, ticked by hand. The plan
itself is derived and cached in memory against the graph's own size, so a
rebuild invalidates it and nothing on disk can ever disagree with the graph.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time

from . import search
from .tgchannel import log

# ── the shape of a plan ───────────────────────────────────────────────────
# A curriculum nobody can hold in their head is not a curriculum. Every cap
# here is a reading decision rather than a data one — the graph keeps
# everything, and a step dropped from the plan is still one click away in the
# Graph tab.
BREADTH         = 60      # steps in one plan
MIN_SUPPORT     = 2       # videos a concept needs before it earns a step
MIN_SHARED      = 3       # videos two concepts must share before order is claimed
P_MIN           = 0.60    # P(A|B): how reliably the prerequisite turns up
MARGIN          = 0.20    # P(A|B) − P(B|A): how one-sided that has to be
MAX_PREREQ      = 3       # prerequisites kept per step, strongest first
MOMENTS_IN_PLAN = 5       # watch material carried in the plan itself
MOMENTS_IN_STEP = 24      # …and in one step's drill-down
SCOPE_CAP       = 400     # videos a goal search may pull into scope
KEYS_PER_LOOKUP = 180     # keys per moment query, well under the variable limit
CONCEPT_KINDS   = ("tag", "hashtag")
POINT_WIDTH_S   = 2.5     # a frame note has no end; give it one so it can play
CACHE_MAX       = 8

_DDL = (
    # Progress is per concept, not per plan. A thing learned stays learned when
    # the goal changes, and the same concept reached from two different goals is
    # the same concept; `goal` records where it was ticked off, which is context
    # rather than identity.
    "CREATE TABLE IF NOT EXISTS roadmap_progress ("
    "  step_id TEXT PRIMARY KEY,"
    "  state TEXT NOT NULL,"          # done | skip
    "  goal TEXT,"
    "  at REAL)",
)

_LOCK = threading.RLock()
_CACHE = {}                       # (goal, breadth, support, fingerprint) → plan
_FTS_BAD = re.compile(r'["():^*\-+,]')


def ensure_schema(conn: sqlite3.Connection) -> None:
    for ddl in _DDL:
        conn.execute(ddl)
    conn.commit()


def invalidate() -> None:
    """Forget every cached plan. Called after the graph is rebuilt by hand."""
    with _LOCK:
        _CACHE.clear()


def _fingerprint(conn: sqlite3.Connection) -> str:
    """Cheap proof that the graph has not moved under a cached plan."""
    try:
        nodes = conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
        edges = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
    except sqlite3.Error:
        return ""
    return f"{nodes}:{edges}"


def _archive_videos(conn: sqlite3.Connection) -> int:
    try:
        return int(conn.execute(
            "SELECT COUNT(*) FROM video_index").fetchone()[0])
    except sqlite3.Error:
        return 0


# ══════════════════════════════════════════════════════════════════════════
# SCOPE
# ══════════════════════════════════════════════════════════════════════════
def _scope(conn: sqlite3.Connection, goal: str) -> dict:
    """Which videos this plan is about, and how much each one counts.

    An empty goal plans over everything. A goal runs one hybrid search and
    plans over what it found, keeping the search's own scores — so a concept
    sitting in the most relevant reels outranks one that merely appears often.

    A goal matching almost nothing falls back to the whole archive and says so.
    Three videos cannot establish that one idea precedes another, and a plan
    built from them would be confident nonsense.
    """
    goal = (goal or "").strip()
    if not goal:
        return {"mode": "archive", "goal": "", "keys": None, "weights": {},
                "note": "every indexed video"}
    try:
        found = search.search(conn, goal, limit=SCOPE_CAP, candidates=3000)
    except Exception as e:                                   # noqa: BLE001
        log(f"roadmap: goal search failed — {type(e).__name__}: {e}", "WARN")
        return {"mode": "archive", "goal": goal, "keys": None, "weights": {},
                "note": f"the goal search failed ({type(e).__name__}) — "
                        f"planning over the whole archive instead"}
    rows = found.get("results") or []
    keys, weights = [], {}
    for r in rows:
        key = str(r.get("video_key") or "")
        if not key:
            continue
        keys.append(key)
        weights[key] = float(r.get("score") or 0.0) or 1.0
    if len(keys) < 3:
        return {"mode": "archive", "goal": goal, "keys": None, "weights": {},
                "note": f"only {len(keys)} video(s) match “{goal}” — planning "
                        f"over the whole archive, because an order inferred "
                        f"from three videos would not mean anything"}
    more = int(found.get("total") or 0) > len(keys)
    return {"mode": "goal", "goal": goal, "keys": keys, "weights": weights,
            "note": (f"{len(keys)} video(s) matching “{goal}”"
                     + (f" — the search found more, the strongest {SCOPE_CAP} "
                        f"are planned" if more else ""))}


# ══════════════════════════════════════════════════════════════════════════
# CONCEPTS
# ══════════════════════════════════════════════════════════════════════════
def _concepts(conn: sqlite3.Connection, scope: dict, breadth: int,
              min_support: int) -> list:
    """The concepts worth a step, each with the videos it covers.

    Two queries rather than one pass in Python. The first ranks concepts by how
    many videos they reach, which SQLite does over `graph_edges(dst)`; only the
    survivors have their member sets read back. On a large archive the edge
    table is the biggest thing in the database and this never loads all of it.
    """
    keys = scope.get("keys")
    scoped = [f"v:{k}" for k in (keys or [])]
    marks = ",".join("?" * len(scoped)) if scoped else ""
    where = ["n.kind IN (" + ",".join("?" * len(CONCEPT_KINDS)) + ")"]
    args = list(CONCEPT_KINDS)
    if scoped:
        where.append(f"e.src IN ({marks})")
        args += scoped
    else:
        where.append("e.src LIKE 'v:%'")
    try:
        rows = conn.execute(
            "SELECT n.id, n.kind, n.label, n.sub, "
            "       COUNT(DISTINCT e.src) AS videos "
            "FROM graph_nodes n JOIN graph_edges e ON e.dst = n.id "
            "WHERE " + " AND ".join(where) +
            " GROUP BY n.id HAVING videos >= ? "
            "ORDER BY videos DESC, n.label LIMIT ?",
            args + [int(min_support), int(breadth)]).fetchall()
    except sqlite3.Error as e:
        log(f"roadmap: nothing to plan from — {e}", "WARN")
        return []
    kept = [{"id": r[0], "kind": r[1], "label": r[2], "sub": r[3] or "",
             "videos": int(r[4])} for r in rows]
    if not kept:
        return []

    ids = [c["id"] for c in kept]
    sql = ("SELECT dst, src FROM graph_edges "
           f"WHERE dst IN ({','.join('?' * len(ids))})")
    args = list(ids)
    if scoped:
        sql += f" AND src IN ({marks})"
        args += scoped
    sets = {i: set() for i in ids}
    try:
        for dst, src in conn.execute(sql, args):
            src = str(src)
            if src.startswith("v:"):
                sets[dst].add(src[2:])
    except sqlite3.Error as e:
        log(f"roadmap: could not read concept membership — {e}", "WARN")
        return []

    weights = scope.get("weights") or {}
    out = []
    for c in kept:
        c["keys"] = sets.get(c["id"], set())
        c["videos"] = len(c["keys"])
        # Reach, not count: inside a goal scope the search's own relevance is
        # the better ranking signal, and outside one it collapses to the count.
        c["reach"] = round(sum(weights.get(k, 1.0) for k in c["keys"]), 4)
        if c["videos"] >= min_support:
            out.append(c)
    return out


# ══════════════════════════════════════════════════════════════════════════
# ORDER
# ══════════════════════════════════════════════════════════════════════════
def _order(concepts: list) -> list:
    """Prerequisite edges, from one-sided co-occurrence.

    Each unordered pair is looked at once, and the broader concept is the only
    candidate prerequisite: `|A| > |B|` is tested first, so equal-support pairs
    produce nothing and the relation can never contain a cycle. Only the
    strongest few incoming edges per step survive, because a step listing eight
    prerequisites teaches nobody anything about what to watch first.
    """
    by_dst = {}
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = concepts[i], concepts[j]
            if len(a["keys"]) == len(b["keys"]):
                continue
            if len(a["keys"]) < len(b["keys"]):
                a, b = b, a                       # a is the broader of the two
            shared = len(a["keys"] & b["keys"])
            if shared < MIN_SHARED:
                continue
            p_fwd = shared / float(len(b["keys"]))          # P(A|B)
            p_bak = shared / float(len(a["keys"]))          # P(B|A)
            if p_fwd < P_MIN or (p_fwd - p_bak) < MARGIN:
                continue
            by_dst.setdefault(b["id"], []).append({
                "src": a["id"], "dst": b["id"], "shared": shared,
                "p_forward": round(p_fwd, 4), "p_back": round(p_bak, 4),
                "strength": round(p_fwd * (p_fwd - p_bak), 5),
            })
    out = []
    for cand in by_dst.values():
        cand.sort(key=lambda e: -e["strength"])
        out.extend(cand[:MAX_PREREQ])
    return out


def _layer(concepts: list, edges: list) -> dict:
    """Longest-path levels, by Kahn, with a guard that should never fire.

    The strict support comparison in `_order` makes a cycle impossible, so the
    guard is not load-bearing. It exists because an infinite loop inside a web
    request is a far worse failure than a dropped edge: a stall drops the
    weakest unresolved prerequisite, logs it, and carries on.
    """
    ids = [c["id"] for c in concepts]
    known = set(ids)
    incoming = {i: set() for i in ids}
    outgoing = {i: set() for i in ids}
    strength = {}
    for e in edges:
        if e["src"] not in known or e["dst"] not in known:
            continue
        incoming[e["dst"]].add(e["src"])
        outgoing[e["src"]].add(e["dst"])
        strength[(e["src"], e["dst"])] = e["strength"]

    left = {i: set(v) for i, v in incoming.items()}
    ready = [i for i in ids if not left[i]]
    level, guard = {}, 0
    while len(level) < len(ids):
        if not ready:
            guard += 1
            unresolved = [(s, k) for k, s in strength.items()
                          if k[1] not in level and k[0] in left.get(k[1], ())]
            if guard > len(ids) or not unresolved:
                break
            _s, (src, dst) = min(unresolved)
            left[dst].discard(src)
            log(f"roadmap: broke a prerequisite loop at {src} → {dst}", "WARN")
            ready = [i for i in ids if i not in level and not left[i]]
            continue
        nid = ready.pop()
        if nid in level:
            continue
        level[nid] = 1 + max([level.get(p, 0) for p in incoming[nid]] or [0])
        for nxt in outgoing[nid]:
            left[nxt].discard(nid)
            if nxt not in level and not left[nxt]:
                ready.append(nxt)
    for i in ids:
        level.setdefault(i, 1)
    return level


# ══════════════════════════════════════════════════════════════════════════
# WATCH MATERIAL
# ══════════════════════════════════════════════════════════════════════════
def _moments(conn: sqlite3.Connection, token: str, keys: list,
             limit: int) -> list:
    """The passages worth watching for one concept, best first.

    FTS over the phrase first, because the moment that says the word is the
    moment to watch. A concept mined out of a list column may never appear in
    any transcript, so the fallback is the strongest passages of the videos that
    carry it — the right videos, just not a pinpointed instant. Which of the two
    happened is reported per moment as `said`, so the interface can be honest
    about it instead of implying a quote that does not exist.
    """
    keys = [k for k in (keys or []) if k][:KEYS_PER_LOOKUP]
    if not keys:
        return []
    marks = ",".join("?" * len(keys))
    phrase = '"' + _FTS_BAD.sub(" ", str(token or "")).strip() + '"'
    rows = []
    if len(phrase) > 3:
        try:
            rows = conn.execute(
                "SELECT m.id, m.video_key, m.t_start, m.t_end, m.source, "
                "       m.weight, m.text "
                "FROM moments_fts f JOIN moments m ON m.id = f.rowid "
                f"WHERE moments_fts MATCH ? AND m.video_key IN ({marks}) "
                "ORDER BY bm25(moments_fts) LIMIT ?",
                [phrase] + keys + [int(limit)]).fetchall()
        except sqlite3.Error:
            rows = []
    said = bool(rows)
    if not rows:
        try:
            rows = conn.execute(
                "SELECT id, video_key, t_start, t_end, source, weight, text "
                f"FROM moments WHERE video_key IN ({marks}) "
                "ORDER BY weight DESC, LENGTH(text) DESC LIMIT ?",
                keys + [int(limit)]).fetchall()
        except sqlite3.Error:
            return []
    out = []
    for mid, key, t0, t1, source, weight, text in rows:
        start = float(t0 or 0.0)
        end = float(t1) if t1 is not None else start + POINT_WIDTH_S
        if end <= start:
            end = start + POINT_WIDTH_S
        out.append({"id": int(mid), "video_key": str(key),
                    "t_start": round(start, 2), "t_end": round(end, 2),
                    "seconds": round(end - start, 2),
                    "source": source or "meta",
                    "weight": round(float(weight or 1.0), 3),
                    "text": (text or "")[:600], "said": said})
    return out


def _videos(conn: sqlite3.Connection, keys) -> dict:
    """Enough about each video for a card, keyed by video_key."""
    keys = sorted({k for k in (keys or []) if k})
    out = {}
    for i in range(0, len(keys), 800):
        part = keys[i:i + 800]
        try:
            cur = conn.execute(
                "SELECT video_key, title, caption, creator, category, "
                "       duration, msg_id, moment_count, created_at "
                f"FROM video_index WHERE video_key IN ({','.join('?' * len(part))})",
                part)
        except sqlite3.Error:
            return out
        names = [d[0] for d in cur.description]
        for row in cur.fetchall():
            rec = dict(zip(names, row))
            out[str(rec["video_key"])] = rec
    return out


# ══════════════════════════════════════════════════════════════════════════
# THE PLAN
# ══════════════════════════════════════════════════════════════════════════
def _stage_label(level: int, top: int) -> tuple:
    """A stage's name and what being in it means. Descriptive, not motivational."""
    if level == 1:
        return ("Foundations",
                "nothing here needs anything else first")
    if level == top:
        return ("Furthest out",
                "everything here sits on top of every earlier stage")
    return (f"Stage {level}",
            "each of these needs something from an earlier stage")


def _empty(scope: dict, took: float, note: str) -> dict:
    return {"ok": True, "goal": scope.get("goal", ""),
            "mode": scope.get("mode", "archive"),
            "scope_note": scope.get("note", ""), "note": note,
            "stages": [], "steps": [], "edges": [], "videos": {},
            "stats": {"concepts": 0, "stages": 0, "videos": 0, "minutes": 0.0,
                      "moments": 0, "scope_videos": 0},
            "built_ms": round(took * 1000, 1)}


def _build(conn: sqlite3.Connection, goal: str, breadth: int,
           min_support: int) -> dict:
    """Everything about a plan except who has ticked what off."""
    t0 = time.perf_counter()
    scope = _scope(conn, goal)
    concepts = _concepts(conn, scope, breadth, min_support)
    if not concepts:
        return _empty(
            scope, time.perf_counter() - t0,
            "No concept in scope reaches enough videos to order. The roadmap "
            "reads the same graph the Graph tab does, so build the index and "
            "the graph first — or widen the goal.")

    edges = _order(concepts)
    level = _layer(concepts, edges)
    prereq, unlocks = {}, {}
    for e in edges:
        prereq.setdefault(e["dst"], []).append(e)
        unlocks.setdefault(e["src"], []).append(e["dst"])

    scope_n = len(scope["keys"]) if scope.get("keys") else _archive_videos(conn)
    weights = scope.get("weights") or {}
    by_label = {c["id"]: c["label"] for c in concepts}
    steps, seen_keys = [], set()
    for c in concepts:
        ranked = sorted(c["keys"], key=lambda k: -weights.get(k, 1.0))
        moments = _moments(conn, c["label"].lstrip("#"), ranked,
                           MOMENTS_IN_PLAN)
        keys = ranked[:24]
        seen_keys.update(m["video_key"] for m in moments)
        seen_keys.update(keys)
        steps.append({
            "id": c["id"], "label": c["label"], "kind": c["kind"],
            "group": c["sub"], "level": level.get(c["id"], 1),
            "videos": c["videos"], "reach": c["reach"],
            "share": round(c["videos"] / float(max(scope_n, 1)), 4),
            "keys": keys,
            "prereq": [{"id": e["src"], "label": by_label.get(e["src"], ""),
                        "shared": e["shared"], "p_forward": e["p_forward"],
                        "p_back": e["p_back"], "strength": e["strength"]}
                       for e in sorted(prereq.get(c["id"], []),
                                       key=lambda e: -e["strength"])],
            "unlocks": [{"id": i, "label": by_label.get(i, "")}
                        for i in unlocks.get(c["id"], [])],
            "moments": moments,
            "seconds": round(sum(m["seconds"] for m in moments), 1),
            "said": any(m["said"] for m in moments),
        })

    steps.sort(key=lambda s: (s["level"], -s["reach"], s["label"]))
    top = max((s["level"] for s in steps), default=1)
    stages = []
    for lvl in range(1, top + 1):
        mine = [s for s in steps if s["level"] == lvl]
        if not mine:
            continue
        title, why = _stage_label(lvl, top)
        stages.append({"level": lvl, "title": title, "why": why,
                       "steps": [s["id"] for s in mine],
                       "seconds": round(sum(s["seconds"] for s in mine), 1)})

    took = time.perf_counter() - t0
    return {"ok": True, "goal": scope["goal"], "mode": scope["mode"],
            "scope_note": scope["note"], "note": "",
            "stages": stages, "steps": steps, "edges": edges,
            "videos": _videos(conn, seen_keys),
            "stats": {"concepts": len(steps), "stages": len(stages),
                      "videos": len({k for s in steps for k in s["keys"]}),
                      "minutes": round(sum(s["seconds"] for s in steps) / 60.0, 1),
                      "moments": sum(len(s["moments"]) for s in steps),
                      "scope_videos": scope_n,
                      "ordered": len(edges)},
            "built_ms": round(took * 1000, 1)}


def plan(conn: sqlite3.Connection, goal: str = "", breadth: int = BREADTH,
         min_support: int = MIN_SUPPORT) -> dict:
    """A plan, with progress laid over it.

    The structure is cached and the ticks are not: marking a step off must not
    invalidate a build that takes a second, and a cached structure must never
    carry somebody's progress from before the tick. So the two are joined here,
    on every request, and the cache holds only what the graph can change.
    """
    ensure_schema(conn)
    breadth = max(6, min(int(breadth or BREADTH), 200))
    min_support = max(1, min(int(min_support or MIN_SUPPORT), 50))
    key = ((goal or "").strip().lower(), breadth, min_support,
           _fingerprint(conn))
    with _LOCK:
        built = _CACHE.get(key)
    cached = built is not None
    if not cached:
        built = _build(conn, goal, breadth, min_support)
        with _LOCK:
            if len(_CACHE) >= CACHE_MAX:
                _CACHE.clear()
            _CACHE[key] = built

    marks = progress(conn)
    steps, done, skipped = [], 0, 0
    for s in built.get("steps") or []:
        row = marks.get(s["id"]) or {}
        state = str(row.get("state") or "")
        if state == "done":
            done += 1
        elif state == "skip":
            skipped += 1
        steps.append(dict(s, state=state, marked_at=row.get("at") or 0))

    at_hand = {s["id"]: s for s in steps}
    stages = []
    for st in built.get("stages") or []:
        mine = [at_hand[i] for i in (st.get("steps") or []) if i in at_hand]
        stages.append(dict(
            st, count=len(mine),
            done=sum(1 for s in mine if s["state"] == "done"),
            marked=sum(1 for s in mine if s["state"])))

    # A step whose prerequisites are all ticked off is the one to watch next,
    # which is the only question a curriculum has to answer at any moment.
    ready = [s["id"] for s in steps
             if not s["state"]
             and all((marks.get(p["id"]) or {}).get("state")
                     for p in s["prereq"])]

    out = dict(built)
    out["steps"] = steps
    out["stages"] = stages
    out["ready"] = ready
    out["cached"] = cached
    stats = dict(built.get("stats") or {})
    n = len(steps)
    stats.update(
        done=done, skipped=skipped, marked=done + skipped,
        percent=round(100.0 * (done + skipped) / n, 1) if n else 0.0,
        remaining_minutes=round(
            sum(s["seconds"] for s in steps if not s["state"]) / 60.0, 1),
        ready=len(ready))
    out["stats"] = stats
    return out


# ══════════════════════════════════════════════════════════════════════════
# ONE STEP
# ══════════════════════════════════════════════════════════════════════════
def step(conn: sqlite3.Connection, step_id: str, goal: str = "",
         limit: int = MOMENTS_IN_STEP) -> dict:
    """One concept in full: every video it covers and every moment to watch.

    The drill-down is a separate request on purpose. A plan carrying twenty-four
    passages for sixty steps is a megabyte of text nobody reads, and the five in
    the plan are enough to decide whether to open the step at all.
    """
    ensure_schema(conn)
    step_id = str(step_id or "")
    try:
        row = conn.execute(
            "SELECT id, kind, label, sub, weight, meta FROM graph_nodes "
            "WHERE id = ?", (step_id,)).fetchone()
    except sqlite3.Error as e:
        return {"ok": False, "note": f"the graph could not be read — {e}"}
    if not row:
        return {"ok": False, "note": "no such concept in the graph"}

    scope = _scope(conn, goal)
    scoped = set(scope.get("keys") or ())
    try:
        keys = [str(r[0])[2:] for r in conn.execute(
            "SELECT src FROM graph_edges WHERE dst = ? AND src LIKE 'v:%'",
            (step_id,))]
    except sqlite3.Error:
        keys = []
    inside = [k for k in keys if not scoped or k in scoped]
    weights = scope.get("weights") or {}
    ranked = sorted(inside, key=lambda k: -weights.get(k, 1.0))
    label = str(row[2] or "")
    moments = _moments(conn, label.lstrip("#"), ranked, limit)

    marks = progress(conn)
    mine = marks.get(step_id) or {}
    return {"ok": True, "id": step_id, "kind": row[1], "label": label,
            "group": row[3] or "", "degree": round(float(row[4] or 0.0), 2),
            "state": str(mine.get("state") or ""),
            "marked_at": mine.get("at") or 0,
            "goal": scope.get("goal", ""), "mode": scope.get("mode", "archive"),
            "videos_total": len(keys), "videos_in_scope": len(inside),
            "keys": ranked[:120], "moments": moments,
            "seconds": round(sum(m["seconds"] for m in moments), 1),
            "said": any(m["said"] for m in moments),
            "videos": _videos(conn, set(ranked[:120])
                              | {m["video_key"] for m in moments})}


def suggest(conn: sqlite3.Connection, limit: int = 14) -> list:
    """Goals worth offering: the concepts that cover the most of the archive."""
    try:
        rows = conn.execute(
            "SELECT n.label, n.kind, n.sub, COUNT(DISTINCT e.src) AS videos "
            "FROM graph_nodes n JOIN graph_edges e ON e.dst = n.id "
            f"WHERE n.kind IN ({','.join('?' * len(CONCEPT_KINDS))}) "
            "AND e.src LIKE 'v:%' "
            "GROUP BY n.id ORDER BY videos DESC, n.label LIMIT ?",
            list(CONCEPT_KINDS) + [int(limit)]).fetchall()
    except sqlite3.Error:
        return []
    return [{"label": r[0], "kind": r[1], "group": r[2] or "",
             "videos": int(r[3])} for r in rows]


# ══════════════════════════════════════════════════════════════════════════
# PROGRESS
# ══════════════════════════════════════════════════════════════════════════
def progress(conn: sqlite3.Connection) -> dict:
    """Every tick, keyed by step id. Missing table reads as no progress."""
    try:
        rows = conn.execute(
            "SELECT step_id, state, goal, at FROM roadmap_progress").fetchall()
    except sqlite3.Error:
        return {}
    return {str(r[0]): {"state": str(r[1] or ""), "goal": r[2] or "",
                        "at": float(r[3] or 0.0)} for r in rows}


def mark(conn: sqlite3.Connection, step_id: str, state: str,
         goal: str = "") -> dict:
    """Tick a step off, skip it, or clear it — an empty state clears.

    Nothing is validated against the current plan. A concept marked under one
    goal stays marked under the next, and a concept that has dropped out of the
    plan because the breadth cap moved keeps its tick for when it comes back.
    """
    ensure_schema(conn)
    step_id = str(step_id or "").strip()
    if not step_id:
        return {"ok": False, "note": "no step given"}
    state = str(state or "").strip().lower()
    if state not in ("done", "skip", ""):
        return {"ok": False, "note": f"unknown state “{state}”"}
    try:
        if not state:
            conn.execute("DELETE FROM roadmap_progress WHERE step_id = ?",
                         (step_id,))
        else:
            conn.execute(
                "INSERT INTO roadmap_progress(step_id, state, goal, at) "
                "VALUES(?,?,?,?) ON CONFLICT(step_id) DO UPDATE SET "
                "state = excluded.state, goal = excluded.goal, at = excluded.at",
                (step_id, state, (goal or "").strip(), time.time()))
        conn.commit()
    except sqlite3.Error as e:
        return {"ok": False, "note": f"could not store that — {e}"}
    return {"ok": True, "step_id": step_id, "state": state,
            "counts": counts(conn)}


def clear(conn: sqlite3.Connection) -> dict:
    """Forget every tick. The plan itself is derived, so nothing else is lost."""
    ensure_schema(conn)
    try:
        n = conn.execute("SELECT COUNT(*) FROM roadmap_progress").fetchone()[0]
        conn.execute("DELETE FROM roadmap_progress")
        conn.commit()
    except sqlite3.Error as e:
        return {"ok": False, "note": f"could not clear progress — {e}"}
    return {"ok": True, "cleared": int(n), "counts": counts(conn)}


def counts(conn: sqlite3.Connection) -> dict:
    out = {"done": 0, "skip": 0}
    try:
        for state, n in conn.execute(
                "SELECT state, COUNT(*) FROM roadmap_progress GROUP BY state"):
            out[str(state)] = int(n)
    except sqlite3.Error:
        pass
    return out
