"""Why a shard the channel refuses must not take the watermark with it.

The measured defect, from the 22 August session log:

    Shard bce94897-0001 not uploaded: Request Entity Too Large

`upload.call` does not retry that — a 4xx whose body is HTML rather than JSON is
the request's fault and the same bytes would fail the same way — and `_publish`
then advanced `shard_lo_id`, `shard_lo_vec`, `shard_lo_fvec` and `shard_lo_fmet`
off the export's own stats regardless, so nothing would ever export those rows
again. The file stayed on a Kaggle disk that does not survive the session. One
413 was permanent evidence loss.

What this pins:

  1. `export_shard(budget_bytes=N)` produces a file at or under N, and reports
     the ids it actually reached rather than the ids it was asked for.
  2. The header of a cut shard names the range the file really covers, so
     `note_shard` and any later reader are told the truth.
  3. Nothing is lost across the cut: exporting the remainder from the reported
     watermark and importing both files reconstructs every row.
  4. A cut shard omits coverage for videos whose evidence spilled past it, and
     ships it once the remainder goes — the one shape `import_shard` warns about
     is not reachable by the engine cutting its own shard.
  5. The budget never stalls: a budget smaller than one row still ships one row,
     so the watermark always advances.
  6. `budget_bytes=0` is byte-identical in content to the unbudgeted export this
     replaces.
  7. A held shard keeps its path, is handed back by `held_shards()`, stops being
     handed back once sent or once out of attempts, and is counted by the status
     panel.
  8. A shard the channel has taken does not stay on the disk. Nothing used to
     remove one: `mark_shard_sent` clears the `path` column without touching the
     file, `_publish` never unlinked its upload, and `intake.evict` walks
     `cache_dir` for working *directories* — so `shard_dir` was outside every
     mechanism the engine has for disk pressure. Measured in this archive:
     74 shard files over nine sessions, all 74 published, 248.7 MB, of which
     135.8 MB was the one session that cut 30. And under the floor `_publish`
     defers instead of discarding, because a held shard is the only copy of rows
     the watermark has stepped over.

Run from the repo root: python _t_shard_size.py
"""

import gzip
import io
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

FAILS = []
TMP = []


def check(name, good, detail=""):
    print(f"  {'PASS' if good else 'FAIL'}  {name}"
          + (f" — {detail}" if detail else ""))
    if not good:
        FAILS.append(name)


def workdir():
    d = tempfile.mkdtemp(prefix="vios_shardsize_")
    TMP.append(d)
    return d


def fresh_store(videos=6, claims_each=40, spaces=3, metrics=3):
    """A real Store with real evidence in all four watermarked tables.

    The blobs are deliberately random rather than zeros: gzip would compress a
    run of zeros to nothing and the budget would never be reached, which would
    make every assertion below pass for the wrong reason.

    One row per `(video, observer, space)` is all the uid allows for a
    frame-vector, and one per `(video, observer, name)` for a metric, so breadth
    comes from the space and metric names rather than from a loop count.
    """
    from vios.process.store import Store

    d = workdir()
    st = Store(os.path.join(d, "evidence.db"))
    obs = st.observer("caption", "test-model", "r1")
    rnd = random.Random(20260901)
    for v in range(videos):
        key = f"REEL{v:02d}"
        st.add_video(key, duration=30.0, url=f"https://x/{key}")
        st.set_shots(key, [{"t0": i * 3.0, "t1": i * 3.0 + 3.0}
                           for i in range(3)], "test")
        st.add_claims(key, obs, [
            {"channel": "caption", "kind": "line", "shot_idx": i % 3,
             "ordinal": i,
             # Long-ish, high-entropy text: a claim in this archive is a
             # sentence, not a token, and the ratio matters to the budget.
             "value": f"a person in a {os.urandom(8).hex()} jacket holding a "
                      f"{os.urandom(8).hex()} object, frame {i}",
             "confidence": 0.5 + (i % 4) / 10.0}
            for i in range(claims_each)])
        for s in range(spaces):
            space = f"clip{s}"
            for shot in range(3):
                st.add_vector(key, space,
                              [rnd.random() for _ in range(512)], obs,
                              shot_idx=shot)
            st.add_frame_vectors(key, space, list(range(24)),
                                 [[rnd.random() for _ in range(128)]
                                  for _ in range(24)], obs)
        for m in range(metrics):
            st.add_frame_metric(key, f"metric{m}", list(range(64)),
                                [rnd.random() for _ in range(64)], obs)
    return st, d


def read_shard(path):
    """The shard's header and its rows, by type."""
    head, rows = None, {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if head is None and d.get("_") == "vios-evidence-shard":
                head = d
                continue
            rows.setdefault(d.get("t"), []).append(d)
    return head, rows


def slack_of(path):
    """A computed upper bound on how far past its budget a shard may be.

    Two things are deliberately outside the budget: the row being written when
    the cut was noticed, and the whole coverage section. Both are measurable
    after the fact, and gzip cannot turn n bytes of input into more than about n
    bytes of output — so their uncompressed sizes bound the overshoot. Asserting
    `bytes <= budget` flat would be asserting something the design does not
    promise; asserting `bytes <= budget + this` is the promise.
    """
    ev, cov = 0, 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            n = len(line.encode("utf-8"))
            if '"t": "coverage"' in line or '"t":"coverage"' in line:
                cov += n
            else:
                ev = max(ev, n)
    return ev + cov + 1024


# ── 1-2. the cut, and a header that tells the truth about it ─────────────
def test_cut_is_bounded_and_honest():
    st, d = fresh_store()
    hi = (st.max_claim_id(), st.max_vector_id(),
          st.max_frame_vector_id(), st.max_frame_metric_id())

    whole = os.path.join(d, "whole.jsonl.gz")
    full = st.export_shard(whole, component="all")
    check("the unbudgeted export is not cut", full["cut"] is False,
          f"{full['bytes'] / 1024:.0f} KB, {full['claims']} claims, "
          f"{full['vectors']} vectors, {full['frame_vectors']} frame-vector "
          f"rows, {full['frame_metrics']} frame-metric rows")
    check("and it covers the whole range",
          (full["hi_id"], full["hi_vec"], full["hi_fvec"], full["hi_fmet"]) == hi,
          str(hi))

    # A budget a fifth of the whole: big enough that the prologue and several
    # thousand claims fit, small enough that the cut lands inside the evidence.
    budget = max(4096, full["bytes"] // 5)
    part = os.path.join(d, "part.jsonl.gz")
    cut = st.export_shard(part, component="all", budget_bytes=budget)

    check("a shard over budget reports itself cut", cut["cut"] is True,
          f"{cut['bytes']} bytes against a {budget}-byte budget")
    slack = slack_of(part)
    check("and it is inside the budget plus its stated overshoot",
          cut["bytes"] <= budget + slack,
          f"{cut['bytes']} <= {budget} + {slack} (one row + the coverage "
          f"section); {100.0 * cut['bytes'] / budget:.1f}% of budget")
    check("and it is not cut far too early",
          cut["bytes"] >= budget * 0.6,
          f"{cut['bytes']} of {budget} — a shard cut at a fraction of its "
          f"allowance would multiply the number of uploads for no reason")
    check("it stopped short of the range it was asked for",
          cut["hi_id"] < hi[0] or cut["hi_vec"] < hi[1]
          or cut["hi_fvec"] < hi[2] or cut["hi_fmet"] < hi[3],
          f"claims to {cut['hi_id']}/{hi[0]}, vectors to {cut['hi_vec']}/{hi[1]}")
    check("and it says what it was asked for, for the log",
          cut["want"].get("hi_id") == hi[0]
          and cut["want"].get("hi_fmet") == hi[3],
          str(cut["want"]))

    head, rows = read_shard(part)
    check("the header names the range the file really covers",
          (head["hi_id"], head["hi_vec"], head["hi_fvec"], head["hi_fmet"])
          == (cut["hi_id"], cut["hi_vec"], cut["hi_fvec"], cut["hi_fmet"]),
          f"header {head['hi_id']}/{head['hi_vec']}/{head['hi_fvec']}/"
          f"{head['hi_fmet']} vs stats {cut['hi_id']}/{cut['hi_vec']}/"
          f"{cut['hi_fvec']}/{cut['hi_fmet']} — `note_shard` records the "
          f"header's claim")
    check("the counts in the stats are the rows in the file",
          (len(rows.get("claim", [])), len(rows.get("vector", [])),
           len(rows.get("frame_vector", [])), len(rows.get("frame_metric", [])))
          == (cut["claims"], cut["vectors"], cut["frame_vectors"],
              cut["frame_metrics"]),
          f"{ {k: len(v) for k, v in rows.items()} }")
    check("claims travelled first",
          cut["claims"] > 0,
          f"{cut['claims']} claims — the cheap dense rows are not what a burst "
          f"of frame vectors is allowed to displace")
    st.conn.close()


# ── 3. nothing is lost across the cut ────────────────────────────────────
def test_remainder_loses_nothing():
    st, d = fresh_store()
    hi = (st.max_claim_id(), st.max_vector_id(),
          st.max_frame_vector_id(), st.max_frame_metric_id())
    whole = os.path.join(d, "whole.jsonl.gz")
    full = st.export_shard(whole, component="all")
    budget = max(4096, full["bytes"] // 5)

    # Publish the way `_publish` does: export against the watermark, then move
    # the watermark to what the export says it reached, and go round again.
    parts, lo, guard = [], (0, 0, 0, 0), 0
    while (lo[0] < hi[0] or lo[1] < hi[1] or lo[2] < hi[2] or lo[3] < hi[3]):
        guard += 1
        if guard > 200:
            check("the publish loop terminates", False,
                  f"still going after {guard} shards at {lo}")
            break
        p = os.path.join(d, f"part{guard:03d}.jsonl.gz")
        s = st.export_shard(p, lo[0], hi[0], lo[1], hi[1], "all",
                            lo[2], hi[2], lo[3], hi[3], budget_bytes=budget)
        nxt = (s["hi_id"], s["hi_vec"], s["hi_fvec"], s["hi_fmet"])
        if nxt == lo:
            check("every shard advances the watermark", False,
                  f"shard {guard} reached {nxt}, the same place it started — "
                  f"this is the stall the `wrote` guard in `full()` prevents")
            break
        parts.append((p, s))
        lo = nxt
    else:
        check("the publish loop terminates", True,
              f"{len(parts)} shards for a {full['bytes'] / 1024:.0f} KB export "
              f"at a {budget / 1024:.0f} KB budget")

    check("more than one shard was needed", len(parts) > 1, str(len(parts)))
    over = [(os.path.basename(p), s["bytes"], budget + slack_of(p))
            for p, s in parts if s["bytes"] > budget + slack_of(p)]
    check("every part is inside its budget plus overshoot", not over, str(over))
    check("the last part is not cut",
          bool(parts) and not parts[-1][1]["cut"],
          "the remainder fits, so the loop ends because the range is covered")

    # The whole point: replaying the parts into an empty database gives back
    # exactly what replaying the one big shard would have.
    def replay(files):
        from vios.process.store import Store

        target = Store(os.path.join(workdir(), "target.db"))
        for f in files:
            target.import_shard(f)
        got = {t: target.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
               for t in ("claim", "vector", "frame_vector", "frame_metric",
                         "video", "shot")}
        target.conn.close()
        return got

    one = replay([whole])
    many = replay([p for p, _ in parts])
    check("the parts replay to the same database as the whole",
          one == many, f"whole={one} parts={many}")
    check("and that is every row that was written",
          one["claim"] == full["claims"] and one["vector"] == full["vectors"],
          f"{one['claim']} claims, {one['vector']} vectors")
    st.conn.close()


# ── 4. coverage does not travel ahead of the evidence it describes ───────
def test_coverage_waits_for_its_evidence():
    from vios.process import coverage as C

    st, d = fresh_store()
    cov = C.Coverage(st.conn)
    cov.plan(["caption"])
    for v in range(6):
        cov.done(f"REEL{v:02d}", "caption", 12.0, claims=40)

    hi = (st.max_claim_id(), st.max_vector_id(),
          st.max_frame_vector_id(), st.max_frame_metric_id())
    whole = os.path.join(d, "whole.jsonl.gz")
    full = st.export_shard(whole, component="caption")
    _, rows = read_shard(whole)
    check("an uncut shard ships coverage for every video it touches",
          len(rows.get("coverage", [])) == 6,
          f"{len(rows.get('coverage', []))} rows, {full['coverage']} reported")

    budget = max(4096, full["bytes"] // 5)
    part = os.path.join(d, "part.jsonl.gz")
    cut = st.export_shard(part, component="caption", budget_bytes=budget)

    def spilled_after(s):
        return {r[0] for r in st.conn.execute(
            "SELECT video_key FROM claim WHERE id>? "
            "UNION SELECT video_key FROM vector WHERE id>? "
            "UNION SELECT video_key FROM frame_vector WHERE id>? "
            "UNION SELECT video_key FROM frame_metric WHERE id>?",
            (s["hi_id"], s["hi_vec"], s["hi_fvec"], s["hi_fmet"]))}

    # Swept across the whole range of budgets rather than at one guessed
    # fraction. The cut can land in any of the four tables, and where it lands
    # decides how many videos are wholly described by the file — so the
    # interesting case (some coverage ships, some is held back) only appears at
    # some fractions, and a single-budget test would either miss it or be tuned
    # to hit it. The invariant is asserted at every fraction; the selectivity is
    # asserted once, over all of them.
    partial, disjoint = [], []
    for frac in (0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 0.995):
        p = os.path.join(d, f"cov{int(frac * 1000)}.jsonl.gz")
        s = st.export_shard(p, component="caption",
                            budget_bytes=max(4096, int(full["bytes"] * frac)))
        _, rr = read_shard(p)
        shipped = {r["video_key"] for r in rr.get("coverage", [])}
        sp = spilled_after(s)
        if shipped & sp:
            disjoint.append((frac, sorted(shipped & sp)))
        if shipped and sp:
            partial.append((frac, len(shipped), len(sp)))
    check("no coverage row ever travels for a video cut in half",
          not disjoint,
          str(disjoint) or "eight budgets from a fifth to all of it; a `done` "
          "row beside half its claims is what `import_shard` warns about, and "
          "the engine must not be the thing that creates it")
    check("and the exclusion is selective, not a blanket silence",
          bool(partial),
          f"{partial} — (fraction, videos shipped, videos held back); a cut "
          f"that dropped every coverage row would satisfy the check above "
          f"while losing the reconcile that stops re-earned work")

    head, rows = read_shard(part)
    shipped = {r["video_key"] for r in rows.get("coverage", [])}
    spilled = spilled_after(cut)

    # And it is not lost — the shard carrying the remainder re-reads coverage.
    rest = os.path.join(d, "rest.jsonl.gz")
    st.export_shard(rest, cut["hi_id"], hi[0], cut["hi_vec"], hi[1], "caption",
                    cut["hi_fvec"], hi[2], cut["hi_fmet"], hi[3])
    _, rrows = read_shard(rest)
    later = {r["video_key"] for r in rrows.get("coverage", [])}
    check("the remainder shard carries the coverage the cut one held back",
          spilled <= later,
          f"held back {sorted(spilled)}, shipped later {sorted(later)}")

    from vios.process.store import Store
    target = Store(os.path.join(workdir(), "target.db"))
    target.import_shard(part)
    target.import_shard(rest)
    n = target.conn.execute(
        "SELECT COUNT(*) FROM coverage WHERE state='done'").fetchone()[0]
    check("and a restore over both ends up with all six done", n == 6, str(n))
    target.conn.close()
    st.conn.close()


# ── 5. the budget can be absurd and the watermark still moves ────────────
def test_never_stalls():
    st, d = fresh_store(videos=2, claims_each=6, spaces=1, metrics=1)
    tiny = os.path.join(d, "tiny.jsonl.gz")
    s = st.export_shard(tiny, component="all", budget_bytes=1)
    check("a one-byte budget still ships a row",
          s["claims"] + s["vectors"] + s["frame_vectors"] + s["frame_metrics"]
          >= 1,
          f"{s['claims']}/{s['vectors']}/{s['frame_vectors']}/"
          f"{s['frame_metrics']} — the prologue alone is over any such budget, "
          f"and a shard that ships nothing advances nothing")
    check("and it advances the watermark", s["hi_id"] > 0, str(s["hi_id"]))
    check("and it is honest about being cut", s["cut"] is True, str(s["cut"]))
    _, rows = read_shard(tiny)
    check("the file is readable", len(rows.get("claim", [])) == s["claims"],
          f"{len(rows.get('claim', []))} claim rows")
    st.conn.close()


# ── 6. the unbudgeted path is what it was ───────────────────────────────
def test_unbudgeted_is_unchanged():
    st, d = fresh_store(videos=3)
    a = os.path.join(d, "a.jsonl.gz")
    b = os.path.join(d, "b.jsonl.gz")
    sa = st.export_shard(a, component="all")
    sb = st.export_shard(b, component="all", budget_bytes=0)
    ha, ra = read_shard(a)
    hb, rb = read_shard(b)
    ha.pop("at", None)
    hb.pop("at", None)
    check("budget_bytes=0 is the same header", ha == hb, f"{ha} vs {hb}")
    check("and the same rows",
          {k: len(v) for k, v in ra.items()} == {k: len(v) for k, v in rb.items()},
          str({k: len(v) for k, v in ra.items()}))
    check("and reports cut=False with no `want`",
          sb["cut"] is False and sb["want"] == {}, str(sb))
    check("both cover the full range", sa["hi_id"] == sb["hi_id"] == st.max_claim_id())
    st.conn.close()


# ── 7. a held shard is findable, retryable and countable ────────────────
def test_held_shards():
    st, d = fresh_store(videos=2, claims_each=5, spaces=1, metrics=1)
    p = os.path.join(d, "held.jsonl.gz")
    stats = st.export_shard(p, component="all")

    st.note_shard("SITE-0001", "all", None, stats)
    held = st.held_shards()
    check("a shard noted with no message id is held",
          [h["shard_id"] for h in held] == ["SITE-0001"], str(held))
    check("and its path came with it", held[0]["path"] == p, str(held[0]["path"]))
    check("and one attempt is already counted",
          held[0]["attempts"] == 1,
          f"{held[0]['attempts']} — the send that failed was an attempt")

    n = st.held_shard_failed("SITE-0001")
    check("a failed retry is counted", n == 2, str(n))
    check("and it is still offered under the cap",
          len(st.held_shards(8, 6)) == 1, "6 attempts allowed, 2 spent")
    for _ in range(4):
        st.held_shard_failed("SITE-0001")
    check("out of attempts it is not offered any more",
          st.held_shards(8, 6) == [], "6 spent of 6")
    check("but it is still visible in `shards()`",
          any(s["shard_id"] == "SITE-0001" and not s["msg_id"]
              and s["path"] for s in st.shards()),
          "which is what the status panel's `held` count reads")

    st.mark_shard_sent("SITE-0001", 4242)
    check("once sent it is no longer held", st.held_shards() == [])
    row = [s for s in st.shards() if s["shard_id"] == "SITE-0001"][0]
    check("the message id is recorded and the path let go",
          row["msg_id"] == 4242 and not row["path"], str(row))

    # A published shard is never held, and a row naming a file that is gone is
    # not offered — a fresh session's `shard` table can be full of both.
    st.note_shard("SITE-0002", "all", 99, stats)
    gone = os.path.join(d, "gone.jsonl.gz")
    shutil.copyfile(p, gone)
    st.note_shard("SITE-0003", "all", None, {**stats, "path": gone})
    os.remove(gone)
    check("a published shard is not held and a vanished file is not offered",
          st.held_shards() == [],
          "the restore path replays rows whose files never existed here")
    st.conn.close()


# ── 8. a shard the channel has taken does not stay on the disk ───────────
def _bare_engine(shard_dir, store):
    """The few attributes the shard-file methods touch, and nothing else.

    Built by binding the real functions onto a stand-in rather than by
    constructing a `ProcessEngine`, because the code under test is then the code
    that ships: none of these three methods reads anything a real engine would
    have to probe a GPU or a channel for.
    """
    from vios.process.engine import ProcessEngine

    class Bare:
        def __init__(self):
            self.shard_dir = shard_dir
            self.store = store
            self.disk_floor_mb = 4_000
            self.index, self.partitions = 0, 1
            self.session = {"shards": 0}
            self.last_publish = 0.0
            self.since_publish = 7
            self._tg = None
            self._lock = threading.RLock()
            self._pub_lock = threading.RLock()
            self.logs = []

        def _log(self, text, level="info"):
            self.logs.append((level, str(text)))

        def _retry_held(self, store):
            return 0

        def said(self, needle):
            return [t for _, t in self.logs if needle in t]

    for name in ("_drop_shard_file", "_reclaim_shards", "_publish"):
        setattr(Bare, name, getattr(ProcessEngine, name))
    return Bare()


def _put(shard_dir, name, size=2048):
    p = os.path.join(shard_dir, name)
    with open(p, "wb") as f:
        f.write(os.urandom(size))
    return p


def test_published_shard_is_not_kept():
    from vios.process import intake

    st, d = fresh_store(videos=2, claims_each=5, spaces=1, metrics=1)
    shard_dir = os.path.join(d, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    eng = _bare_engine(shard_dir, st)

    # Four files in one directory: a published shard, a held one, a stage
    # bundle and one of its parts. Exactly one of them is redundant.
    up = _put(shard_dir, intake.shard_name("SITE-0001"))
    held = _put(shard_dir, intake.shard_name("SITE-0002"))
    bundle = _put(shard_dir, "vios-stage-SITE-language-0001.tar.gz")
    part = _put(shard_dir, "vios-stage-SITE-language-0001.tar.gz.part000")
    stats = st.export_shard(os.path.join(d, "probe.jsonl.gz"), component="all")
    st.note_shard("SITE-0001", "all", 5001, stats)
    st.note_shard("SITE-0002", "all", None, {**stats, "path": held})

    n = eng._drop_shard_file(up, "SITE-0001")
    check("_drop_shard_file reports the bytes it freed", n == 2048, str(n))
    check("and the file is gone", not os.path.exists(up))
    check("a file that is not there frees nothing and says nothing",
          eng._drop_shard_file(up, "SITE-0001") == 0 and not eng.logs,
          str(eng.logs))

    _put(shard_dir, intake.shard_name("SITE-0001"), 4096)
    back = eng._reclaim_shards()
    check("the sweep collects a shard whose row carries a message id",
          back["removed"] == 1 and back["bytes"] == 4096, str(back))
    check("and leaves the held one, which is the only copy of its rows",
          os.path.exists(held))
    check("and does not touch a stage bundle or one of its parts",
          os.path.exists(bundle) and os.path.exists(part),
          "matched by name against published rows, not by looking shard-shaped")
    check("running it again is a no-op",
          eng._reclaim_shards()["removed"] == 0)

    n_held, held_bytes = st.held_shard_bytes()
    check("held_shard_bytes measures the file, not the export column",
          (n_held, held_bytes) == (1, 2048),
          f"{n_held} file(s), {held_bytes} bytes on disk; the column says "
          f"{stats['bytes']}, which is what the export produced")

    # ── under the floor, defer; and defer without deleting anything ─────
    real_free = intake.free_mb
    intake.free_mb = lambda _p: 10          # MB, far under the 4,000 floor
    keys = ("shard_lo_id", "shard_lo_vec", "shard_lo_fvec", "shard_lo_fmet",
            "shard_seq")
    try:
        eng.logs.clear()
        before = {k: st.get_meta(k, "0") for k in keys}
        out = eng._publish("floor-test")
        check("under the floor no shard is written", out == "", repr(out))
        check("and no watermark moved",
              before == {k: st.get_meta(k, "0") for k in keys},
              str(before))
        check("and the rows are still counted as unpublished",
              eng.since_publish == 7, str(eng.since_publish))
        check("the clock moved, so the warning is not repeated every tick",
              eng.last_publish > 0)
        said = eng.said("Shard deferred")
        check("the line names the floor and what is holding the disk",
              said and "4000 MB floor" in said[0]
              and "1 held shard(s)" in said[0], str(said))
        check("and says the evidence is still safe, because it is",
              said and "watermark has not moved" in said[0], str(said))
        check("nothing was deleted to make room",
              os.path.exists(held),
              "the opposite trade from a stage bundle: no rebuild exists")
        check("and no new shard file was written either",
              sorted(f for f in os.listdir(shard_dir)
                     if f.startswith(intake.SHARD_PREFIX))
              == [os.path.basename(held)],
              str(sorted(os.listdir(shard_dir))))
    finally:
        intake.free_mb = real_free

    # ── no channel: the file stays, because it is the only copy ─────────
    eng.logs.clear()
    sid = eng._publish("no-channel")
    kept = os.path.join(shard_dir, intake.shard_name(sid))
    check("with room but no channel a shard is still exported", bool(sid),
          str(sid))
    check("and its file stays on the disk", os.path.exists(kept))
    check("which is what `held_shards` then offers",
          [h["shard_id"] for h in st.held_shards()] == ["SITE-0002", sid],
          str([h["shard_id"] for h in st.held_shards()]))

    # ── with a channel: it goes up, and then it goes away ──────────────
    class FakeTg:
        token, channel = "x", "@c"

        def __init__(self):
            self.sent = []

        def send_document(self, path, caption, file_name=None):
            self.sent.append((os.path.basename(path), os.path.getsize(path)))
            return {"message_id": 7007}

    obs = st.observer("caption", "test-model", "r1")
    st.add_claims("REEL00", obs, [{"channel": "caption", "kind": "line",
                                   "shot_idx": 0, "ordinal": 99,
                                   "value": "one more row to publish"}])
    eng._tg = FakeTg()
    eng.logs.clear()
    sid2 = eng._publish("published")
    check("a shard with room and a channel is published", bool(sid2),
          str(sid2))
    check("the uploader was handed a real file",
          len(eng._tg.sent) == 1 and eng._tg.sent[0][1] > 0,
          str(eng._tg.sent))
    check("and the local copy is gone once the row carries its message id",
          not os.path.exists(os.path.join(shard_dir,
                                          intake.shard_name(sid2))),
          "135.8 MB of these used to sit here for the rest of the session")
    row = [s for s in st.shards() if s["shard_id"] == sid2][0]
    check("the row is what makes deleting it safe",
          row["msg_id"] == 7007 and not row["path"], str(row))
    check("and the two held files are untouched by any of it",
          os.path.exists(held) and os.path.exists(kept),
          "neither of them is in the channel")
    st.conn.close()


# ── the engine's side of it, on the source ───────────────────────────────
def test_engine_wiring():
    src = open("vios/process/engine.py", encoding="utf-8").read()
    check("the budget is a named constant with the cap in its reasoning",
          "SHARD_BUDGET_BYTES = 40 * 1024 * 1024" in src
          and "Request Entity Too Large" in src)
    check("_publish passes it to export_shard",
          "budget_bytes=SHARD_BUDGET_BYTES" in src)
    check("_publish retries held shards before exporting a new one",
          src.find("self._retry_held(store)") <
          src.find('lo_id = int(store.get_meta("shard_lo_id"'),
          "older evidence goes first; the channel is read back in order")
    check("a cut forces another publish tick",
          'if stats.get("cut"):\n                    ' in src
          and "self.since_publish = 1" in src,
          "otherwise the remainder waits for a video that may never come")
    check("the status panel counts held shards", '"held": sum(' in src)
    check("and how much disk they are", '"held_mb": round(' in src)
    check("a published shard's file is dropped after the watermark, not before",
          src.find("store.checkpoint()\n\n            # Only now is the local")
          < src.find("self._drop_shard_file(path, sid)"),
          "the row is what makes the file redundant, so the row goes first")
    check("the floor is checked before a 40 MB copy is written",
          src.find("free = intake.free_mb(self.shard_dir)")
          < src.find("budget_bytes=SHARD_BUDGET_BYTES"))
    check("and it defers rather than discarding held evidence",
          "Shard deferred" in src and "_discard" not in src.split(
              "Shard deferred")[0].rsplit("def _publish", 1)[-1],
          "no delete between the floor check and the deferral")
    ui = open("process_ui.html", encoding="utf-8").read()
    check("and the tab shows the count", "dHeld" in ui and "held locally" in ui)
    check("with the megabytes beside it", "held_mb" in ui)


def main():
    print("the cut, and a header that tells the truth about it")
    test_cut_is_bounded_and_honest()
    print("nothing is lost across the cut")
    test_remainder_loses_nothing()
    print("coverage waits for its evidence")
    test_coverage_waits_for_its_evidence()
    print("the budget never stalls")
    test_never_stalls()
    print("the unbudgeted path is unchanged")
    test_unbudgeted_is_unchanged()
    print("a held shard is findable, retryable and countable")
    test_held_shards()
    print("a shard the channel has taken does not stay on the disk")
    test_published_shard_is_not_kept()
    print("the engine's side of it")
    test_engine_wiring()

    for d in TMP:
        shutil.rmtree(d, ignore_errors=True)
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} — " + "; ".join(FAILS))
        return 1
    print("SHARD SIZE OK — a shard is cut before the channel refuses it, the "
          "watermark advances only over what shipped, a held file is retried "
          "and counted instead of quietly dying with the session, and a "
          "published one is deleted instead of holding the disk until it does")
    return 0


if __name__ == "__main__":
    sys.exit(main())
