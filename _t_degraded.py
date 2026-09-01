"""A pass that answered without one of its inputs, and the ladder it never had.

The measured defect. `_run_pass` computed the soft inputs it was missing, logged
one line about them at `warn`, and then called `cov.done(...)` — the identical
call a complete run makes. `done` is the one state with no way back:
`revive_failed` reconsiders `failed`, `reclaim_unavailable` reconsiders
`skipped`, and `reconcile` explicitly refuses to touch `done`. So the thin
reading was permanent, and the only trace of it died with the notebook session.

In `~/VIOS-Data/atlas.db`, measured from `claim.observer_id` and `created_at`
because that archive was restored from shards written before coverage travelled
and has no coverage table at all:

  * `keyphrase` wrote claims for all 30 videos. 28 of them hold no OCR evidence
    whatsoever — it read no on-screen text on any of them.
  * On the 2 videos that do have OCR, the 230 OCR claims were written 559,942–
    559,994 s (6.5 days) *after* `keyphrase` and `hook` had already been marked
    done. Reclaiming OCR produced text that nothing was ever going to read.
  * Six of the eight soft-consuming passes have not run yet, so the gap widens
    the moment the language wave does.

What this pins, in the order the fix depends on it:

  1. `marker()` normalises an absence to one deterministic string.
  2. `done()` records it, and — the direction that matters — clears it.
  3. `_observer_params` splits the identity, and an *empty* absence returns
     today's exact dict, so no observer id already in an archive changes. That is
     the whole safety argument for touching evidence identity.
  4. Why it has to: a claim uid excludes the value and every insert is
     `OR IGNORE`, so under one id a re-run's better answer is silently dropped.
  5. `reoffer_degraded` hands the row back — only when the missing input has
     finished *and* written something, only for passes this session can run, and
     it terminates.
  6. `INSTR`, not `LIKE`: `_` in a component id is not a wildcard.
  7. `reconcile` carries the marker, so a restore cannot erase it, and still
     accepts the 6-tuples written before it did.
  8. `degraded_observers` reproduces the ids a thin run wrote, from the registry
     alone — the restore case has no coverage table to read.
  9. The column travels in a shard, and `_settled_coverage` no longer ships an
     empty coverage section when the local table predates a migration.

Run from the repo root: python _t_degraded.py
"""
import io
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

# Before any vios/atlas import: a Store resolves its cache and home at import
# time, and a test must not write into the real archive on this laptop.
_BASE = tempfile.mkdtemp(prefix="vios_degraded_base_")
os.environ["VIOS_BASE_DIR"] = _BASE
os.environ["ATLAS_HOME"] = os.path.join(_BASE, "atlas")
os.environ["ATLAS_CACHE_DIR"] = os.path.join(_BASE, "cache")
os.environ["VIOS_OMNI"] = "0"
os.makedirs(os.environ["ATLAS_HOME"], exist_ok=True)

FAILS = []


def check(name, good, detail=""):
    print(f"  {'PASS' if good else 'FAIL'}  {name}"
          + (f" — {detail}" if detail else ""))
    if not good:
        FAILS.append(name)


def fresh_cov(n=4, **kw):
    """A real Coverage over a real database, with `n` videos."""
    from vios.process import coverage as C

    path = os.path.join(tempfile.mkdtemp(prefix="vios_degraded_"), "a.db")
    conn = sqlite3.connect(path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE video(video_key TEXT PRIMARY KEY, "
                 "partition INTEGER DEFAULT 0)")
    conn.executemany("INSERT INTO video(video_key) VALUES (?)",
                     [(f"REEL{i:02d}",) for i in range(n)])
    conn.commit()
    return C, C.Coverage(conn, **kw), conn


# ── 1. the marker ─────────────────────────────────────────────────────────
def test_marker():
    from vios.process.coverage import marker

    check("sorted, so two identical runs agree",
          marker(["ocr", "caption"]) == marker(["caption", "ocr"]) ==
          "caption,ocr",
          "the string is compared, shipped and hashed; claim order would make "
          "one absence look like two")
    check("empty is empty", marker(()) == "" and marker(None) == "",
          "and `done` turns that into NULL, which is what 'had everything' is")
    check("blanks and Nones drop out", marker([None, "", "ocr"]) == "ocr")
    check("commas are stripped", marker(["a,x", "b"]) == "ax,b",
          "the column is a comma-joined list, so a name containing one would "
          "make the membership test wrong; none does today")
    check("idempotent on its own output",
          marker(marker(["b", "a"]).split(",")) == "a,b")


# ── 2. done records it, and clears it ─────────────────────────────────────
def test_done_records_and_clears():
    C, cov, conn = fresh_cov()
    cov.plan(["keyphrase", "ocr"])

    def col(key):
        return conn.execute("SELECT degraded FROM coverage WHERE video_key=? "
                            "AND component='keyphrase'", (key,)).fetchone()[0]

    cov.done("REEL00", "keyphrase", 3.0, claims=20, degraded=["ocr"])
    cov.done("REEL01", "keyphrase", 3.0, claims=20)
    check("a thin run is recorded as thin", col("REEL00") == "ocr")
    check("a complete run records NULL, not ''", col("REEL01") is None,
          "so `degraded IS NOT NULL AND degraded<>''` is one test, not two")

    # The clearing direction is the one that makes the ladder terminate.
    cov.done("REEL00", "keyphrase", 4.0, claims=31, degraded=())
    check("a complete re-run clears the marker", col("REEL00") is None,
          "a marker that only accumulated would re-offer a row forever")
    row = conn.execute("SELECT claims, seconds FROM coverage WHERE "
                       "video_key='REEL00' AND component='keyphrase'").fetchone()
    check("and the re-run's own counts replace the thin ones",
          row["claims"] == 31 and abs(row["seconds"] - 4.0) < 0.01, dict(row))
    conn.close()


# ── 3. the identity split, and why it is needed ───────────────────────────
def test_observer_identity():
    from vios.process import registry
    from vios.process.engine import _observer_params
    from vios.process.store import observer_id as oid_of

    comp = registry.BY_ID["keyphrase"]
    base = dict(comp.params or {})

    same = _observer_params(comp.params, ())
    check("no absence means today's exact params",
          same == base and _observer_params(comp.params, None) == base,
          "so every observer id already written stays valid — this is the "
          "whole safety argument for touching evidence identity")
    check("and that is bit-identical as an id",
          oid_of("keyphrase", comp.model or comp.family, comp.revision, same)
          == oid_of("keyphrase", comp.model or comp.family, comp.revision,
                    comp.params),
          "a complete run is unaffected by this change")

    thin = _observer_params(comp.params, ["ocr"])
    check("an absence is recorded in the params",
          thin.get("without") == ["ocr"] and "without" not in base,
          f"{thin} — and the input dict is not mutated")
    ids = {oid_of("keyphrase", comp.model or comp.family, comp.revision, p)
           for p in (same, thin,
                     _observer_params(comp.params, ["ocr", "transcribe"]))}
    check("and three different absences are three different observers",
          len(ids) == 3, f"{len(ids)} distinct ids")
    check("order of the absence does not matter",
          _observer_params(comp.params, ["b", "a"])
          == _observer_params(comp.params, ["a", "b"]),
          "sorted inside, so the id does not depend on registry order")


def test_why_identity_must_split():
    """Under one observer id the re-run's better answer is silently discarded.

    Not a theory about the schema: `_uid` deliberately leaves the *value* out of
    the hash so a shard can be replayed twice without doubling anything, and
    `add_claims` inserts `OR IGNORE`. So a re-offered pass writing under the id
    it used the first time collides on every uid, `add_claims` reports 0, and
    `done(claims=0)` then overwrites the count that was true. A re-offer without
    the identity split is worse than the gap it closes.
    """
    from vios.process import store as S

    path = os.path.join(tempfile.mkdtemp(prefix="vios_uid_"), "ev.db")
    st = S.Store(path)
    st.add_video(video_key="REEL00", url="https://instagram.com/p/REEL00/",
                 uploader="x", duration=None, width=1080, height=1920,
                 bytes=1, taken_at=None, meta={})
    st.set_shots("REEL00", [{"t0": 0.0, "t1": 4.0}], "pyscenedetect")
    one = st.observer("keyphrase", "gemma", "v1", {})
    n1 = st.add_claims("REEL00", one, [
        {"channel": "concept", "kind": "keyphrase", "shot_idx": 0,
         "value": "garlic butter steak"}])
    n2 = st.add_claims("REEL00", one, [
        {"channel": "concept", "kind": "keyphrase", "shot_idx": 0,
         "value": "garlic butter steak, 54C, medium rare"}])
    check("a better answer under the same observer is dropped",
          n1 == 1 and n2 == 0,
          "INSERT OR IGNORE on a uid that excludes the value — this is the "
          "reason the fix had to reach evidence identity")
    kept = st.conn.execute("SELECT value FROM claim WHERE video_key='REEL00'"
                           ).fetchone()[0]
    check("and the thin value is what stays", kept == "garlic butter steak",
          repr(kept))

    two = st.observer("keyphrase", "gemma", "v1", {"without": ["ocr"]})
    check("the absence produces a different observer row", two != one)
    n3 = st.add_claims("REEL00", two, [
        {"channel": "concept", "kind": "keyphrase", "shot_idx": 0,
         "value": "garlic butter steak, 54C, medium rare"}])
    check("and under it the fuller answer lands", n3 == 1,
          "both readings kept, which is the append-only rule doing its job")
    st.close()


# ── 4. the ladder ─────────────────────────────────────────────────────────
def test_reoffer():
    C, cov, conn = fresh_cov()
    plan = ["keyphrase", "ocr"]
    cov.plan(plan)

    # REEL00: thin, and OCR has since produced text     → re-offer
    # REEL01: thin, and OCR finished having found none   → do not
    # REEL02: thin, and OCR has not run at all           → do not
    # REEL03: complete                                   → never
    cov.done("REEL00", "keyphrase", 3.0, claims=20, degraded=["ocr"])
    cov.done("REEL01", "keyphrase", 3.0, claims=20, degraded=["ocr"])
    cov.done("REEL02", "keyphrase", 3.0, claims=20, degraded=["ocr"])
    cov.done("REEL03", "keyphrase", 3.0, claims=25)
    cov.done("REEL01", "ocr", 90.0, claims=0)

    t = cov.thin()
    check("thin() counts the rows that answered without an input",
          t["rows"] == 3 and t["per_component"] == {"keyphrase": 3}, str(t))
    check("and none of them is fixable yet", t["waiting"] == 0,
          "no named input has written anything")
    got = cov.reoffer_degraded(plan)
    check("so nothing is re-offered", got["reoffered"] == 0, str(got))

    cov.done("REEL00", "ocr", 90.0, claims=230)
    check("evidence arriving makes exactly one row fixable",
          cov.thin()["waiting"] == 1, str(cov.thin()))
    got = cov.reoffer_degraded(plan)
    check("and that row is handed back",
          got["reoffered"] == 1 and got["components"] == {"keyphrase": 1},
          str(got))
    state = dict(conn.execute(
        "SELECT video_key, state FROM coverage WHERE component='keyphrase'"
    ).fetchall())
    check("only that one moved",
          state == {"REEL00": C.QUEUED, "REEL01": C.DONE,
                    "REEL02": C.DONE, "REEL03": C.DONE}, str(state))
    check("an input that finished with nothing is not a reason to re-run",
          state["REEL01"] == C.DONE,
          "there was no on-screen text; re-reading would cost a GPU pass to "
          "produce the identical answer")

    row = conn.execute("SELECT attempts, claims, degraded FROM coverage WHERE "
                       "video_key='REEL00' AND component='keyphrase'").fetchone()
    check("the re-offered row is a real retry", row["attempts"] == 0)
    check("its counts are left alone until the re-run finishes",
          row["claims"] == 20 and row["degraded"] == "ocr",
          "they are still the record of what this row actually produced")
    check("and it is claimable now",
          "REEL00" in cov.candidates(["keyphrase"], 10),
          "before this column existed no mechanism could reach it")

    check("re-offering twice moves nothing the second time",
          cov.reoffer_degraded(plan)["reoffered"] == 0,
          "the row is already queued, and this selects only `done`")

    # And it terminates: the re-run now sees OCR, so it names nothing.
    cov.done("REEL00", "keyphrase", 4.0, claims=44, degraded=())
    check("after the re-run the row is no longer thin",
          cov.thin()["rows"] == 2 and
          cov.reoffer_degraded(plan)["reoffered"] == 0,
          "the marker shrank, so this stops selecting it — a second re-offer "
          "needs a second input to arrive")
    conn.close()


def test_reoffer_is_scoped():
    C, cov, conn = fresh_cov()
    cov.plan(["keyphrase", "concepts", "ocr"])
    cov.done("REEL00", "keyphrase", 3.0, claims=20, degraded=["ocr"])
    cov.done("REEL00", "concepts", 3.0, claims=20, degraded=["ocr"])
    cov.done("REEL00", "ocr", 90.0, claims=230)

    got = cov.reoffer_degraded(["keyphrase", "ocr"])
    check("only passes this session can run are re-offered",
          got["reoffered"] == 1 and got["components"] == {"keyphrase": 1},
          f"{got} — a usable answer must not become a queued row that nothing "
          f"will ever claim")
    check("the pass that is not selected keeps its answer",
          conn.execute("SELECT state FROM coverage WHERE component='concepts'"
                       ).fetchone()[0] == C.DONE,
          "and stays thin, so it is re-offered whenever it is next enabled")
    check("an empty selection re-offers nothing",
          cov.reoffer_degraded([]) == {"reoffered": 0, "components": {}},
          "not 'everything', which is what a bare IN () would have meant")
    check("no selection at all still means every component",
          cov.reoffer_degraded()["reoffered"] == 1,
          "`concepts`, the one left behind above")
    conn.close()


def test_instr_not_like():
    """`_` in a component id must not behave as a wildcard.

    `LIKE '%,'||d.component||',%'` would have `o_r` match a row that went
    without `ocr`, and this test decides whether to spend a GPU pass again.
    Real ids in the registry contain `-`, which LIKE is safe with — so the
    hazard would sit in the code unexercised until the first underscored id.
    """
    C, cov, conn = fresh_cov()
    cov.plan(["keyphrase", "ocr"])
    conn.execute("INSERT OR IGNORE INTO coverage(video_key,component) "
                 "VALUES('REEL00','o_r')")
    conn.commit()
    cov.done("REEL00", "keyphrase", 3.0, claims=20, degraded=["ocr"])
    cov.done("REEL00", "o_r", 1.0, claims=7)
    check("a one-character-wildcard match does not count as the input arriving",
          cov.reoffer_degraded()["reoffered"] == 0 and
          cov.thin()["waiting"] == 0,
          "`o_r` finished with claims; `ocr` did not")
    cov.done("REEL00", "ocr", 90.0, claims=230)
    check("and the real input still does",
          cov.reoffer_degraded()["reoffered"] == 1)
    conn.close()


def test_prefix_is_not_membership():
    """`ocr` must not match a row that went without `ocr-lite`."""
    C, cov, conn = fresh_cov()
    cov.plan(["keyphrase"])
    for extra in ("ocr-lite", "ocr"):
        conn.execute("INSERT OR IGNORE INTO coverage(video_key,component) "
                     "VALUES(?,?)", ("REEL00", extra))
    conn.commit()
    cov.done("REEL00", "keyphrase", 3.0, claims=20, degraded=["ocr-lite"])
    cov.done("REEL00", "ocr", 90.0, claims=230)
    check("a prefix of the missing name is not the missing name",
          cov.reoffer_degraded()["reoffered"] == 0,
          "the list is comma-wrapped on both sides for exactly this")
    cov.done("REEL00", "ocr-lite", 90.0, claims=5)
    check("the named one is", cov.reoffer_degraded()["reoffered"] == 1)
    conn.close()


def test_multiple_absences():
    """Two inputs missing: one arriving re-offers, and the re-run narrows it."""
    C, cov, conn = fresh_cov()
    cov.plan(["keyphrase", "ocr", "transcribe"])
    cov.done("REEL00", "keyphrase", 3.0, claims=9,
             degraded=["ocr", "transcribe"])
    cov.done("REEL00", "transcribe", 60.0, claims=40)
    check("one of two arriving is enough to re-offer",
          cov.reoffer_degraded()["reoffered"] == 1)
    # The re-run has speech now but still no text.
    cov.done("REEL00", "keyphrase", 5.0, claims=30, degraded=["ocr"])
    check("and the marker narrows to what is still missing",
          conn.execute("SELECT degraded FROM coverage WHERE "
                       "component='keyphrase'").fetchone()[0] == "ocr")
    check("which does not re-offer again on its own",
          cov.reoffer_degraded()["reoffered"] == 0,
          "the arrived input is no longer named; a second re-offer needs a "
          "second arrival")
    cov.done("REEL00", "ocr", 90.0, claims=230)
    check("the second arrival re-offers once more",
          cov.reoffer_degraded()["reoffered"] == 1,
          "bounded by the registry: at most four soft inputs on any pass")
    conn.close()


def test_matrix_and_stage_report():
    # Two videos, so "the stage is complete" is reachable and the claim that a
    # complete stage can still be thin is actually being tested.
    C, cov, conn = fresh_cov(2)
    cov.plan(["keyphrase", "ocr"])
    cov.done("REEL00", "keyphrase", 3.0, claims=20, degraded=["ocr"])
    cov.done("REEL01", "keyphrase", 3.0, claims=25)
    row = [m for m in cov.matrix() if m["component"] == "keyphrase"][0]
    check("the matrix reports thin beside done",
          row["done"] == 2 and row["thin"] == 1, str(row))

    cov.reoffer_degraded()          # nothing arrived; still 1 thin, still done
    cov.done("REEL00", "ocr", 90.0, claims=230)
    cov.reoffer_degraded()
    row = [m for m in cov.matrix() if m["component"] == "keyphrase"][0]
    check("a re-offered row is not counted as a thin answer",
          row["done"] == 1 and row["thin"] == 0,
          f"{row} — it carries the marker on purpose, but its answer is "
          f"currently being re-earned")

    from vios.process import registry
    stage = registry.BY_ID["keyphrase"].stage
    rep = cov.stage_report(stage, ["keyphrase"])
    per = {e["component"]: e for e in rep["per_component"]}
    check("stage_report carries thin per component and in the totals",
          rep.get("thin") == 0 and per["keyphrase"]["thin"] == 0
          and not rep["complete"], str(rep.get("thin")))
    cov.done("REEL00", "keyphrase", 4.0, claims=22, degraded=["ocr"])
    rep = cov.stage_report(stage, ["keyphrase"])
    per = {e["component"]: e for e in rep["per_component"]}
    check("and a complete stage can still report a thin answer",
          rep["complete"] and rep["thin"] == 1 and per["keyphrase"]["thin"] == 1,
          "the bundle is what gets read months later, with no session left — "
          "`complete` alone was the whole sentence for 28 of 30 videos")
    conn.close()


# ── 5. reconcile, both directions ─────────────────────────────────────────
def test_reconcile_carries_the_marker():
    C, cov, conn = fresh_cov(2)
    cov.plan(["keyphrase"])
    # The 7th field, as `reconcile_now` now sends it.
    out = cov.reconcile([("REEL00", "keyphrase", "obs-thin", 20, 0, 0.0,
                          ["ocr"]),
                         ("REEL01", "keyphrase", "obs-full", 25, 0, 0.0, ())])
    check("reconcile marks both done", out["rows"] == 2, str(out))
    got = dict(conn.execute("SELECT video_key, degraded FROM coverage").fetchall())
    check("and the thin one comes back thin",
          got == {"REEL00": "ocr", "REEL01": None},
          f"{got} — the laptop archive has no coverage table at all, so every "
          f"row in it arrives through here")
    check("so it is visible to the ladder",
          cov.thin()["rows"] == 1, str(cov.thin()))

    # Back-compatibility: the 6-tuple every earlier caller wrote.
    C2, cov2, conn2 = fresh_cov()
    cov2.plan(["keyphrase"])
    out2 = cov2.reconcile([("REEL00", "keyphrase", "obs", 20, 0, 0.0)])
    check("a 6-tuple still reconciles",
          out2["rows"] == 1 and
          conn2.execute("SELECT degraded FROM coverage").fetchone()[0] is None,
          "no caller is obliged to know about the column")
    conn.close()
    conn2.close()


def test_degraded_observers():
    """The ids a thin run wrote, reproduced without reading the coverage table.

    The case that matters is the one with nothing to read: a Kaggle container
    starts empty, replays shards into the evidence tables, and asks which
    observers it should recognise. Evidence under an unrecognised id is treated
    as a superseded revision and ignored — so the row stays `queued`, the pass
    re-runs, and every claim collides on a uid that is already there.
    """
    from vios.process import registry
    from vios.process.engine import ProcessEngine, _observer_params
    from vios.process.store import observer_id as oid_of

    sel = ["keyphrase", "hook", "ocr", "transcribe"]
    got = ProcessEngine.degraded_observers(_Stub(), sel)
    comp = registry.BY_ID["keyphrase"]

    want = oid_of("keyphrase", comp.model or comp.family, comp.revision,
                  _observer_params(comp.params, ["ocr"]))
    check("the id a keyphrase run without ocr wrote is recognised",
          got.get(want) == ("keyphrase", ["ocr"]), str(got.get(want)))
    want2 = oid_of("keyphrase", comp.model or comp.family, comp.revision,
                   _observer_params(comp.params, ["caption", "ocr"]))
    check("and so is every combination of its soft inputs",
          got.get(want2) == ("keyphrase", ["caption", "ocr"]),
          str(got.get(want2)))
    expect = sum((1 << len(registry.BY_ID[c].soft)) - 1 for c in sel)
    check("one id per non-empty subset, and the registry bounds it",
          len(got) == expect,
          f"{len(got)} ids for {len(sel)} passes — soft inputs are at most four")
    check("a pass with no soft inputs contributes none",
          not any(cid == "ocr" for cid, _ in got.values()),
          "ocr reads nothing soft, so it has no thin reading to recognise")

    complete = oid_of("keyphrase", comp.model or comp.family, comp.revision,
                      comp.params)
    check("and the complete id is never in here",
          complete not in got,
          "that one is `expected_observers`, and the two must not collide")


class _Stub:
    """Just enough engine for `degraded_observers`: a selection and a coverage.

    `degraded_observers` reads `self.coverage.degraded_variants()` as a
    supplement and nothing else off the instance, so an unbound call with this
    is the honest way to test the derivation without booting a sweep.
    """
    selected: list = []

    class coverage:                                  # noqa: N801
        @staticmethod
        def degraded_variants():
            return []


# ── 6. it travels ─────────────────────────────────────────────────────────
def test_shard_round_trip():
    from vios.process import coverage as C
    from vios.process import store as S

    base = tempfile.mkdtemp(prefix="vios_shard_")
    src = S.Store(os.path.join(base, "a.db"))
    src.add_video(video_key="REEL00", url="https://instagram.com/p/REEL00/",
                  uploader="x", duration=None, width=1080, height=1920,
                  bytes=1, taken_at=None, meta={})
    src.set_shots("REEL00", [{"t0": 0.0, "t1": 4.0}], "pyscenedetect")
    cov = C.Coverage(src.conn)
    cov.plan(["keyphrase"], ["REEL00"])
    oid = src.observer("keyphrase", "gemma", "v1", {"without": ["ocr"]})
    src.add_claims("REEL00", oid, [
        {"channel": "concept", "kind": "keyphrase", "shot_idx": 0,
         "value": "steak"}])
    cov.done("REEL00", "keyphrase", 3.0, claims=1, observer_id=oid,
             degraded=["ocr"])

    shard = os.path.join(base, "s.jsonl.gz")
    stats = src.export_shard(shard, 0, src.max_claim_id(), 0, 0, "gpu")
    check("the shard carries the coverage row", stats.get("coverage") == 1,
          str(stats))

    dst = S.Store(os.path.join(base, "b.db"))
    res = dst.import_shard(shard)
    check("and the target replays it", res.get("coverage") == 1, str(res))
    got = dst.conn.execute("SELECT state, degraded FROM coverage WHERE "
                           "video_key='REEL00'").fetchone()
    check("with the marker intact", tuple(got) == ("done", "ocr"),
          f"{tuple(got)} — a thin reading arriving somewhere else looking "
          f"complete is the whole defect this column ends")
    src.close()
    dst.close()


def test_export_survives_an_older_table():
    """A coverage table that predates a migration must still ship what it has.

    The export named every column in `_COVERAGE_COLS` unconditionally, and the
    one `except sqlite3.OperationalError` around it was written for "no coverage
    table". A database whose table predates `revivals` — or now `degraded` —
    raised `no such column` into that same arm, and the shard shipped with *no
    coverage at all*: silent, total, and indistinguishable from a database that
    had never run a pass.
    """
    from vios.process import store as S

    base = tempfile.mkdtemp(prefix="vios_oldcov_")
    st = S.Store(os.path.join(base, "a.db"))
    st.add_video(video_key="REEL00", url="https://instagram.com/p/REEL00/",
                 uploader="x", duration=None, width=1080, height=1920,
                 bytes=1, taken_at=None, meta={})
    st.set_shots("REEL00", [{"t0": 0.0, "t1": 4.0}], "pyscenedetect")
    # The 2026-08 shape: no next_try_at, no revivals, no degraded.
    st.conn.execute("CREATE TABLE coverage(video_key TEXT, component TEXT, "
                    "state TEXT, attempts INTEGER DEFAULT 0, seconds REAL, "
                    "claims INTEGER, vectors INTEGER, observer_id TEXT, "
                    "last_error TEXT, started_at REAL, done_at REAL, "
                    "PRIMARY KEY(video_key, component))")
    st.conn.execute("INSERT INTO coverage(video_key,component,state,claims) "
                    "VALUES('REEL00','keyphrase','done',20)")
    st.conn.commit()
    oid = st.observer("keyphrase", "gemma", "v1", {})
    st.add_claims("REEL00", oid, [
        {"channel": "concept", "kind": "keyphrase", "shot_idx": 0,
         "value": "steak"}])
    shard = os.path.join(base, "s.jsonl.gz")
    stats = st.export_shard(shard, 0, st.max_claim_id(), 0, 0, "gpu")
    check("an older coverage table still ships its rows",
          stats.get("coverage") == 1,
          f"{stats.get('coverage')} — before this it shipped 0 and said nothing")

    dst = S.Store(os.path.join(base, "b.db"))
    res = dst.import_shard(shard)
    check("and the target accepts a row with fields it did not send",
          res.get("coverage") == 1, str(res))
    got = dst.conn.execute("SELECT state, degraded FROM coverage").fetchone()
    check("reading as complete, which is all that shard can say",
          tuple(got) == ("done", None), str(tuple(got)))
    st.close()
    dst.close()


def test_replay_migrates_an_old_target():
    """The restore path must not lose coverage to `no such column` either.

    `_ensure_coverage_table` ran `CREATE TABLE IF NOT EXISTS`, which is a no-op
    against a table an earlier build already made — and then the upsert named
    `revivals`/`degraded`. `no such column` is a `sqlite3.Error`, so the shard's
    whole coverage section landed in the `skipped` arm while the restore
    reported success.
    """
    from vios.process import store as S

    base = tempfile.mkdtemp(prefix="vios_oldtarget_")
    src = S.Store(os.path.join(base, "a.db"))
    src.add_video(video_key="REEL00", url="https://instagram.com/p/REEL00/",
                  uploader="x", duration=None, width=1080, height=1920,
                  bytes=1, taken_at=None, meta={})
    src.set_shots("REEL00", [{"t0": 0.0, "t1": 4.0}], "pyscenedetect")
    from vios.process import coverage as C
    cov = C.Coverage(src.conn)
    cov.plan(["keyphrase"], ["REEL00"])
    oid = src.observer("keyphrase", "gemma", "v1", {"without": ["ocr"]})
    src.add_claims("REEL00", oid, [
        {"channel": "concept", "kind": "keyphrase", "shot_idx": 0,
         "value": "steak"}])
    cov.done("REEL00", "keyphrase", 3.0, claims=1, observer_id=oid,
             degraded=["ocr"])
    shard = os.path.join(base, "s.jsonl.gz")
    src.export_shard(shard, 0, src.max_claim_id(), 0, 0, "gpu")

    dst = S.Store(os.path.join(base, "b.db"))
    dst.conn.execute("CREATE TABLE coverage(video_key TEXT, component TEXT, "
                     "state TEXT, attempts INTEGER DEFAULT 0, seconds REAL, "
                     "claims INTEGER, vectors INTEGER, observer_id TEXT, "
                     "last_error TEXT, started_at REAL, done_at REAL, "
                     "PRIMARY KEY(video_key, component))")
    dst.conn.commit()
    res = dst.import_shard(shard)
    check("a target whose table predates the column still lands the row",
          res.get("coverage") == 1 and not res.get("skipped"), str(res))
    got = dst.conn.execute("SELECT state, degraded FROM coverage").fetchone()
    check("and the column was added to hold it", tuple(got) == ("done", "ocr"),
          str(tuple(got)))
    src.close()
    dst.close()


# ── 7. the engine wires it the only way that works ────────────────────────
def test_engine_wiring():
    src = open("vios/process/engine.py", encoding="utf-8").read()

    check("the terminal write carries the absence",
          "cov.done(key, cid, seconds, n_claims, n_vectors, observer, thin)"
          in src,
          "the row and the evidence are derived from the same list")
    check("and the observer id is derived from it too",
          "_observer_params(comp.params, thin)" in src)
    check("`thin` is wider than the veto list",
          'thin = [n for n in comp.soft if states.get(n) != "done"]' in src,
          "`degraded` is terminal states only, and cannot see an input that "
          "was never selected this session — the likeliest case of all")

    i_rec = src.find("self.reconcile_now(runnable)")
    i_off = src.find("cov.reoffer_degraded(runnable)")
    check("the re-offer runs after reconcile", 0 < i_rec < i_off,
          f"reconcile at {i_rec}, re-offer at {i_off} — reconcile is what turns "
          f"restored evidence into a done row with a claim count, so asking "
          f"first would answer 'no' on the one rotation where it is 'yes'")
    check("and it is scoped to the plan", "cov.reoffer_degraded(runnable)" in src,
          "a re-offer for a pass this session will not claim leaves a stage "
          "pending with nothing able to finish it")
    check("the status payload exposes it", '"thin": thin,' in src)


def test_ui_shows_it():
    html = open("process_ui.html", encoding="utf-8").read()
    check("the tab renders the thin count", "renderThin(s)" in html
          and 'id="dThin"' in html)
    check("and a stage can show complete and thin at once",
          "answered thin" in html,
          "`complete` was the whole sentence for 28 of 30 videos")


def main():
    print("the marker")
    test_marker()
    print("done records it, and clears it")
    test_done_records_and_clears()
    print("the identity split, and why it is needed")
    test_observer_identity()
    test_why_identity_must_split()
    print("the ladder `done` never had")
    test_reoffer()
    test_reoffer_is_scoped()
    test_instr_not_like()
    test_prefix_is_not_membership()
    test_multiple_absences()
    test_matrix_and_stage_report()
    print("reconcile, both directions")
    test_reconcile_carries_the_marker()
    test_degraded_observers()
    print("it travels")
    test_shard_round_trip()
    test_export_survives_an_older_table()
    test_replay_migrates_an_old_target()
    print("the engine and the tab")
    test_engine_wiring()
    test_ui_shows_it()

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} — " + "; ".join(FAILS))
        return 1
    print("DEGRADED OK — a pass that answered without an input says so on its "
          "row and in its observer id, the answer travels marked, and the row "
          "is handed back once that evidence exists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
