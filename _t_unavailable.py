"""Why a pass that could not start must not be retired like one that declined.

The measured defect: the archive has `ocr` claims for 2 of 30 videos. Before the
EasyOCR fallback landed on 2026-08-17 the OCR loader raised
`SkipPass("paddleocr is not installed")` — `paddlepaddle` is deliberately not in
requirements.txt — and the engine turned every one of those into a coverage row
in state `skipped`. Skipped is terminal in every direction: `TERMINAL` includes
it, `claim` and `candidates` never select it, `revive_failed` filters on
`failed`, `reconcile` updates only `queued`/`failed`, and `plan` is
`INSERT OR IGNORE`. So the fallback — which works — could only ever reach the two
rows that had not been retired yet, and those two are exactly the two videos with
OCR evidence in the database today.

What this pins:

  1. `PassUnavailable` is a *subclass* of `SkipPass`, so the two `except SkipPass`
     handlers that do not touch coverage keep behaving as before.
  2. `engine.py` catches it BEFORE `SkipPass`. Order is the whole fix; a subclass
     caught second never fires.
  3. `Coverage.unsettled` distinguishes a dependency that declined from one that
     is mid-retry, which is what stops one OOM on `shots` from permanently
     retiring `keyframes` and everything downstream.
  4. `Coverage.reclaim_unavailable` hands back the rows already retired and leaves
     genuine declines alone.
  5. Every string this codebase raises as `PassUnavailable` is matched by
     `UNAVAILABLE_LIKE`, and no environment fault is still raised as a plain
     `SkipPass`. Both directions, mechanically, over the real sources.

Run from the repo root: python _t_unavailable.py
"""

import ast
import fnmatch
import io
import os
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The reasons under test contain the em dashes and middots this codebase writes,
# and a cp1252 console turns a failure message into a UnicodeEncodeError, which
# reads as a broken test rather than a broken engine.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

RUNNERS = "vios/process/runners"
FAILS = []


def check(name, good, detail=""):
    print(f"  {'PASS' if good else 'FAIL'}  {name}"
          + (f" — {detail}" if detail else ""))
    if not good:
        FAILS.append(name)


def fresh_cov(**kw):
    """A real Coverage over a real database, with 30 videos and one component."""
    from vios.process import coverage as C

    path = os.path.join(tempfile.mkdtemp(prefix="vios_unavail_"), "a.db")
    conn = sqlite3.connect(path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE video(video_key TEXT PRIMARY KEY, "
                 "partition INTEGER DEFAULT 0)")
    conn.executemany("INSERT INTO video(video_key) VALUES (?)",
                     [(f"REEL{i:02d}",) for i in range(30)])
    conn.commit()
    return C, C.Coverage(conn, **kw), conn


# ── 1. the class, and the handler order that makes it mean anything ───────
def test_subclass_and_order():
    from vios.process.runners.base import DeferPass, PassUnavailable, SkipPass

    check("PassUnavailable is a SkipPass",
          issubclass(PassUnavailable, SkipPass),
          "so `except SkipPass` sites that do not touch coverage are unchanged")
    check("and it is not a DeferPass",
          not issubclass(PassUnavailable, DeferPass),
          "a defer spends no attempt; this one is meant to spend them")

    src = open("vios/process/engine.py", encoding="utf-8").read()
    i_un = src.find("except PassUnavailable as exc:")
    i_sk = src.find("except SkipPass as exc:\n            # A decline is terminal")
    check("engine.py catches PassUnavailable before SkipPass",
          0 < i_un < i_sk,
          f"unavailable at {i_un}, skip at {i_sk} — a subclass caught second "
          f"never fires")

    # The routing, asserted on the source rather than by running a GPU pass:
    # the two handlers must reach different coverage calls.
    win = src[i_un:i_sk]
    check("the unavailable handler records a failure, not a skip",
          "cov.fail(" in win and "cov.skip(" not in win,
          "fail() is what MAX_ATTEMPTS and revive_failed can see")

    # The incidental handlers must stay incidental: a subclass passing through
    # them must not reach coverage at all. The body is taken to the end of the
    # `except` block by indentation rather than a fixed number of lines — the
    # coverage handler's own comment is longer than any fixed window I would
    # have picked, and a window that stops short reports "writes no coverage"
    # about the one handler that does.
    writers = []
    for where in ("vios/process/engine.py", "vios/process/runners/base.py"):
        lines = open(where, encoding="utf-8").read().split("\n")
        for j, line in enumerate(lines, 1):
            if "except SkipPass" not in line:
                continue
            indent = len(line) - len(line.lstrip())
            body = []
            for nxt in lines[j:]:
                if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                    break
                body.append(nxt)
            body = "\n".join(body)
            if "cov.skip(" in body or "cov.fail(" in body:
                writers.append(f"{where}:{j}")
                check(f"{where}:{j} is the coverage handler",
                      "cov.skip(" in body and "cov.fail(" not in body,
                      f"{len(body.splitlines())} lines — the one place a decline "
                      f"becomes a row, and it must not also fail")
            else:
                check(f"{where}:{j} does not write coverage", True,
                      "a PassUnavailable through here is caught and logged only")
    check("exactly one `except SkipPass` writes coverage", len(writers) == 1,
          ", ".join(writers) or "none found — the handler moved or was renamed")


# ── 2. failed is reachable, skipped is not ───────────────────────────────
def test_lifetimes_differ():
    C, cov, conn = fresh_cov()
    cov.plan(["ocr"])

    cov.skip("REEL00", "ocr", "no frame could be read")
    # Claimed before it is failed, because that is the only path by which the
    # engine reaches a raise, and it is `claim_for` that increments `attempts` —
    # which `fail()` then reads to choose the backoff. Failing a row that was
    # never claimed is a different branch (`attempts == 0` → `next_try_at = 0`,
    # retry at once) and not the one a pass that could not start takes.
    taken = cov.claim_for("REEL01", ["ocr"])
    check("the row is claimed before the pass runs", taken == ["ocr"], str(taken))
    cov.fail("REEL01", "ocr", "unavailable: paddleocr is not installed")

    claimable = set(cov.candidates(["ocr"], 50))
    check("a skipped row is invisible to the queue",
          "REEL00" not in claimable, "which is correct for a real decline")
    # A row that failed once goes back to `queued` with a backoff, so it is not
    # claimable this second — that is the point of RETRY_BACKOFF, and it is a
    # different thing from being invisible.
    row = conn.execute("SELECT state, next_try_at FROM coverage "
                       "WHERE video_key='REEL01'").fetchone()
    check("a failed row is queued again with a backoff",
          row["state"] == C.QUEUED and row["next_try_at"] > time.time(),
          f"state={row['state']}, in {row['next_try_at'] - time.time():.0f}s")

    # Burn the ladder: three attempts, then FAILED, then revive_failed sees it.
    conn.execute("UPDATE coverage SET attempts=? WHERE video_key='REEL01'",
                 (C.MAX_ATTEMPTS,))
    state = cov.fail("REEL01", "ocr", "unavailable: paddleocr is not installed")
    check("out of attempts it becomes failed", state == C.FAILED, state)
    conn.execute("UPDATE coverage SET next_try_at=0 WHERE video_key='REEL01'")
    got = cov.revive_failed()
    check("and revive_failed hands it back", got["revived"] == 1, str(got))

    # The same treatment can never reach the skipped row.
    conn.execute("UPDATE coverage SET next_try_at=0 WHERE video_key='REEL00'")
    check("revive_failed still cannot see the skipped row",
          cov.revive_failed()["revived"] == 0,
          "which is why an environment fault must not go there")
    conn.close()


# ── 3. a dependency mid-retry is not a dependency that declined ──────────
def test_defer_costs_nothing():
    """The two new waits are `defer`s, and a wait must not spend a life.

    `claim_for` increments `attempts` before the pass is reached, so every
    `defer` in `_run_pass` is deferring a row that has already been charged for
    a run that did not happen. Left uncorrected, a video waiting six hours on
    `shots` accumulates attempts while doing nothing, and the first genuine
    failure afterwards reads `attempts >= MAX_ATTEMPTS` and jumps the 10- and
    45-minute backoff for the four-hour ladder.
    """
    C, cov, conn = fresh_cov()
    cov.plan(["keyframes"])

    for i in range(6):
        cov.claim_for("REEL00", ["keyframes"])
        cov.defer("REEL00", "keyframes", "waiting on shots, which failed but "
                                         "has retries left", 300.0)
    row = conn.execute("SELECT state, attempts, next_try_at FROM coverage "
                       "WHERE video_key='REEL00'").fetchone()
    check("six waits spend no attempts",
          row["attempts"] == 0 and row["state"] == C.QUEUED,
          f"attempts={row['attempts']}, state={row['state']} — twice "
          f"MAX_ATTEMPTS of waiting and the row is as new as it was")
    check("and it is waiting on a clock, not invisible",
          row["next_try_at"] > time.time(),
          f"due in {row['next_try_at'] - time.time():.0f}s")

    # And the point of all that: the first real failure still gets the cheap
    # retries it would have got if the waiting had never happened.
    conn.execute("UPDATE coverage SET next_try_at=0")
    conn.commit()
    cov.claim_for("REEL00", ["keyframes"])
    state = cov.fail("REEL00", "keyframes", "RuntimeError: CUDA out of memory")
    row = conn.execute("SELECT attempts, next_try_at FROM coverage "
                       "WHERE video_key='REEL00'").fetchone()
    check("the first real failure still gets the first backoff",
          state == C.QUEUED
          and abs(row["next_try_at"] - time.time() - C.RETRY_BACKOFF[0]) < 5,
          f"state={state}, in {row['next_try_at'] - time.time():.0f}s "
          f"(RETRY_BACKOFF[0]={C.RETRY_BACKOFF[0]}), attempts="
          f"{row['attempts']}")
    conn.close()


def test_unsettled():
    C, cov, conn = fresh_cov()
    cov.plan(["shots", "keyframes"])

    # Four shapes of `shots` for four videos, one per lifetime.
    cov.done("REEL00", "shots", 1.0, claims=3)
    cov.skip("REEL01", "shots", "container reports zero duration")
    cov.fail("REEL02", "shots", "RuntimeError: CUDA out of memory")
    conn.execute("UPDATE coverage SET state=?, revivals=?, next_try_at=? "
                 "WHERE video_key='REEL03' AND component='shots'",
                 (C.FAILED, C.MAX_REVIVALS, 0.0))
    conn.execute("UPDATE coverage SET state=?, next_try_at=? "
                 "WHERE video_key='REEL02' AND component='shots'",
                 (C.FAILED, time.time() + C.REVIVE_AFTER))
    conn.commit()

    check("a done dependency is settled",
          cov.unsettled("REEL00", ["shots"]) == {},
          "nothing to wait for")
    check("a declined dependency is settled",
          cov.unsettled("REEL01", ["shots"]) == {},
          "it will decline again next session, so the consumer is retired too")
    w = cov.unsettled("REEL02", ["shots"])
    check("a failed dependency with revivals left is NOT settled",
          list(w) == ["shots"] and w["shots"] > time.time(),
          f"{w} — and it reports its own next_try_at so the consumer waits "
          f"exactly that long instead of re-asking every minute")
    check("a failed dependency out of revivals is settled",
          cov.unsettled("REEL03", ["shots"]) == {},
          "six revivals spent; nothing more is coming")
    check("a queued dependency is not settled",
          list(cov.unsettled("REEL04", ["shots"])) == ["shots"],
          "planned and never run")
    check("unsettled asked about nothing answers nothing",
          cov.unsettled("REEL00", []) == {} and cov.unsettled("REEL00", None) == {},
          "so the gate can call it unconditionally")
    conn.close()


# ── 4. handing back the rows that are already retired ────────────────────
def test_reclaim():
    C, cov, conn = fresh_cov()
    cov.plan(["ocr", "transcribe", "shots"])

    # Exactly what the 8-16 August sweeps wrote, and what they should not have.
    for i in range(28):
        cov.skip(f"REEL{i:02d}", "ocr", "paddleocr is not installed")
    cov.done("REEL28", "ocr", 90.0, claims=80)
    cov.done("REEL29", "ocr", 90.0, claims=5)
    # Genuine declines, which must not move.
    for i in range(6):
        cov.skip(f"REEL{i:02d}", "transcribe", "no speech detected")
    cov.skip("REEL06", "shots", "container reports zero duration — file is "
                                "truncated")
    cov.skip("REEL07", "shots", "depends on probe, which produced nothing for "
                                "this video")

    back = cov.reclaim_unavailable()
    check("the retired OCR rows come back",
          back["requeued"] == 28, str(back["requeued"]))
    check("and the log can name which pass lost them",
          back["components"] == {"ocr": 28}, str(back["components"]))
    check("the two that were already done are untouched",
          conn.execute("SELECT COUNT(*) FROM coverage WHERE component='ocr' "
                       "AND state=?", (C.DONE,)).fetchone()[0] == 2,
          "reclaim filters on state='skipped', so it cannot double-write")
    check("a real decline stays declined",
          conn.execute("SELECT COUNT(*) FROM coverage WHERE state=?",
                       (C.SKIPPED,)).fetchone()[0] == 8,
          "6 no-speech + 1 zero-duration + 1 dependency skip")
    check("the reclaimed rows are claimable now",
          len(cov.candidates(["ocr"], 50)) == 28,
          "which is the whole point — before this they were invisible forever")
    check("attempts were cleared so the retry is a real retry",
          conn.execute("SELECT COUNT(*) FROM coverage WHERE component='ocr' "
                       "AND state=? AND attempts=0",
                       (C.QUEUED,)).fetchone()[0] == 28)
    check("last_error is kept",
          (conn.execute("SELECT last_error FROM coverage WHERE "
                        "video_key='REEL00' AND component='ocr'").fetchone()[0]
           or "").startswith("paddleocr"),
          "so a second failure can be compared with the first")

    again = cov.reclaim_unavailable()
    check("running it twice reclaims nothing the second time",
          again["requeued"] == 0,
          "idempotent on its own, before the meta marker is even consulted")
    conn.close()

    # And the empty case, which is a result rather than a silence: it rules out
    # the one alternative a log this old cannot otherwise settle.
    _, cov2, conn2 = fresh_cov()
    cov2.plan(["ocr"])
    cov2.skip("REEL00", "ocr", "no frame could be read")
    got = cov2.reclaim_unavailable()
    check("a database with no environment declines reports zero",
          got == {"requeued": 0, "components": {}}, str(got))
    conn2.close()


# ── 5. the two directions, over every raise site in the package ──────────
def _raise_strings(kind):
    """Every `raise <kind>(...)` in the runners, rendered to a shape.

    f-string placeholders become a token, because what has to be recognisable
    later is the fixed text around them. A raise whose whole argument is dynamic
    renders to nothing but the token and is reported separately: a reason this
    code did not write cannot be matched by a pattern this code owns.
    """
    out = []
    for fn in sorted(os.listdir(RUNNERS)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(RUNNERS, fn)
        tree = ast.parse(open(path, encoding="utf-8").read(), path)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
                continue
            name = getattr(node.exc.func, "id", "") or \
                getattr(node.exc.func, "attr", "")
            if name != kind or not node.exc.args:
                continue
            out.append((f"{fn}:{node.lineno}", _render(node.exc.args[0])))
    return out


def _render(node):
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else "<X>"
    if isinstance(node, ast.JoinedStr):
        return "".join(_render(v) for v in node.values)
    if isinstance(node, ast.FormattedValue):
        return "<X>"
    if isinstance(node, ast.BinOp):          # "a" + "b", "a" + join(...)
        return _render(node.left) + _render(node.right)
    if isinstance(node, ast.BoolOp):         # x or "fallback"
        return " ".join(_render(v) for v in node.values)
    return "<X>"


def test_both_directions():
    from vios.process.coverage import Coverage

    pats = [p.replace("%", "*") for p in Coverage.UNAVAILABLE_LIKE]

    def matched(text):
        low = text.lower()
        return any(fnmatch.fnmatchcase(low, p.lower()) for p in pats)

    unavail = _raise_strings("PassUnavailable")
    check("every environment fault is a PassUnavailable", len(unavail) >= 23,
          f"{len(unavail)} site(s)")
    unmatched = [(w, t) for w, t in unavail if not matched(t)]
    check("and every one of them is matchable by UNAVAILABLE_LIKE",
          not unmatched,
          "; ".join(f"{w} {t[:60]!r}" for w, t in unmatched)
          or "so a row already retired for any of them can be handed back")

    skips = _raise_strings("SkipPass")
    leaked = [(w, t) for w, t in skips if matched(t)]
    check("and no environment fault is still a plain SkipPass",
          not leaked,
          "; ".join(f"{w} {t[:60]!r}" for w, t in leaked)
          or f"{len(skips)} declines checked, none names a missing package, "
             f"model, key or tool")

    # The declines that remain must be about the video, not the machine. This is
    # the tripwire for the words that turned out to mean "the environment": if a
    # new SkipPass starts using one, it is a terminal row for something a later
    # session would fix.
    smells = ("not installed", "could not be loaded", "could not start",
              "not on PATH", "no api key", "unreachable", "not reachable")
    suspect = [(w, t) for w, t in skips
               if any(s in t.lower() for s in smells)]
    check("no remaining SkipPass reads like a machine fault", not suspect,
          "; ".join(f"{w} {t[:60]!r}" for w, t in suspect))


def main():
    print("the class and the handler order")
    test_subclass_and_order()
    print("failed is reachable, skipped is not")
    test_lifetimes_differ()
    print("a dependency mid-retry")
    test_defer_costs_nothing()
    test_unsettled()
    print("handing back what was already retired")
    test_reclaim()
    print("every raise site, both directions")
    test_both_directions()

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} — " + "; ".join(FAILS))
        return 1
    print("UNAVAILABLE OK — a pass that could not start is retried, a pass that "
          "declined is not, and the rows already retired for a missing backend "
          "are reachable again")
    return 0


if __name__ == "__main__":
    sys.exit(main())


