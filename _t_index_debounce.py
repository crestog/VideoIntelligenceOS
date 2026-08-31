"""Does a cold scan still re-index once per shard?

The measured defect: 82 imports drove ~76 whole-archive rebuilds — DELETE every
moment, re-INSERT, FTS5 rebuild, graph, dense re-embed, UMAP — and threw 75 away,
which was essentially the entire 488-second Atlas boot for a 1,895-passage
archive.

`server._index_if_due` is the debounce. Five things have to hold, and the last
two are the ones that make batching safe rather than lossy:

1. a burst of shards drives ONE rebuild, not one per shard
2. the FIRST shard still rebuilds immediately, because indexing mid-scan exists
   so the first bundle is searchable while the rest download
3. the forced end-of-scan flush always builds when rows are still pending
4. `reflect.fingerprint` moves only when the SCHEMA moves — so a scan that
   imported rows into existing columns cannot rely on the fingerprint to notice,
   which is why (3) must be forced. Asserted directly against a real database.
5. a rebuild that FAILS gives the pending rows back. `index.rebuild` reports
   failure as a return value rather than an exception, so this is the one that
   decides whether `database is locked` costs a session its index.

The stubs are deliberate and different per test: most of these care how often
`_index_if_stale` is called and with what `force`, not what a rebuild does — but
(5) runs the real `_index_if_stale` against a real database and stubs
`index.rebuild` underneath it, because the discarded return value is the defect.

Run from the repo root: python _t_index_debounce.py
"""

import os
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILS = []


def check(name, good, detail=""):
    print(f"  {'PASS' if good else 'FAIL'}  {name}"
          + (f" — {detail}" if detail else ""))
    if not good:
        FAILS.append(name)


def fresh(server, gap=45.0):
    """Reset the debounce to a just-started process."""
    server._INDEX_PENDING = 0
    server._INDEX_LAST = 0.0
    server._INDEX_MIN_GAP = gap


def install_spy(server):
    calls = []

    def spy(conn, force=False):
        calls.append(bool(force))
        return True

    server._index_if_stale = spy
    return calls


def test_burst(server):
    calls = install_spy(server)
    fresh(server)

    # 82 shards landing back to back, every one carrying rows — a cold scan.
    for _ in range(82):
        server._index_if_due(None, 24)

    check("a cold scan of 82 shards does not rebuild 82 times",
          len(calls) <= 2, f"{len(calls)} rebuild(s)")
    check("the first shard rebuilds immediately", len(calls) >= 1,
          "so the first bundle is searchable while the rest download")
    check("every rebuild in the burst was forced by pending rows",
          all(calls), str(calls))

    before = len(calls)
    server._index_if_due(None, force=True)
    check("the forced end-of-scan flush builds the rest",
          len(calls) == before + 1, f"{len(calls) - before} extra rebuild(s)")
    check("nothing is left pending afterwards", server._INDEX_PENDING == 0,
          f"pending={server._INDEX_PENDING}")

    # 82 rebuilds at the ~6 s/rebuild the log implies is 488 s; this is 2.
    saved = 82 - len(calls)
    print(f"        (informational: {len(calls)} rebuild(s) instead of 82 — "
          f"{saved} avoided)")


def test_gap_elapsing(server):
    calls = install_spy(server)
    fresh(server, gap=0.35)

    server._index_if_due(None, 10)          # first: immediate
    server._index_if_due(None, 10)          # inside the gap: held
    held = len(calls)
    time.sleep(0.45)
    server._index_if_due(None, 10)          # gap elapsed: builds
    check("a shard after the gap elapses does rebuild",
          len(calls) == held + 1, f"{held} then {len(calls)}")


def test_no_rows_no_build(server):
    calls = install_spy(server)
    fresh(server)
    server._index_if_due(None, 0)
    check("a shard that carried no rows builds nothing", not calls,
          f"{len(calls)} rebuild(s)")
    check("an unforced call with nothing pending builds nothing",
          not server._index_if_due(None, 0))


def test_force_with_nothing_pending(server):
    calls = install_spy(server)
    fresh(server)
    server._index_if_due(None, force=True)
    check("a forced flush with nothing pending defers to the fingerprint",
          calls == [False],
          f"{calls} — force=False lets _index_if_stale decide, which is what "
          f"carries an index over from a previous run")


def test_pending_survives_a_failure(server):
    calls = []

    def boom(conn, force=False):
        calls.append(bool(force))
        raise sqlite3.OperationalError("database is locked")

    server._index_if_stale = boom
    fresh(server)
    try:
        server._index_if_due(None, 40)
    except sqlite3.OperationalError:
        pass
    check("a failed rebuild does not swallow the pending rows",
          server._INDEX_PENDING == 40, f"pending={server._INDEX_PENDING}")

    ok = install_spy(server)
    server._INDEX_LAST = 0.0
    server._index_if_due(None, 0)
    check("the retry after a failure is still forced", ok == [True], str(ok))


def test_fingerprint_is_schema_only():
    """Why the end-of-scan flush must be forced, asserted not assumed."""
    from atlas import reflect

    path = os.path.join(tempfile.mkdtemp(prefix="vios_fp_"), "fp.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE claims (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    empty = reflect.fingerprint(conn)

    conn.executemany("INSERT INTO claims (value) VALUES (?)",
                     [(f"row {i}",) for i in range(500)])
    conn.commit()
    with_rows = reflect.fingerprint(conn)
    check("500 imported rows do not move the fingerprint", empty == with_rows,
          "so a debounce without a forced flush would lose every one of them")

    conn.execute("ALTER TABLE claims ADD COLUMN confidence REAL")
    conn.commit()
    check("a new column does move the fingerprint",
          reflect.fingerprint(conn) != with_rows,
          "which is the case _index_if_stale's own guard is for")
    conn.close()


def test_reported_failure_is_a_failure(server):
    """A rebuild that returns `ok: False` must not be read as a success.

    `index.rebuild` catches its own exceptions and answers with a dict, so the
    failure that actually happened on Kaggle — `index build failed —
    OperationalError: database is locked` — arrives here as a return value. If
    that is discarded, the debounce spends its pending count on a build that
    indexed nothing and wrote no fingerprint, and every later guard agrees there
    is nothing left to do.
    """
    from atlas import index as index_mod, ingest

    # A real connection: `_index_if_stale` itself is under test here, not a stub,
    # and it counts `moments`, reads meta and fingerprints the schema before it
    # ever reaches the rebuild.
    path = os.path.join(tempfile.mkdtemp(prefix="vios_fail_"), "a.db")
    conn = sqlite3.connect(path)
    ingest.ensure_meta(conn)

    keep_rebuild = index_mod.rebuild
    keep_graph = server._rebuild_graph
    graph_calls = []
    server._rebuild_graph = lambda conn: graph_calls.append(1)
    index_mod.rebuild = lambda conn, embed=False: {
        "ok": False, "note": "OperationalError: database is locked"}
    try:
        fresh(server)
        raised = ""
        try:
            server._index_if_due(conn, 61)
        except Exception as e:                              # noqa: BLE001
            raised = f"{type(e).__name__}: {e}"
        check("a rebuild reporting ok=False raises instead of returning True",
              "IndexBuildFailed" in raised, raised or "nothing raised")
        check("the reported reason survives into the exception",
              "database is locked" in raised, raised)
        check("a failed rebuild does not derive a graph from it",
              not graph_calls, f"{len(graph_calls)} graph rebuild(s)")
        check("a reported failure gives the pending rows back",
              server._INDEX_PENDING == 61,
              f"pending={server._INDEX_PENDING}")
    finally:
        index_mod.rebuild = keep_rebuild
        server._rebuild_graph = keep_graph
        conn.close()


def main():
    os.environ.setdefault("ATLAS_SKIP_BOOT", "1")
    from atlas import server

    keep = server._index_if_stale
    try:
        print("a cold scan of 82 shards")
        test_burst(server)
        print("the gap elapsing mid-scan")
        test_gap_elapsing(server)
        print("shards that carried nothing")
        test_no_rows_no_build(server)
        test_force_with_nothing_pending(server)
        print("a rebuild that fails")
        test_pending_survives_a_failure(server)
        server._index_if_stale = keep
        test_reported_failure_is_a_failure(server)
        print("why the final flush has to be forced")
        test_fingerprint_is_schema_only()
    finally:
        server._index_if_stale = keep

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} — " + "; ".join(FAILS))
        return 1
    print("DEBOUNCE OK — one rebuild for a cold scan, and nothing left "
          "unindexed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
