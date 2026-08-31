"""Why `index build failed — OperationalError: database is locked` happened.

The measured defect: a boot that imported 82 shards logged that line and lost the
index for the session. Nothing was holding a lock too long and no pragma was
missing — WAL is on (ingest.py:149), `busy_timeout` is set twice over
(ingest.py:147 and :153), and every write in this archive takes milliseconds.

The cause is a *read-to-write promotion*. `INSERT` issued inside a `for row in
conn.execute(SELECT)` loop asks SQLite to turn the deferred read transaction that
the SELECT opened into a write transaction. SQLite refuses that outright —
`SQLITE_BUSY_SNAPSHOT`, whose message is the same "database is locked" — the
moment any other connection has committed since the snapshot was taken. It
refuses without ever calling the busy handler, so the 60-second timeout is not
consulted and the failure is instant.

There were exactly two such loops in `atlas/`, both now materialised with
`.fetchall()`:

  identity.py:1181  the collections pass, reached by every `index.rebuild` via
                    `identity.refresh` before a single source row is read
  ingest.py:373     `_add_missing`, whose `except sqlite3.Error: pass` turned the
                    refusal into a destination table one column narrower than the
                    snapshot it had just imported

And one containment: a build that dies mid-transaction used to leave the shard
importer's own connection inside it, where every later write failed identically.

Run from the repo root: python _t_snapshot.py
"""

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILS = []


def check(name, good, detail=""):
    print(f"  {'PASS' if good else 'FAIL'}  {name}"
          + (f" — {detail}" if detail else ""))
    if not good:
        FAILS.append(name)


def connect(path):
    """The repo's own connection settings — ingest.connect, minus the config."""
    conn = sqlite3.connect(path, timeout=60.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def fresh_db(prefix):
    path = os.path.join(tempfile.mkdtemp(prefix=prefix), "a.db")
    conn = connect(path)
    conn.execute("CREATE TABLE video(video_key TEXT PRIMARY KEY, meta TEXT, "
                 "msg_id INTEGER, url TEXT)")
    conn.executemany(
        "INSERT INTO video(video_key, meta) VALUES (?,?)",
        [(f"REEL{i}", '{"capture": {"collections": ["saved", "cooking"]}}')
         for i in range(40)])
    conn.commit()
    return path, conn


def contender(path):
    """A second connection that commits, the way the `atlas-map` thread does."""
    other = connect(path)
    other.execute("CREATE TABLE IF NOT EXISTS other(k TEXT PRIMARY KEY)")
    other.commit()
    n = [0]

    def bump():
        n[0] += 1
        other.execute("INSERT OR REPLACE INTO other VALUES (?)", (f"k{n[0]}",))
        other.commit()
    return other, bump


# ── 1. The mechanism, demonstrated ────────────────────────────────────────
def test_mechanism():
    path, conn = fresh_db("vios_snap_")
    conn.execute("CREATE TABLE hit(k TEXT PRIMARY KEY)")
    conn.commit()
    other, bump = contender(path)

    check("the connection is in autocommit before the loop",
          not conn.in_transaction,
          "so the INSERT below is a promotion, not a first statement")

    wrote, err = 0, None
    try:
        for i, (vk, _meta) in enumerate(
                conn.execute("SELECT video_key, meta FROM video")):
            if i == 0:
                bump()
            conn.execute("INSERT OR IGNORE INTO hit VALUES (?)", (vk,))
            wrote += 1
    except sqlite3.OperationalError as e:
        err = e

    name = getattr(err, "sqlite_errorname", "")
    check("writing through a stepping cursor is refused", err is not None,
          f"wrote {wrote} row(s) before failing")
    check("and it is refused as SQLITE_BUSY_SNAPSHOT", name == "SQLITE_BUSY_SNAPSHOT",
          f"{name or 'no error'}: {err}")
    check("whose message is the one the boot log printed",
          "database is locked" in str(err).lower(), str(err))
    check("the busy handler is never consulted", wrote == 0,
          "it fails at row 0, so busy_timeout=60000 cannot help")
    check("and the connection is left inside the transaction",
          conn.in_transaction,
          "which is why every later write on it fails identically")

    # The same interleaving, materialised.
    conn.rollback()
    wrote, err = 0, None
    try:
        for i, (vk, _meta) in enumerate(
                conn.execute("SELECT video_key, meta FROM video").fetchall()):
            if i == 0:
                bump()
            conn.execute("INSERT OR IGNORE INTO hit VALUES (?)", (vk,))
            wrote += 1
        conn.commit()
    except sqlite3.OperationalError as e:
        err = e
    check("the same loop with .fetchall() writes every row",
          err is None and wrote == 40, f"wrote {wrote}, err={err}")
    conn.close()
    other.close()


# ── 2. The collections pass, against the real function ────────────────────
def test_rebuild_collections():
    from atlas import identity

    path, conn = fresh_db("vios_snap_ident_")
    identity.ensure(conn)
    other, bump = contender(path)
    bump()                      # commit *before* the pass, invalidating nothing
    #                             yet — the snapshot is taken inside it.

    # A commit landing while `rebuild_collections` runs is what the map thread
    # does. There is no hook to interleave one, so the pre-commit plus WAL is
    # what the fix has to survive; the mechanism itself is pinned by test 1.
    out = identity.rebuild_collections(conn, ledger_path="")
    check("rebuild_collections writes a membership for every video",
          out["from_meta"] == 80, f"{out['from_meta']} row(s) for 40 videos × 2")
    got = conn.execute("SELECT COUNT(*) FROM video_collection").fetchone()[0]
    check("and they are all in video_collection", got == 80, f"{got} row(s)")
    check("the source loop is materialised",
          ".fetchall()" in _source_after(identity, "SELECT video_key, meta FROM video"),
          "so the INSERT is its transaction's first statement")
    conn.close()
    other.close()


def _source_after(module, needle, window=60):
    """The source around `needle`, so a fix cannot be silently reverted."""
    import inspect
    src = inspect.getsource(module)
    i = src.find(needle)
    return src[i:i + window] if i >= 0 else ""


# ── 3. Widening a table, against the real function ────────────────────────
def test_add_missing():
    from atlas import ingest

    path, conn = fresh_db("vios_snap_wide_")
    src = os.path.join(os.path.dirname(path), "src.db")
    snap = connect(src)
    snap.execute("CREATE TABLE video(video_key TEXT, meta TEXT, "
                 "hook_score REAL, ocr_text TEXT)")
    snap.commit()
    snap.close()
    conn.execute("ATTACH DATABASE ? AS src", (src,))
    other, bump = contender(path)
    bump()

    ingest._add_missing(conn, "video", "", "video")
    cols = {r[1] for r in conn.execute('PRAGMA table_info("video")').fetchall()}
    check("_add_missing adds every column the snapshot has",
          {"hook_score", "ocr_text"} <= cols,
          f"missing {sorted({'hook_score', 'ocr_text'} - cols)}")
    check("its PRAGMA cursor is materialised",
          ".fetchall()" in _source_after(
              ingest, 'PRAGMA src.table_info("{src}")'),
          "an ALTER inside it is a promotion, and the except swallows it")
    conn.close()
    other.close()


# ── 4. A build that dies leaves a usable connection ───────────────────────
def test_failed_build_rolls_back():
    """`index.rebuild`'s handler must roll back and name the phase."""
    from atlas import index, ingest

    path, conn = fresh_db("vios_snap_roll_")
    ingest.ensure_meta(conn)

    # Fail the build inside a transaction the way a refused promotion does.
    keep = index.ensure_schema

    def boom(c):
        c.execute("CREATE TABLE IF NOT EXISTS scratch(k TEXT)")
        c.execute("INSERT INTO scratch VALUES ('x')")     # opens a transaction
        raise sqlite3.OperationalError("database is locked")

    index.ensure_schema = boom
    try:
        result = index.rebuild(conn, embed=False)
    finally:
        index.ensure_schema = keep

    check("a build that raises answers ok=False", result.get("ok") is False,
          str(result)[:90])
    check("and the note names the phase it died in",
          "while" in str(result.get("note", "")),
          str(result.get("note", ""))[:110])
    check("the connection is out of the transaction afterwards",
          not conn.in_transaction,
          "without the rollback, every later import write fails the same way")
    wrote = None
    try:
        conn.execute("INSERT OR REPLACE INTO video(video_key) VALUES ('AFTER')")
        conn.commit()
        wrote = conn.execute(
            "SELECT COUNT(*) FROM video WHERE video_key='AFTER'").fetchone()[0]
    except sqlite3.Error as e:
        wrote = f"raised {e}"
    check("so the next shard can still write", wrote == 1, str(wrote))
    conn.close()


def main():
    print("the mechanism")
    test_mechanism()
    print("the collections pass")
    test_rebuild_collections()
    print("widening a table")
    test_add_missing()
    print("a build that fails mid-transaction")
    test_failed_build_rolls_back()

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} — " + "; ".join(FAILS))
        return 1
    print("SNAPSHOT OK — no write is issued through a stepping cursor, and a "
          "failed build hands the connection back clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
