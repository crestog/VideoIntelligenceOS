"""
vios.capture.ledger — the permanent record of what has been captured.

This is the single most important file in the capture plane, because it is the
only thing that is not regenerable. Everything else — the queue, the temp
files, the running process, the Kaggle session — is disposable. The ledger is
the answer to "have I already got this reel?", and it has to keep answering
that correctly after the notebook dies, after the account is rotated, after
six months and a new laptop.

Three properties it must have, and how each is bought:

  Durable across a process kill.
      WAL journal, `synchronous=FULL`, and one commit per item. A row is
      written *before* the fetch begins (state `fetching`) and updated after
      the upload returns, so a hard kill leaves evidence that work started —
      which the next run repairs rather than repeats blindly.

  Durable across the machine vanishing.
      Kaggle deletes everything. So the ledger is snapshotted to the Telegram
      channel as a document every `SNAPSHOT_EVERY` items, and `restore()`
      pulls the newest snapshot back before a run begins. Telegram is the
      permanent store for the archive; it is the permanent store for the
      bookkeeping too.

  Durable even if both of those are lost.
      The channel itself is the ground truth: every uploaded video carries its
      permalink in the caption. `vios.capture.seed` walks the channel and
      rebuilds the ledger from those captions. That is how the 552 reels
      captured by the old Colab script are adopted without re-downloading a
      single byte.

The key is the Instagram shortcode, not the URL, because the same reel reaches
us as /reel/, /reels/, /p/ and /tv/ with and without query strings, and from
several collections at once.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time

SCHEMA_VERSION = 1

# States a queue item can be in. Everything except `uploaded` and `unavailable`
# is retryable; those two are terminal.
QUEUED = "queued"
FETCHING = "fetching"
UPLOADED = "uploaded"
FAILED = "failed"
UNAVAILABLE = "unavailable"   # deleted / private / geo-blocked — do not retry
SKIPPED = "skipped"           # excluded by category filter

TERMINAL = (UPLOADED, UNAVAILABLE, SKIPPED)

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS item (
    key           TEXT PRIMARY KEY,     -- instagram shortcode
    url           TEXT NOT NULL,        -- canonical permalink
    kind          TEXT,                 -- reel | p | tv
    state         TEXT NOT NULL,
    added_at      REAL NOT NULL,
    source        TEXT,                 -- which input produced this row
    position      INTEGER,              -- order within the input, for FIFO

    attempts      INTEGER NOT NULL DEFAULT 0,
    last_try_at   REAL,
    next_try_at   REAL NOT NULL DEFAULT 0,
    last_error    TEXT,

    -- filled in once the bytes are safely in Telegram
    done_at       REAL,
    msg_id        INTEGER,
    record_msg_id INTEGER,
    file_id       TEXT,
    file_size     INTEGER,
    sha256        TEXT,
    ext           TEXT,
    duration      REAL,
    width         INTEGER,
    height        INTEGER,

    -- denormalised post facts, so the UI and the processing plane can filter
    -- and sort without opening a single record JSON
    uploader      TEXT,
    title         TEXT,
    views         INTEGER,
    likes         INTEGER,
    comment_count INTEGER,
    comments_got  INTEGER,
    taken_at      REAL,
    lang          TEXT
);

CREATE INDEX IF NOT EXISTS item_state ON item(state, next_try_at, position);
CREATE INDEX IF NOT EXISTS item_done  ON item(done_at);
CREATE INDEX IF NOT EXISTS item_up    ON item(uploader);

-- One reel can live in several saved collections. Kept separate so a second
-- import adds memberships without rewriting the item.
CREATE TABLE IF NOT EXISTS membership (
    key        TEXT NOT NULL,
    collection TEXT NOT NULL,
    PRIMARY KEY (key, collection)
);
CREATE INDEX IF NOT EXISTS membership_col ON membership(collection);

-- Append-only journal. This is what the UI's activity feed reads, and what
-- makes a post-mortem possible after an unattended week.
CREATE TABLE IF NOT EXISTS event (
    id    INTEGER PRIMARY KEY,
    at    REAL NOT NULL,
    kind  TEXT NOT NULL,
    key   TEXT,
    text  TEXT
);
CREATE INDEX IF NOT EXISTS event_at ON event(at);

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""

# Matches every shape Instagram uses for a single post permalink. The trailing
# group is deliberately not anchored to `/` so bare shortcodes pasted into a
# markdown file still match.
PERMALINK = re.compile(
    r"https?://(?:www\.)?instagram\.com/"
    r"(?:[A-Za-z0-9_.]+/)?"                 # optional /<username>/ infix
    r"(reel|reels|p|tv)/"
    r"([A-Za-z0-9_-]{5,})",
    re.IGNORECASE,
)


def canonical(url: str) -> tuple[str, str, str] | None:
    """(key, canonical_url, kind) for an Instagram permalink, or None.

    Normalising here is what makes the ledger's promise hold: the same reel
    arriving as a /p/ link from the export and a /reel/ link from a markdown
    file must collide on one row, or it gets downloaded twice.
    """
    m = PERMALINK.search(url or "")
    if not m:
        return None
    kind = m.group(1).lower()
    if kind == "reels":
        kind = "reel"
    key = m.group(2)
    return key, f"https://www.instagram.com/{kind}/{key}/", kind


class Ledger:
    """SQLite-backed capture ledger. Cheap to open, safe to keep open."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        # FULL, not NORMAL: the cost is a few milliseconds per reel on a loop
        # that sleeps two minutes between reels, and the benefit is that a
        # power cut cannot lose the record of an upload that already happened.
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.set_meta("schema_version", str(SCHEMA_VERSION))
        self.conn.commit()

    # ── plumbing ─────────────────────────────────────────────────────────
    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except sqlite3.Error:
            pass

    def checkpoint(self):
        """Fold the write-ahead log back into the database file.

        Mandatory before the file is copied or uploaded. In WAL mode the most
        recent commits live in `<db>-wal`, not in `<db>` — so snapshotting the
        .db alone would ship a ledger that is missing exactly the reels
        captured since the last automatic checkpoint, which is the opposite of
        what a snapshot is for.
        """
        try:
            self.conn.commit()
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass

    def get_meta(self, k: str, default=None):
        row = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return row["v"] if row else default

    def set_meta(self, k: str, v: str):
        self.conn.execute(
            "INSERT INTO meta(k,v) VALUES(?,?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))

    def log(self, kind: str, text: str = "", key: str | None = None):
        self.conn.execute(
            "INSERT INTO event(at,kind,key,text) VALUES(?,?,?,?)",
            (time.time(), kind, key, text[:2000]))
        self.conn.commit()

    def events(self, limit: int = 100, after_id: int = 0) -> list:
        rows = self.conn.execute(
            "SELECT * FROM event WHERE id>? ORDER BY id DESC LIMIT ?",
            (after_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ── enqueue ──────────────────────────────────────────────────────────
    def add(self, url: str, collection: str | None = None,
            source: str = "", position: int | None = None) -> str | None:
        """Add one permalink. Returns its key, or None if not a permalink.

        Idempotent by design and safe to run over the same export twice: an
        existing row keeps its state, so re-importing after three months adds
        only what is new and never resurrects a finished item.
        """
        can = canonical(url)
        if not can:
            return None
        key, curl, kind = can
        now = time.time()
        cur = self.conn.execute("SELECT state FROM item WHERE key=?", (key,))
        row = cur.fetchone()
        if row is None:
            if position is None:
                position = self._next_position()
            self.conn.execute(
                "INSERT INTO item(key,url,kind,state,added_at,source,position) "
                "VALUES(?,?,?,?,?,?,?)",
                (key, curl, kind, QUEUED, now, source, position))
        if collection:
            self.conn.execute(
                "INSERT OR IGNORE INTO membership(key,collection) VALUES(?,?)",
                (key, collection.strip()))
        return key

    def add_many(self, items, source: str = "") -> dict:
        """Bulk import. `items` is an iterable of (url, collection).

        One transaction for the whole file — importing 5,000 links one commit
        at a time takes minutes on a spinning ledger and milliseconds here.
        """
        added = dup = bad = 0
        pos = self._next_position()
        seen_now = set()
        for url, collection in items:
            can = canonical(url)
            if not can:
                bad += 1
                continue
            key = can[0]
            existed = key in seen_now or self._exists(key)
            self.add(url, collection, source, position=pos if not existed else None)
            if existed:
                dup += 1
            else:
                added += 1
                pos += 1
                seen_now.add(key)
        self.conn.commit()
        self.log("import", f"{source}: +{added} new, {dup} already known, "
                           f"{bad} unrecognised")
        return {"added": added, "duplicate": dup, "unrecognised": bad}

    def _exists(self, key: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM item WHERE key=?", (key,)).fetchone() is not None

    def _next_position(self) -> int:
        row = self.conn.execute("SELECT MAX(position) AS m FROM item").fetchone()
        return int((row["m"] or 0)) + 1

    # ── the work loop's view ─────────────────────────────────────────────
    def claim_next(self, skip_collections=()) -> dict | None:
        """The next item to fetch, marked `fetching` before it is returned.

        Marking before returning is what makes a hard kill recoverable: the
        row says work started, `repair_stale` sees a `fetching` row with an
        old timestamp on the next boot and puts it back in the queue with its
        attempt counted.
        """
        params: list = [QUEUED, FAILED, time.time()]
        sql = ("SELECT * FROM item WHERE state IN (?,?) AND next_try_at<=? ")
        if skip_collections:
            marks = ",".join("?" * len(skip_collections))
            sql += (f"AND key NOT IN (SELECT key FROM membership "
                    f"WHERE collection IN ({marks})) ")
            params.extend(skip_collections)
        sql += "ORDER BY position ASC LIMIT 1"
        row = self.conn.execute(sql, params).fetchone()
        if row is None:
            return None
        self.conn.execute(
            "UPDATE item SET state=?, last_try_at=?, attempts=attempts+1 "
            "WHERE key=?", (FETCHING, time.time(), row["key"]))
        self.conn.commit()
        out = dict(row)
        out["state"] = FETCHING
        out["attempts"] = out["attempts"] + 1
        return out

    def repair_stale(self, older_than: float = 1800) -> int:
        """Put half-finished items back in the queue.

        Anything left `fetching` when a run starts belonged to a process that
        no longer exists. Its attempt has already been counted, so a link that
        crashes the fetcher every time still runs out of attempts instead of
        looping forever.
        """
        cutoff = time.time() - older_than
        cur = self.conn.execute(
            "UPDATE item SET state=? WHERE state=? AND (last_try_at IS NULL "
            "OR last_try_at < ?)", (QUEUED, FETCHING, cutoff))
        self.conn.commit()
        if cur.rowcount:
            self.log("repair", f"{cur.rowcount} interrupted item(s) requeued")
        return cur.rowcount

    def mark_uploaded(self, key: str, **fields):
        cols = ("msg_id", "record_msg_id", "file_id", "file_size", "sha256",
                "ext", "duration", "width", "height", "uploader", "title",
                "views", "likes", "comment_count", "comments_got", "taken_at",
                "lang")
        sets = ["state=?", "done_at=?", "last_error=NULL"]
        vals: list = [UPLOADED, time.time()]
        for c in cols:
            if c in fields and fields[c] is not None:
                sets.append(f"{c}=?")
                vals.append(fields[c])
        vals.append(key)
        self.conn.execute(f"UPDATE item SET {', '.join(sets)} WHERE key=?", vals)
        self.conn.commit()

    def mark_failed(self, key: str, error: str, retry_in: float = 900,
                    max_attempts: int = 5):
        """Record a failure and schedule the retry.

        Past `max_attempts` the item is parked as `failed` with `next_try_at`
        far in the future rather than deleted — a dead link today may be a
        cookie problem, and the row is still evidence the reel was saved.
        """
        row = self.conn.execute(
            "SELECT attempts FROM item WHERE key=?", (key,)).fetchone()
        attempts = int(row["attempts"]) if row else 1
        if attempts >= max_attempts:
            retry_in = 86400 * 30
        self.conn.execute(
            "UPDATE item SET state=?, last_error=?, next_try_at=? WHERE key=?",
            (FAILED, str(error)[:900], time.time() + retry_in, key))
        self.conn.commit()

    def mark_unavailable(self, key: str, reason: str):
        """Terminal. The post is gone from Instagram; nothing will fix that."""
        self.conn.execute(
            "UPDATE item SET state=?, last_error=?, done_at=? WHERE key=?",
            (UNAVAILABLE, str(reason)[:900], time.time(), key))
        self.conn.commit()
        self.log("unavailable", reason[:200], key)

    def requeue(self, states=(FAILED,), reset_attempts: bool = True) -> int:
        marks = ",".join("?" * len(states))
        sql = f"UPDATE item SET state=?, next_try_at=0"
        if reset_attempts:
            sql += ", attempts=0"
        sql += f" WHERE state IN ({marks})"
        cur = self.conn.execute(sql, [QUEUED, *states])
        self.conn.commit()
        return cur.rowcount

    # ── reporting ────────────────────────────────────────────────────────
    def counts(self) -> dict:
        rows = self.conn.execute(
            "SELECT state, COUNT(*) n FROM item GROUP BY state").fetchall()
        out = {r["state"]: r["n"] for r in rows}
        out["total"] = sum(out.values())
        out["remaining"] = (out.get(QUEUED, 0) + out.get(FAILED, 0)
                            + out.get(FETCHING, 0))
        return out

    def collections(self) -> list:
        rows = self.conn.execute(
            "SELECT m.collection AS name, COUNT(*) AS n, "
            "  SUM(CASE WHEN i.state='uploaded' THEN 1 ELSE 0 END) AS done "
            "FROM membership m JOIN item i ON i.key=m.key "
            "GROUP BY m.collection ORDER BY n DESC").fetchall()
        return [dict(r) for r in rows]

    def recent(self, limit: int = 40, state: str | None = None) -> list:
        if state:
            rows = self.conn.execute(
                "SELECT * FROM item WHERE state=? "
                "ORDER BY COALESCE(done_at, last_try_at, added_at) DESC LIMIT ?",
                (state, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM item ORDER BY COALESCE(done_at, last_try_at) "
                "DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def next_due_at(self, skip_collections=()) -> float | None:
        """When the earliest retryable item becomes claimable, or None if
        there is genuinely nothing left.

        `claim_next` returning None is ambiguous — it means either "the queue
        is empty" or "everything left is cooling off". The engine has to tell
        those apart, because the first means stop and the second means wait.
        """
        sql = ("SELECT MIN(next_try_at) AS t FROM item "
               "WHERE state IN (?,?)")
        params = [QUEUED, FAILED]
        if skip_collections:
            marks = ",".join("?" * len(skip_collections))
            sql += (f" AND key NOT IN (SELECT key FROM membership "
                    f"WHERE collection IN ({marks}))")
            params.extend(skip_collections)
        row = self.conn.execute(sql, params).fetchone()
        return row["t"] if row and row["t"] is not None else None

    def failures(self, limit: int = 100) -> list:
        """Everything that did not land, with `state` so the UI can tell the
        two kinds apart: `failed` will be retried on its own, `unavailable`
        means the post is gone from Instagram and no amount of waiting brings
        it back. Presenting those identically makes a clean run look broken.
        """
        rows = self.conn.execute(
            "SELECT key,url,state,attempts,last_error,last_try_at,next_try_at "
            "FROM item WHERE state IN (?,?) ORDER BY last_try_at DESC LIMIT ?",
            (FAILED, UNAVAILABLE, limit)).fetchall()
        return [dict(r) for r in rows]

    def throughput(self, window: float = 3600) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) n FROM item WHERE done_at > ?",
            (time.time() - window,)).fetchone()
        return int(row["n"])

    def export_urls(self, state: str = UPLOADED) -> list:
        rows = self.conn.execute(
            "SELECT url FROM item WHERE state=? ORDER BY position", (state,))
        return [r["url"] for r in rows]

    def adopt(self, key: str, url: str, msg_id: int, **fields):
        """Record a reel that is already in the channel.

        Used by the seeder for the reels the old Colab script uploaded. The
        row goes straight to `uploaded` without ever being queued, so those
        552 are never fetched again — which is the entire point.
        """
        can = canonical(url) or (key, url, "reel")
        now = time.time()
        self.conn.execute(
            "INSERT OR IGNORE INTO item(key,url,kind,state,added_at,source,"
            "position) VALUES(?,?,?,?,?,?,?)",
            (can[0], can[1], can[2], UPLOADED, now, "seed",
             self._next_position()))
        self.mark_uploaded(can[0], msg_id=msg_id, **fields)


def open_ledger(path: str) -> Ledger:
    return Ledger(path)


def dump_json(ledger: Ledger, path: str) -> str:
    """A human-readable mirror of the ledger, written next to the db.

    Not used by the code — it exists so that if every piece of software here
    is gone in five years, the list of what was captured is still a text file
    anyone can read.
    """
    rows = ledger.conn.execute(
        "SELECT key,url,state,uploader,msg_id,done_at FROM item "
        "ORDER BY position").fetchall()
    payload = {"schema": SCHEMA_VERSION, "written_at": time.time(),
               "items": [dict(r) for r in rows]}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path
