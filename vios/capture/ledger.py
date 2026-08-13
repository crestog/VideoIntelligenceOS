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

# Retries, and how long a row that has used them up waits before it is tried
# again anyway. Parking a row for thirty days is indistinguishable from giving
# up: the reason a fetch fails is usually a condition — expired cookies, a rate
# limit, a host refusing connections — that stops being true in hours, and a
# queue that will not look again until next month has, in practice, handed the
# problem to whoever remembers to press Requeue. Four hours, six times, is a
# full day of retrying without attention; after that the row is genuinely stuck
# and the failures panel says so rather than implying a button will fix it.
CAPTURE_REVIVE_AFTER = 4 * 3600
CAPTURE_MAX_REVIVALS = 6
CAPTURE_PARKED = 86400 * 30    # only once every revival is spent

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
    revivals      INTEGER NOT NULL DEFAULT 0,  -- automatic requeues spent
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
    lang          TEXT,

    -- The asset set (clips + manifest) this video has in the channel, if any.
    -- `assets_msg_id` is the manifest message, and the manifest is the commit
    -- point: it exists only once every clip it names has been uploaded. So a
    -- NULL here means "this video has no asset set", which is the question the
    -- backfill asks 62 times on a ledger captured before the asset set existed.
    assets_msg_id INTEGER,
    assets_clips  INTEGER,
    assets_at     REAL,
    assets_note   TEXT
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


# ── videos that were never on Instagram ──────────────────────────────────
# Dropping a video straight into the channel from a phone is the fastest way
# to add something to the archive, and until now it was the one way that did
# not work: every key in this ledger is an Instagram shortcode, so a video with
# no permalink had no identity and `canonical()` returned None for it.
#
# The message id is the identity instead. It is already unique within the
# channel, already permanent, and already the thing every downloader here takes
# — so `up_4471` needs no lookup table and survives a ledger rebuilt from
# nothing but a channel scan. The `up_` prefix keeps it from ever colliding
# with a shortcode: Instagram's alphabet includes `_`, but a shortcode is 11
# characters of base64 and never starts with `up_` followed by digits only.
UPLOAD_PREFIX = "up_"
UPLOAD_KIND = "upload"
UPLOAD_SOURCE = "telegram-upload"

_UPLOAD_KEY = re.compile(r"^up_(\d+)$")


def upload_key(msg_id) -> str:
    """The ledger key for a bare video sitting at `msg_id` in the channel."""
    return f"{UPLOAD_PREFIX}{int(msg_id)}"


def is_upload(key: str) -> bool:
    return bool(_UPLOAD_KEY.match(str(key or "")))


def upload_msg_id(key: str) -> int:
    """The message id back out of an upload key, or 0."""
    m = _UPLOAD_KEY.match(str(key or ""))
    return int(m.group(1)) if m else 0


class Ledger:
    """SQLite-backed capture ledger. Cheap to open, safe to keep open."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()
        # FULL, not NORMAL: the cost is a few milliseconds per reel on a loop
        # that sleeps two minutes between reels, and the benefit is that a
        # power cut cannot lose the record of an upload that already happened.
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.set_meta("schema_version", str(SCHEMA_VERSION))
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a ledger was first written.

        `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a new
        column in `_SCHEMA` never reaches a ledger that predates it. Opening
        rather than rebuilding matters here more than anywhere else in the
        system: this table is the record of which reels are already in Telegram,
        and losing it means re-uploading thousands of files.
        """
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(item)")}
        for name, spec in (("revivals", "INTEGER NOT NULL DEFAULT 0"),
                           ("assets_msg_id", "INTEGER"),
                           ("assets_clips", "INTEGER"),
                           ("assets_at", "REAL"),
                           ("assets_note", "TEXT")):
            if name not in have:
                self.conn.execute(f"ALTER TABLE item ADD COLUMN {name} {spec}")

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

        The input is *pairs*, and one reel saved into three collections arrives
        as three pairs. So the first thing this does is fold the pairs down to
        one entry per reel carrying a set of collections. Counting the pairs
        instead is what produced the "6,879 already known" on a brand new
        ledger: those were not reels anybody had captured, they were the second
        and third collection membership of reels added moments earlier in the
        same loop. A count that says "already known" has to mean *the archive
        already has this*, or it is worse than no count at all.
        """
        folded: dict = {}
        order: list = []
        bad = 0
        for url, collection in items:
            can = canonical(url)
            if not can:
                bad += 1
                continue
            key = can[0]
            if key not in folded:
                folded[key] = (can[1], set())
                order.append(key)
            if collection:
                folded[key][1].add(collection.strip())

        pos = self._next_position()
        added = known = done = links = 0
        for key in order:
            url, cols = folded[key]
            row = self.conn.execute(
                "SELECT state FROM item WHERE key=?", (key,)).fetchone()
            if row is None:
                self.add(url, None, source, position=pos)
                pos += 1
                added += 1
            else:
                known += 1
                if row["state"] in TERMINAL:
                    done += 1
            # Memberships are added for new and existing reels alike: finding
            # out that reel X is also in "recipes" is new information even when
            # reel X was captured in March.
            for col in sorted(cols):
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO membership(key,collection) "
                    "VALUES(?,?)", (key, col))
                links += cur.rowcount or 0
        self.conn.commit()
        self.log("import",
                 f"{source}: {len(order)} reels in the file — {added} new to "
                 f"the queue, {known} already in the ledger ({done} of those "
                 f"already captured), {links} new collection tags, "
                 f"{bad} lines that were not permalinks")
        return {"added": added, "duplicate": known, "captured": done,
                "unique": len(order), "memberships": links,
                "unrecognised": bad}

    def _exists(self, key: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM item WHERE key=?", (key,)).fetchone() is not None

    # ── which channel do these message ids mean? ──────────────────────────
    def bind_channel(self, channel) -> dict:
        """Record which channel this ledger's message ids point into.

        A `msg_id` is meaningless on its own: message 38 exists in every
        channel and is a different message in each. So the first time a channel
        is configured, the ledger remembers it; from then on, a *different*
        channel is a fact worth shouting about, because every `uploaded` row
        now names a message that channel does not have.

        That is the failure the processing plane reported as
        "could not download the original from Telegram (message 38)". Nothing
        was wrong with message 38. It was in the previous channel.

        Returns one of three verdicts:
          {"state": "bound"}    first time, nothing to check against
          {"state": "same"}     matches — the normal case, every run
          {"state": "changed"}  with counts, and `stale` rows to be reset
        """
        new = str(channel or "").strip()
        if not new:
            return {"state": "unset"}
        old = str(self.get_meta("channel_id", "") or "").strip()
        if not old:
            self.set_meta("channel_id", new)
            self.conn.commit()
            return {"state": "bound", "channel": new}
        if old == new:
            return {"state": "same", "channel": new}

        stale = int(self.conn.execute(
            "SELECT COUNT(*) AS n FROM item WHERE state=? AND msg_id IS NOT NULL",
            (UPLOADED,)).fetchone()["n"])
        return {"state": "changed", "was": old, "now": new, "stale": stale}

    def rebind_channel(self, channel, requeue: bool = True) -> dict:
        """Point the ledger at a new channel, and deal with the old ids.

        `requeue=True` moves every uploaded row back to the queue with its
        message ids cleared: the bytes are not in the new channel, so as far as
        the archive is concerned they were never captured, and the honest thing
        is to fetch them again. `requeue=False` keeps the rows and only rebinds
        — correct when the channel was *migrated* (a group upgraded to a
        supergroup keeps its history under a new id) and the messages really
        are still there.
        """
        new = str(channel or "").strip()
        was = str(self.get_meta("channel_id", "") or "")
        moved = 0
        if requeue:
            cur = self.conn.execute(
                "UPDATE item SET state=?, msg_id=NULL, record_msg_id=NULL, "
                "file_id=NULL, done_at=NULL, next_try_at=0, attempts=0, "
                "last_error='channel changed; the old message ids were not "
                "valid here' WHERE state=? AND msg_id IS NOT NULL",
                (QUEUED, UPLOADED))
            moved = cur.rowcount or 0
        self.set_meta("channel_id", new)
        self.conn.commit()
        self.log("rebind", f"channel {was or '(none)'} → {new}; "
                           f"{moved} captured reels returned to the queue"
                 if requeue else
                 f"channel {was or '(none)'} → {new}; message ids kept")
        return {"was": was, "now": new, "requeued": moved}

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
                "lang", "assets_msg_id")
        sets = ["state=?", "done_at=?", "last_error=NULL"]
        vals: list = [UPLOADED, time.time()]
        for c in cols:
            if c in fields and fields[c] is not None:
                sets.append(f"{c}=?")
                vals.append(fields[c])
        vals.append(key)
        self.conn.execute(f"UPDATE item SET {', '.join(sets)} WHERE key=?", vals)
        self.conn.commit()

    # ── the asset set ────────────────────────────────────────────────────
    def mark_assets(self, key: str, manifest_msg_id: int, clips: int = 0,
                    note: str = "") -> None:
        """Record that this video's clips and manifest are in the channel.

        Written by the capture loop the moment it publishes an asset set, and by
        the backfill for the videos that were captured before asset sets existed.
        A row with `assets_msg_id` set is never offered to the backfill again,
        which is what makes a run that dies at video 40 resume at 41 rather than
        re-uploading 40 videos' worth of clips.

        A manifest id of 0 is not recorded. `publish_assets` returns 0 when the
        manifest upload itself failed, and treating that as done would leave a
        video whose clips are in the channel but whose index is not — unreachable
        for Atlas and invisible to the retry.
        """
        if not manifest_msg_id:
            if note:
                self.conn.execute(
                    "UPDATE item SET assets_note=? WHERE key=?",
                    (str(note)[:900], key))
                self.conn.commit()
            return
        self.conn.execute(
            "UPDATE item SET assets_msg_id=?, assets_clips=?, assets_at=?, "
            "assets_note=? WHERE key=?",
            (int(manifest_msg_id), int(clips or 0), time.time(),
             str(note)[:900] or None, key))
        self.conn.commit()

    def needs_assets(self, limit: int = 500) -> list:
        """Uploaded videos with no asset set, oldest first.

        Photos are excluded — there is nothing to segment — and so is any row
        whose message id is unknown, because the clips have to be threaded under
        the video's own message and there is nothing to thread them under.
        Oldest first so the backfill's progress matches the archive's order and
        an interrupted run is easy to reason about.
        """
        rows = self.conn.execute(
            "SELECT key,url,kind,msg_id,record_msg_id,file_id,file_size,"
            "       duration,width,height,uploader,title,sha256,ext,taken_at,"
            "       assets_note "
            "FROM item WHERE state=? AND COALESCE(assets_msg_id,0)=0 "
            "  AND COALESCE(msg_id,0)>0 AND COALESCE(ext,'')<>'photo' "
            "ORDER BY COALESCE(done_at, added_at) ASC LIMIT ?",
            (UPLOADED, int(limit))).fetchall()
        return [dict(r) for r in rows]

    def asset_counts(self) -> dict:
        """How much of the archive is playable the fast way."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS videos, "
            "  SUM(CASE WHEN COALESCE(assets_msg_id,0)>0 THEN 1 ELSE 0 END) "
            "    AS with_assets, "
            "  SUM(COALESCE(assets_clips,0)) AS clips "
            "FROM item WHERE state=? AND COALESCE(msg_id,0)>0 "
            "  AND COALESCE(ext,'')<>'photo'", (UPLOADED,)).fetchone()
        videos = int(row["videos"] or 0)
        have = int(row["with_assets"] or 0)
        return {"videos": videos, "with_assets": have,
                "without_assets": max(videos - have, 0),
                "clips": int(row["clips"] or 0)}

    def mark_failed(self, key: str, error: str, retry_in: float = 900,
                    max_attempts: int = 5):
        """Record a failure and schedule the retry.

        Past `max_attempts` the row is not parked for a month; it spends one of
        its revivals, its attempt count is cleared, and it comes back in four
        hours. A fetch fails for a condition that usually stops being true in
        hours — expired cookies, a rate limit, a host refusing connections — and
        a queue that will not look again until next month has in practice handed
        the problem to whoever remembers to press Requeue.

        Only when every revival is spent is the row parked far out. It is never
        deleted: a dead link today may be a cookie problem tomorrow, and the row
        is still evidence the reel was saved.
        """
        row = self.conn.execute(
            "SELECT attempts, revivals FROM item WHERE key=?", (key,)).fetchone()
        attempts = int(row["attempts"]) if row else 1
        revivals = int(row["revivals"] or 0) if row else 0

        if attempts >= max_attempts:
            if revivals < CAPTURE_MAX_REVIVALS:
                # Spend a revival: back in four hours with a clean attempt count.
                self.conn.execute(
                    "UPDATE item SET state=?, last_error=?, next_try_at=?, "
                    "attempts=0, revivals=revivals+1 WHERE key=?",
                    (FAILED, str(error)[:900],
                     time.time() + CAPTURE_REVIVE_AFTER, key))
                self.conn.commit()
                return
            retry_in = CAPTURE_PARKED

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
        """Put rows back in the queue now. The manual override.

        Still worth having with automatic revival in place: it is how you say "I
        just fixed the cookies, do not wait four hours", and it is the only way
        back for a row that has spent every revival. Revivals are cleared too —
        an operator saying "try again" means a full fresh ladder, not one more
        attempt against an exhausted budget.
        """
        marks = ",".join("?" * len(states))
        sql = "UPDATE item SET state=?, next_try_at=0, revivals=0"
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

        A bare uploaded video takes the same path. Its key is `up_<msg_id>` and
        its url is empty, so `canonical()` cannot help — and the old fallback
        `(key, url, "reel")` would have labelled a phone upload a reel and,
        worse, kept whatever string was passed as `url`. `is_upload` is checked
        first so the kind is honest: nothing downstream should be able to look
        at this row and conclude Instagram has a copy.
        """
        if is_upload(key):
            kind, url = UPLOAD_KIND, ""
        else:
            can = canonical(url) or (key, url, "reel")
            key, url, kind = can
        now = time.time()
        self.conn.execute(
            "INSERT OR IGNORE INTO item(key,url,kind,state,added_at,source,"
            "position) VALUES(?,?,?,?,?,?,?)",
            (key, url, kind, UPLOADED, now,
             UPLOAD_SOURCE if kind == UPLOAD_KIND else "seed",
             self._next_position()))
        self.mark_uploaded(key, msg_id=msg_id, **fields)

    def adopt_upload(self, msg_id: int, **fields) -> str:
        """Record a bare video that was dropped into the channel by hand.

        Returns the key. The bytes are already in Telegram — that is the whole
        difference from every other row in this table, which describes work
        still to be done or work whose result was uploaded here. Nothing needs
        fetching, nothing needs an Instagram request, and the row exists solely
        so the processing plane can find the video and never look at it twice.
        """
        key = upload_key(msg_id)
        self.adopt(key, "", int(msg_id), **fields)
        return key


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
