"""
vios.process.coverage — who has processed what, and who is doing it now.

The unit of work is **(video, component)**, never "process this video". That
distinction is the whole reason a second model can be added in month six without
redoing month one. `process_video(key)` is a function whose meaning changes
every time the pipeline changes; `(key, 'narrate:qwen3vl-8b')` means the same
thing forever.

Two mechanisms, and they solve different problems:

**Claiming with a lease** solves *this session might die*. A Kaggle notebook is
killed at the 12-hour wall with no warning and no chance to clean up. A row
marked `running` by a dead worker would be stranded forever, so a claim carries
an expiry; when it lapses the row is available again. No daemon, no heartbeat
protocol, no Redis — a timestamp and a `WHERE` clause.

**Static partitioning** solves *ten accounts must not collide*. Workers do not
negotiate, because they cannot: they are ten separate notebooks on ten separate
accounts with no shared network. Instead each is told "you are worker 3 of 10"
and only ever touches videos whose partition hash mod 10 is 3. Two workers can
never claim the same row because they can never see the same row. The cost is
that an idle worker cannot help a busy one; the benefit is that the system needs
no coordination layer at all, which at ten free-tier notebooks is the right
trade by a wide margin.

Leases still matter inside a partition, because the same account resumes its own
slice after being killed.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import time
import uuid

QUEUED, RUNNING, DONE, FAILED, SKIPPED = (
    "queued", "running", "done", "failed", "skipped")
TERMINAL = (DONE, SKIPPED)

# 40 minutes. Long enough that a slow 32B pass over a 3-minute reel does not
# lose its own lease mid-run; short enough that a killed session's work returns
# to the pool inside one coffee break.
LEASE_SECONDS = 40 * 60

# Attempts before a component is marked failed and stops being retried. Three,
# because the failures that survive three tries are structural — a corrupt file,
# an OOM on a video that is genuinely too long — and grinding on them costs GPU
# hours that other videos would use better.
MAX_ATTEMPTS = 3

# How long a row waits after its Nth failed attempt before it may be claimed
# again. Three attempts back to back are not three chances: a pass that fails in
# two seconds — a missing package, a peer id the library rejects before it ever
# reaches the network — burns all three inside ten seconds and retires, and the
# operator is left pressing Requeue for a condition that would have cleared on
# its own. Spacing the attempts is what makes them independent.
RETRY_BACKOFF = (10 * 60, 45 * 60)      # after attempt 1, after attempt 2

# A row that has burned its attempts is not given up on; it is set aside and
# tried again later, up to this many times. This is the automatic form of the
# Requeue button: same effect, on a schedule, bounded so a genuinely broken
# video cannot consume the queue forever. Six revivals at four hours covers a
# full day of Kaggle sessions, which is long enough for "the model download was
# rate-limited" or "Telegram was refusing connections" to have stopped being
# true, and short enough that the work resumes without anyone watching.
REVIVE_AFTER = 4 * 3600
MAX_REVIVALS = 6

SCHEMA = """
CREATE TABLE IF NOT EXISTS coverage (
    video_key   TEXT NOT NULL,
    component   TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'queued',
    attempts    INTEGER DEFAULT 0,
    worker      TEXT,
    token       TEXT,           -- identifies one claim() call, exactly
    lease_until REAL DEFAULT 0,
    next_try_at REAL DEFAULT 0, -- backoff: not claimable before this moment
    revivals    INTEGER DEFAULT 0,  -- automatic requeues already spent
    started_at  REAL,
    done_at     REAL,
    seconds     REAL,          -- how long it actually took, for the ETA
    claims      INTEGER,       -- what it produced, for the "is this pass
    vectors     INTEGER,       --   actually working" check
    observer_id TEXT,
    last_error  TEXT,
    checkpoint  TEXT,          -- JSON: how far a long job got
    PRIMARY KEY (video_key, component)
);
"""

# Created after the migration, never with the table: an index over `next_try_at`
# cannot be built on a database whose `coverage` predates that column, and
# `CREATE TABLE IF NOT EXISTS` is a silent no-op there rather than an error. The
# ordering — table, then columns, then indexes — is what makes an old database
# open instead of raising "no such column" on the first line of the schema.
INDEXES = """
CREATE INDEX IF NOT EXISTS ix_cov_claim ON coverage(component, state, lease_until);
CREATE INDEX IF NOT EXISTS ix_cov_state ON coverage(state, component);
CREATE INDEX IF NOT EXISTS ix_cov_token ON coverage(token);
CREATE INDEX IF NOT EXISTS ix_cov_retry ON coverage(state, next_try_at);
"""

# Columns added after the first release. A database written by an earlier build
# is opened, not discarded — it holds the coverage of every video processed so
# far, and re-earning that would cost the GPU hours it originally took.
MIGRATIONS = (
    ("next_try_at", "REAL DEFAULT 0"),
    ("revivals", "INTEGER DEFAULT 0"),
)


def worker_id() -> str:
    """A name for this process that survives a restart but distinguishes two
    notebooks. Hostname plus pid: on Kaggle the hostname is the session's, so a
    resumed session gets a new id and does not silently inherit stale leases."""
    return f"{socket.gethostname()[:24]}:{os.getpid()}"


class Coverage:
    """The job table. Shares the evidence store's connection on purpose.

    One file means one snapshot: uploading the database captures both what was
    learned and what is left to learn, in a consistent state, without a
    two-phase commit between two files that could disagree after a crash.
    """

    def __init__(self, conn: sqlite3.Connection,
                 partitions: int = 1, index: int = 0,
                 worker: str = ""):
        if partitions < 1 or not (0 <= index < partitions):
            raise ValueError(
                f"worker {index} of {partitions} is not a valid slice; "
                f"index must be 0 <= i < partitions")
        self.conn = conn
        self.partitions = partitions
        self.index = index
        self.worker = worker or worker_id()
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.executescript(INDEXES)
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after the first release.

        A database written by an earlier build of the processing engine is
        opened, not discarded — it holds the coverage of every video processed
        so far, and re-earning that would cost the GPU hours it originally took.
        Adding a column to an existing table is safe and instant; SQLite makes
        the new column `NULL` in every existing row (or the specified DEFAULT),
        and writes the new shape to every row inserted from here on.
        """
        cur = self.conn.execute("PRAGMA table_info(coverage)")
        have = {r["name"] for r in cur.fetchall()}
        for name, spec in MIGRATIONS:
            if name not in have:
                self.conn.execute(f"ALTER TABLE coverage ADD COLUMN {name} {spec}")

    # ── filling the table ───────────────────────────────────────────────
    def plan(self, components: list, video_keys: list | None = None) -> int:
        """Ensure a row exists for every (video, component) pair.

        Idempotent, and cheap enough to call at the top of every sweep — which
        is exactly when it should be called, because that is when a newly
        enabled component needs rows for videos captured last week.
        """
        if not components:
            return 0
        if video_keys is None:
            video_keys = [r[0] for r in self.conn.execute(
                "SELECT video_key FROM video")]
        rows = [(k, c) for k in video_keys for c in components]
        if not rows:
            return 0
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO coverage(video_key,component) VALUES(?,?)",
            rows)
        self.conn.commit()
        return max(cur.rowcount, 0)

    def _mine(self) -> str:
        """The SQL fragment restricting a query to this worker's slice."""
        if self.partitions <= 1:
            return ""
        return (" AND video_key IN (SELECT video_key FROM video "
                f"WHERE (partition % {int(self.partitions)}) = {int(self.index)})")

    # ── claiming ────────────────────────────────────────────────────────
    def claim(self, component: str, n: int = 1) -> list:
        """Take up to `n` videos for this component, with a lease.

        The claim is a single UPDATE over a subquery rather than a SELECT
        followed by an UPDATE. Two threads in the same process racing between
        those two statements would both believe they own the row; one statement
        cannot race with itself, because SQLite serialises writers.

        The read-back filters on a **token unique to this call**, not on "my
        rows, newest first" and not on the timestamp the update wrote. Both of
        those are wrong for the same reason: this worker may already hold rows
        from an earlier claim — a per-video loop claims one at a time while the
        previous is still marked running — and on Windows `time.time()` has
        ~16 ms resolution, so two claims in the same tick get an identical
        `started_at`. Measured: the second claim re-reported all four rows from
        the first, which would have processed every one of them twice.

        Respects `next_try_at`: a failed row with a future `next_try_at` is not
        claimable until that time arrives — the automatic backoff that answers
        "every failed pass sat behind a manual Requeue button when they want
        automatic retry". A `queued` row is always claimable immediately.
        """
        now = time.time()
        token = uuid.uuid4().hex
        sql = (
            "UPDATE coverage SET state=?, worker=?, token=?, lease_until=?, "
            "started_at=?, attempts=attempts+1 "
            "WHERE rowid IN (SELECT rowid FROM coverage "
            "  WHERE component=? AND ("
            "     (state=? AND next_try_at <= ?) OR (state=? AND lease_until < ?) "
            "     OR (state=? AND attempts < ? AND next_try_at <= ?)) "
            + self._mine() +
            "  ORDER BY attempts, video_key LIMIT ?)")
        self.conn.execute(sql, (
            RUNNING, self.worker, token, now + LEASE_SECONDS, now,
            component, QUEUED, now, RUNNING, now, FAILED, MAX_ATTEMPTS, now, n))
        self.conn.commit()
        return [r["video_key"] for r in self.conn.execute(
            "SELECT video_key FROM coverage WHERE token=? ORDER BY video_key",
            (token,))]

    def candidates(self, components: list, limit: int = 64) -> list:
        """Videos with outstanding work in any of these components.

        `claim` is component-major: it answers "give me videos for this pass".
        The rotation loop needs the transpose — a video, then every pass in the
        loaded cohort that still owes it something — because the source mp4 has
        to come down from Telegram and be decoded once per video, not once per
        pass. Twenty-nine components fetching the same reel twenty-nine times
        would spend more of the sweep downloading than inferring.

        This claims nothing. It is the cheap read that tells the loop which
        video to pick up next; `claim_for` is what takes ownership.

        Respects `next_try_at`: a failed row with a future `next_try_at` is not
        a candidate until that time arrives.
        """
        if not components:
            return []
        now = time.time()
        marks = ",".join("?" * len(components))
        sql = (
            "SELECT video_key, MIN(attempts) AS a FROM coverage "
            f"WHERE component IN ({marks}) AND ("
            "   (state=? AND next_try_at <= ?) OR (state=? AND lease_until < ?) "
            "   OR (state=? AND attempts < ? AND next_try_at <= ?))"
            + self._mine() +
            " GROUP BY video_key ORDER BY a, video_key LIMIT ?")
        args = list(components) + [QUEUED, now, RUNNING, now, FAILED,
                                   MAX_ATTEMPTS, now, int(limit)]
        return [r["video_key"] for r in self.conn.execute(sql, args)]

    def claim_for(self, video_key: str, components: list) -> list:
        """Take this video's claimable rows among these components at once.

        One UPDATE, one token, the same discipline as `claim` and for the same
        reason. Claiming the whole set up front rather than pass by pass means
        the video is either mine for this cohort or someone else's, instead of
        being contended twenty-nine separate times while I hold it open.

        Returns the components actually taken — never assume it is everything
        asked for, since another worker's lapsed lease may have been renewed
        between `candidates` and here.

        Respects `next_try_at`: a failed row with a future `next_try_at` is not
        claimable until that time arrives.
        """
        if not components:
            return []
        now = time.time()
        token = uuid.uuid4().hex
        marks = ",".join("?" * len(components))
        sql = (
            "UPDATE coverage SET state=?, worker=?, token=?, lease_until=?, "
            "started_at=?, attempts=attempts+1 "
            "WHERE rowid IN (SELECT rowid FROM coverage "
            f"  WHERE video_key=? AND component IN ({marks}) AND ("
            "     (state=? AND next_try_at <= ?) OR (state=? AND lease_until < ?) "
            "     OR (state=? AND attempts < ? AND next_try_at <= ?))"
            + self._mine() + ")")
        args = ([RUNNING, self.worker, token, now + LEASE_SECONDS, now,
                 video_key] + list(components) +
                [QUEUED, now, RUNNING, now, FAILED, MAX_ATTEMPTS, now])
        self.conn.execute(sql, args)
        self.conn.commit()
        return [r["component"] for r in self.conn.execute(
            "SELECT component FROM coverage WHERE token=?", (token,))]

    def renew(self, video_key: str, component: str,
              checkpoint: str = "") -> None:
        """Push the lease out. Called by long jobs between frames, so a 32B pass
        over a three-minute reel does not have its own work stolen."""
        self.conn.execute(
            "UPDATE coverage SET lease_until=?, checkpoint=COALESCE(?,checkpoint) "
            "WHERE video_key=? AND component=? AND worker=?",
            (time.time() + LEASE_SECONDS, checkpoint or None,
             video_key, component, self.worker))
        self.conn.commit()

    def release(self, video_key: str, component: str) -> None:
        """Give a row back untouched — the pause path. Note `attempts` is
        decremented: pausing is not a failed attempt, and letting it count would
        mean three pauses retire a video that was never actually tried."""
        self.conn.execute(
            "UPDATE coverage SET state=?, worker=NULL, token=NULL, lease_until=0, "
            "attempts=MAX(attempts-1,0) WHERE video_key=? AND component=? "
            "AND state=?", (QUEUED, video_key, component, RUNNING))
        self.conn.commit()

    # ── finishing ───────────────────────────────────────────────────────
    def done(self, video_key: str, component: str, seconds: float,
             claims: int = 0, vectors: int = 0, observer_id: str = "") -> None:
        self.conn.execute(
            "UPDATE coverage SET state=?, done_at=?, seconds=?, claims=?, "
            "vectors=?, observer_id=?, lease_until=0, token=NULL, "
            "last_error=NULL, checkpoint=NULL WHERE video_key=? AND component=?",
            (DONE, time.time(), seconds, claims, vectors, observer_id,
             video_key, component))
        self.conn.commit()

    def fail(self, video_key: str, component: str, error: str) -> str:
        """Record a failure. Returns the resulting state.

        A row that has burned its attempts is not given up on; it is set aside
        and tried again later. The backoff and revival together answer "every
        failed pass sat behind a manual Requeue button when they want automatic
        retry". A transient issue — missing dependency, rate-limited model
        download, Telegram refusing connections — clears on its own without the
        operator watching.
        """
        row = self.conn.execute(
            "SELECT attempts FROM coverage WHERE video_key=? AND component=?",
            (video_key, component)).fetchone()
        attempts = row["attempts"] if row else MAX_ATTEMPTS
        state = FAILED if attempts >= MAX_ATTEMPTS else QUEUED

        # Backoff between attempts: try → wait 10 min → try → wait 45 min → try
        if state == QUEUED and attempts > 0:
            delay = RETRY_BACKOFF[min(attempts - 1, len(RETRY_BACKOFF) - 1)]
            next_try = time.time() + delay
        # Terminal failure: wait REVIVE_AFTER before a revival requeues it
        elif state == FAILED:
            next_try = time.time() + REVIVE_AFTER
        else:
            next_try = 0

        self.conn.execute(
            "UPDATE coverage SET state=?, last_error=?, lease_until=0, "
            "next_try_at=?, worker=NULL, token=NULL WHERE video_key=? AND component=?",
            (state, str(error)[:800], next_try, video_key, component))
        self.conn.commit()
        return state

    def skip(self, video_key: str, component: str, why: str) -> None:
        """Terminal, but not a failure: no audio means no transcription, and
        that is the planner working correctly. Kept distinct from `done` so the
        coverage matrix does not claim a pass ran when it was declined."""
        self.conn.execute(
            "UPDATE coverage SET state=?, done_at=?, last_error=?, "
            "lease_until=0, worker=NULL, token=NULL WHERE video_key=? AND component=?",
            (SKIPPED, time.time(), str(why)[:200], video_key, component))
        self.conn.commit()

    # ── reading ─────────────────────────────────────────────────────────
    def pending(self, component: str) -> int:
        now = time.time()
        return self.conn.execute(
            "SELECT COUNT(*) FROM coverage WHERE component=? AND ("
            "(state=? AND next_try_at <= ?) OR (state=? AND lease_until < ?) OR "
            "(state=? AND attempts < ? AND next_try_at <= ?))" + self._mine(),
            (component, QUEUED, now, RUNNING, now, FAILED, MAX_ATTEMPTS, now)
        ).fetchone()[0]

    def deferred(self) -> dict:
        """Work that exists but is waiting on a clock, and when it is due.

        Without this the sweep would look at "nothing claimable right now" and
        announce that it had finished, when what is true is that everything left
        is in a backoff window. Returns {"rows": n, "due_at": epoch|0}.
        """
        now = time.time()
        row = self.conn.execute(
            "SELECT COUNT(*) n, MIN(next_try_at) due FROM coverage "
            "WHERE next_try_at > ? AND ("
            "  state=? OR (state=? AND revivals < ?))" + self._mine(),
            (now, QUEUED, FAILED, MAX_REVIVALS)).fetchone()
        return {"rows": row["n"] or 0, "due_at": row["due"] or 0.0}

    def counts(self, component: str = "") -> dict:
        sql = "SELECT state, COUNT(*) n FROM coverage"
        args: list = []
        if component:
            sql += " WHERE component=?"
            args.append(component)
        sql += " GROUP BY state"
        out = {s: 0 for s in (QUEUED, RUNNING, DONE, FAILED, SKIPPED)}
        for r in self.conn.execute(sql, args):
            out[r["state"]] = r["n"]
        out["total"] = sum(out.values())
        return out

    def matrix(self) -> list:
        """Coverage per component: the heat grid the tab draws.

        `seconds` is the measured median-ish mean of completed runs, not an
        estimate from a config file. An ETA computed from what this machine
        actually did is the only kind worth showing — a T4 under a noisy
        neighbour is 30% slower than the same T4 alone, and no static table
        knows that.
        """
        rows = []
        for r in self.conn.execute(
                "SELECT component, state, COUNT(*) n, AVG(seconds) s, "
                "SUM(claims) c, SUM(vectors) v FROM coverage "
                "GROUP BY component, state"):
            rows.append(dict(r))
        by: dict = {}
        for r in rows:
            e = by.setdefault(r["component"], {
                "component": r["component"], "total": 0, "claims": 0,
                "vectors": 0, "seconds": 0.0,
                QUEUED: 0, RUNNING: 0, DONE: 0, FAILED: 0, SKIPPED: 0})
            e[r["state"]] = r["n"]
            e["total"] += r["n"]
            e["claims"] += r["c"] or 0
            e["vectors"] += r["v"] or 0
            if r["state"] == DONE and r["s"]:
                e["seconds"] = round(r["s"], 2)
        for e in by.values():
            left = e[QUEUED] + e[RUNNING] + e[FAILED]
            e["remaining"] = left
            e["percent"] = round(100.0 * (e[DONE] + e[SKIPPED]) /
                                 max(e["total"], 1), 1)
            e["eta_seconds"] = round(left * e["seconds"], 1) if e["seconds"] else None
        return sorted(by.values(), key=lambda e: e["component"])

    def running(self) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM coverage WHERE state=? ORDER BY started_at", (RUNNING,))]

    def failures(self, limit: int = 100) -> list:
        """Failures grouped by component and message, with their retry standing.

        `retrying` vs `needs_attention` is the distinction that matters to
        whoever is reading the panel: the first will clear itself and wants no
        action, the second has spent every automatic retry and is the only kind
        worth interrupting someone for.
        """
        now = time.time()
        rows = []
        for r in self.conn.execute(
                "SELECT component, last_error, COUNT(*) n, "
                "GROUP_CONCAT(video_key) keys, MIN(next_try_at) due, "
                "MAX(revivals) revivals, "
                "SUM(CASE WHEN revivals >= ? THEN 1 ELSE 0 END) stuck "
                "FROM coverage WHERE state=? "
                "GROUP BY component, SUBSTR(last_error,1,80) "
                "ORDER BY n DESC LIMIT ?", (MAX_REVIVALS, FAILED, limit)):
            e = dict(r)
            e["needs_attention"] = e.pop("stuck", 0) or 0
            e["retrying"] = e["n"] - e["needs_attention"]
            due = e.pop("due", 0) or 0
            e["retry_in"] = round(max(due - now, 0)) if due and e["retrying"] else 0
            rows.append(e)
        return rows

    def retry_state(self) -> dict:
        """What the automatic retry is holding, and when it acts next.

        The tab needs this to stop lying in the other direction. Before the
        backoff existed, a failed row was visibly stuck and the operator knew to
        press Requeue; now it will clear itself, and a panel that showed only a
        red count would send them looking for a button they no longer need. So:
        how many rows are waiting on a clock, when the earliest one is due, and
        how many have spent every revival and genuinely do need a person.
        """
        now = time.time()
        d = self.deferred()
        stuck = self.conn.execute(
            "SELECT COUNT(*) FROM coverage WHERE state=? AND revivals >= ?"
            + self._mine(), (FAILED, MAX_REVIVALS)).fetchone()[0]
        revived = self.conn.execute(
            "SELECT COUNT(*) FROM coverage WHERE revivals > 0" + self._mine()
        ).fetchone()[0]
        return {
            "waiting": d["rows"],
            "due_at": d["due_at"],
            "due_in": round(max(d["due_at"] - now, 0)) if d["due_at"] else 0,
            "needs_attention": stuck,
            "ever_revived": revived,
            "max_revivals": MAX_REVIVALS,
            "max_attempts": MAX_ATTEMPTS,
        }

    def for_video(self, video_key: str) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM coverage WHERE video_key=? ORDER BY component",
            (video_key,))]

    # ── operator actions ────────────────────────────────────────────────
    def requeue(self, component: str = "", state: str = FAILED) -> int:
        """Reset rows to queued and clear their attempt count.

        The button for "I fixed the thing that was breaking it". Resetting
        attempts is the point — leaving them at 3 means the retry is claimed
        once and immediately retired again.
        """
        sql = ("UPDATE coverage SET state=?, attempts=0, next_try_at=0, "
               "last_error=NULL, worker=NULL, token=NULL, lease_until=0 "
               "WHERE state=?")
        args: list = [QUEUED, state]
        if component:
            sql += " AND component=?"
            args.append(component)
        cur = self.conn.execute(sql, args)
        self.conn.commit()
        return max(cur.rowcount, 0)

    def reset_component(self, component: str) -> int:
        """Forget a component's coverage entirely, so it sweeps again.

        Used when a model is upgraded: the old claims stay in the evidence store
        under the old observer id, the new run writes new ones under a new id,
        and the interface can show both. Nothing is destroyed by asking a
        question again.
        """
        cur = self.conn.execute(
            "UPDATE coverage SET state=?, attempts=0, revivals=0, worker=NULL, "
            "token=NULL, lease_until=0, next_try_at=0, done_at=NULL, seconds=NULL, "
            "last_error=NULL, checkpoint=NULL WHERE component=?",
            (QUEUED, component))
        self.conn.commit()
        return max(cur.rowcount, 0)

    def drop_component(self, component: str) -> int:
        cur = self.conn.execute("DELETE FROM coverage WHERE component=?",
                                (component,))
        self.conn.commit()
        return max(cur.rowcount, 0)

    def reclaim_stale(self) -> int:
        """Return expired leases to the pool. Called once at startup.

        The lease already handles this lazily — an expired row is claimable —
        but doing it eagerly at boot makes the numbers in the tab honest from
        the first second, instead of showing 200 videos "running" on a worker
        that died last Tuesday.
        """
        cur = self.conn.execute(
            "UPDATE coverage SET state=?, worker=NULL, token=NULL, lease_until=0 "
            "WHERE state=? AND lease_until < ?",
            (QUEUED, RUNNING, time.time()))
        self.conn.commit()
        return max(cur.rowcount, 0)

    def revive_failed(self) -> dict:
        """Requeue rows that ran out of attempts, once their wait has elapsed.

        This is the Requeue button, on a timer. The operator pressing it is
        saying "the thing that broke this is probably fixed now"; four hours of
        elapsed time says the same thing with less certainty and no attention,
        which for a transient fault — a rate-limited model download, a refused
        Telegram connection, a package that was missing until the next session
        installed it — is enough.

        Bounded by `MAX_REVIVALS` so a genuinely broken video cannot keep
        re-entering the queue forever. `last_error` is kept, not overwritten:
        after six revivals the operator needs to see what it actually said.

        Returns {"revived": n, "exhausted": n}, where `exhausted` is the number
        of rows that have spent every revival and now really do need a person.
        """
        now = time.time()
        cur = self.conn.execute(
            "UPDATE coverage SET state=?, attempts=0, next_try_at=0, "
            "revivals=revivals+1, worker=NULL, token=NULL, lease_until=0 "
            "WHERE state=? AND next_try_at <= ? AND revivals < ?" + self._mine(),
            (QUEUED, FAILED, now, MAX_REVIVALS))
        revived = max(cur.rowcount, 0)
        self.conn.commit()
        exhausted = self.conn.execute(
            "SELECT COUNT(*) FROM coverage WHERE state=? AND revivals >= ?"
            + self._mine(), (FAILED, MAX_REVIVALS)).fetchone()[0]
        return {"revived": revived, "exhausted": exhausted}
