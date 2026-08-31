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

    def reconcile(self, rows: list) -> dict:
        """Mark done every pass whose evidence this database already holds.

        This is the step that makes "nothing is ever reprocessed" true rather
        than hopeful. A restore replays shards into the evidence tables and does
        not touch this one, so after it the two disagree: the claims are back
        and the work table still reads `queued`. Left alone, the sweep re-earns
        with the GPU what the channel already had.

        `rows` is `(video_key, component, observer_id, claims, vectors,
        seconds)` — already decided by the caller, because deciding *what
        counts as evidence* needs the registry and this module deliberately
        does not import it. What belongs here is the write, and its rules:

        Rows already `done` are left exactly as they are — their `claims` and
        `seconds` are the record of the run that produced them and are not
        overwritten by a count taken after the fact. `running` rows are left
        because another worker holds that lease and stealing it would let two
        workers believe they own the same pass. `skipped` rows are left because
        a decline is terminal and was made for a stated reason.

        Returns per-component counts: "reconciled 4 118 rows" is a number an
        operator has to trust, and a breakdown is what makes it checkable
        against the coverage matrix on the same screen.
        """
        rows = [r for r in (rows or []) if r and r[0] and r[1]]
        if not rows:
            return {"rows": 0, "matched": 0, "per_component": {}, "videos": 0}

        now = time.time()
        per, vids = {}, set()
        payload = []
        for video_key, component, oid, claims, vectors, seconds in rows:
            payload.append((now, float(seconds or 0.0), int(claims or 0),
                            int(vectors or 0), oid or "", video_key, component))
            per[component] = per.get(component, 0) + 1
            vids.add(video_key)

        cur = self.conn.executemany(
            "UPDATE coverage SET state=?, done_at=?, seconds=?, "
            "claims=?, vectors=?, observer_id=?, lease_until=0, token=NULL, "
            "next_try_at=0, last_error=NULL, checkpoint=NULL "
            "WHERE video_key=? AND component=? AND state IN (?,?)",
            [(DONE,) + p + (QUEUED, FAILED) for p in payload])
        self.conn.commit()
        changed = max(cur.rowcount, 0)
        # `executemany` reports one total across every statement, so the
        # per-component tally is an upper bound whenever some rows were already
        # done. Scaling it down would invent precision; reporting both numbers
        # is the honest shape.
        return {"rows": changed, "matched": len(payload),
                "per_component": per, "videos": len(vids)}

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

    def defer(self, video_key: str, component: str, why: str,
              retry_after: float = 300.0) -> None:
        """Put the row back without spending an attempt.

        The difference from `fail()` is the whole point: `MAX_ATTEMPTS` is never
        approached, because nothing failed. A NIM account with forty requests a
        minute will defer thousands of times across an archive, and every one of
        those is the system working correctly — the pass is simply waiting for
        its turn.

        Which is why the attempt is handed *back* here rather than merely not
        added. Every caller is inside `_run_pass`, and `claim_for` already
        incremented `attempts` before the pass was reached — so a row deferred a
        thousand times would arrive at `attempts=1000` and the docstring above
        would be false. The visible cost was not that it becomes unclaimable
        (`queued` rows are claimable at any `attempts`) but that the first real
        failure afterwards reads `attempts >= MAX_ATTEMPTS` and skips the 10- and
        45-minute backoff straight to the four-hour revival ladder, and that
        `claim`'s `ORDER BY attempts` sends a much-deferred video to the back of
        every queue it is in. `MAX(attempts - 1, 0)` returns the row to the count
        it had before this claim, which is the number of times the pass has
        actually run.

        `last_error` still carries the reason so the tab can say what is being
        waited on, and `deferred()` already reports rows with a future
        `next_try_at`, so this needs no new state value and no migration.
        """
        self.conn.execute(
            "UPDATE coverage SET state=?, last_error=?, lease_until=0, "
            "next_try_at=?, worker=NULL, token=NULL, "
            "attempts=MAX(attempts - 1, 0) "
            "WHERE video_key=? AND component=?",
            (QUEUED, str(why)[:800], time.time() + max(5.0, float(retry_after)),
             video_key, component))
        self.conn.commit()

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

    def stage_report(self, stage: str, components: list) -> dict:
        """Everything one registry stage did, and everything it broke on.

        A stage is the unit the operator actually reasons about — "perception
        finished, language is where it went wrong" — and it is the unit the
        channel bundle is cut on, so this report is what travels with the
        database. It is built entirely from columns that already exist; a stage
        is a grouping of components, not a new kind of state, and inventing a
        `stage` column would have created a second place for the truth to live.

        `errors` carries the individual rows rather than the grouped summary
        `failures()` returns, because the bundle is read later by someone who no
        longer has the session in front of them: the video key is the only thing
        that makes a failure reproducible, and a count of eleven is not.

        `observers` is per component and is the audit trail. Two revisions of
        the same pass produce two observer ids, and a bundle that lists both is
        the record of which reading each claim came from.
        """
        comps = [c for c in (components or []) if c]
        out = {
            "stage": stage,
            "components": comps,
            "videos": 0,
            "counts": {s: 0 for s in (QUEUED, RUNNING, DONE, FAILED, SKIPPED)},
            "claims": 0, "vectors": 0, "seconds": 0.0,
            "per_component": [], "errors": [], "observers": {},
            "started": None, "ended": None, "complete": False,
        }
        out["counts"]["total"] = 0
        if not comps:
            return out

        marks = ",".join("?" * len(comps))
        mine = self._mine()          # " AND video_key IN (...)" — goes in WHERE
        per: dict = {}
        for r in self.conn.execute(
                f"SELECT component, state, COUNT(*) n, SUM(claims) c, "
                f"SUM(vectors) v, SUM(seconds) s, MIN(started_at) t0, "
                f"MAX(done_at) t1 FROM coverage WHERE component IN ({marks})"
                + mine + " GROUP BY component, state", comps):
            e = per.setdefault(r["component"], {
                "component": r["component"], "total": 0,
                QUEUED: 0, RUNNING: 0, DONE: 0, FAILED: 0, SKIPPED: 0,
                "claims": 0, "vectors": 0, "seconds": 0.0})
            e[r["state"]] = r["n"]
            e["total"] += r["n"]
            e["claims"] += r["c"] or 0
            e["vectors"] += r["v"] or 0
            e["seconds"] += round(r["s"] or 0.0, 2)
            out["counts"][r["state"]] = out["counts"].get(r["state"], 0) + r["n"]
            out["counts"]["total"] += r["n"]
            out["claims"] += r["c"] or 0
            out["vectors"] += r["v"] or 0
            out["seconds"] += r["s"] or 0.0
            if r["t0"] and (out["started"] is None or r["t0"] < out["started"]):
                out["started"] = r["t0"]
            if r["t1"] and (out["ended"] is None or r["t1"] > out["ended"]):
                out["ended"] = r["t1"]

        out["seconds"] = round(out["seconds"], 2)
        out["per_component"] = [per[c] for c in comps if c in per]

        row = self.conn.execute(
            f"SELECT COUNT(DISTINCT video_key) n FROM coverage "
            f"WHERE component IN ({marks})" + mine, comps).fetchone()
        out["videos"] = row["n"] or 0

        for r in self.conn.execute(
                f"SELECT component, observer_id, COUNT(*) n FROM coverage "
                f"WHERE component IN ({marks}) AND observer_id IS NOT NULL"
                + mine + " GROUP BY component, observer_id", comps):
            out["observers"].setdefault(r["component"], []).append(
                {"observer_id": r["observer_id"], "videos": r["n"]})

        # Every row that carries an error and is not done, not just the ones
        # that have given up. `fail()` returns a row to `queued` with a backoff
        # until its attempts are spent, so filtering on state='failed' would
        # hide the entire first two attempts of every failure — which is most of
        # them, and the ones a bundle is read to diagnose. The state travels
        # with the row so "still retrying" and "out of attempts" stay distinct.
        #
        # Deliberately unbounded in row count but bounded in text: a list
        # truncated to twenty would hide exactly the long tail it exists to
        # preserve, while a 4 KB traceback repeated four hundred times would
        # swamp the file it travels in.
        for r in self.conn.execute(
                f"SELECT video_key, component, state, attempts, revivals, "
                f"last_error, done_at, next_try_at FROM coverage "
                f"WHERE component IN ({marks}) AND state<>? "
                f"AND last_error IS NOT NULL AND last_error<>''" + mine
                + " ORDER BY component, video_key",
                comps + [DONE]):
            out["errors"].append({
                "video_key": r["video_key"], "component": r["component"],
                "state": r["state"],
                "retrying": r["state"] not in TERMINAL
                            and (r["attempts"] or 0) < MAX_ATTEMPTS,
                "attempts": r["attempts"] or 0, "revivals": r["revivals"] or 0,
                "error": (r["last_error"] or "")[:400],
                "at": r["done_at"], "next_try_at": r["next_try_at"] or 0})

        left = (out["counts"][QUEUED] + out["counts"][RUNNING]
                + out["counts"][FAILED])
        out["complete"] = bool(out["counts"]["total"]) and left == 0
        out["remaining"] = left
        return out

    def stage_fingerprint(self, components: list) -> str:
        """A content hash of what a stage's rows actually say.

        This exists because the counter it replaces was not an identity. The old
        fingerprint was `total:done:skipped:failed:claims:vectors`, and
        `revive_failed()` runs every rotation and flips one row from `failed` to
        `queued` — so `5 failed` became `4 failed`, the whole database was
        vacuumed, gzipped, split and re-uploaded, and the retry that produced the
        change produced no evidence. That is `language-0013` and `-0014`: two
        135 MB byte-siblings, one revived row apart.

        Two decisions make this a real identity rather than a longer counter.

        **States are coarsened into classes.** `queued`, `running` and `failed`
        all hash as `open`, because they are the same statement — *this row has
        not settled* — expressed at different points in a retry cycle that the
        rotation drives on its own. `done` and `skipped` stay distinct from each
        other and from `open`, because those are settlements and a bundle exists
        to record them. A hash over raw states would move on every revival, which
        is precisely the bug.

        **Evidence is counted per row, not globally.** The obvious strengthener
        is to fold in `max_claim_id`/`max_vector_id`, but those advance whenever
        *any* pass anywhere writes anything, so a stage that did nothing this
        cohort would still re-ship its database because another stage was busy.
        Per-row `claims`/`vectors` move only when this stage's own rows gain
        evidence, which is the question being asked. `observer_id` is in for the
        same reason: a re-run under a new pass revision produces the same counts
        from a different reading, and that is a genuinely different database.

        So: a revived retry that produced nothing does not move this hash; a row
        that gained a claim, settled, or was re-observed does.
        """
        import hashlib          # noqa: PLC0415

        comps = sorted({c for c in (components or []) if c})
        h = hashlib.sha256()
        h.update(("|".join(comps) + "\n").encode("utf-8"))
        if not comps:
            return h.hexdigest()[:24]

        marks = ",".join("?" * len(comps))
        mine = self._mine()
        n = 0
        for r in self.conn.execute(
                f"SELECT component, video_key, state, observer_id, claims, "
                f"vectors FROM coverage WHERE component IN ({marks})" + mine
                + " ORDER BY component, video_key", comps):
            # `open` collapses queued/running/failed; see the docstring.
            state = r["state"] if r["state"] in (DONE, SKIPPED) else "open"
            h.update(f"{r['component']}\x1f{r['video_key']}\x1f{state}\x1f"
                     f"{r['observer_id'] or ''}\x1f{r['claims'] or 0}\x1f"
                     f"{r['vectors'] or 0}\n".encode("utf-8"))
            n += 1
        h.update(f"rows={n}\n".encode("utf-8"))
        return h.hexdigest()[:24]

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

    def recent_errors(self, limit: int = 40) -> list:
        """Individual error rows, newest first — what the tab's panel lists.

        `failures()` groups by component and message, which is the right shape
        for "what is wrong with this run" and the wrong shape for "what just
        went wrong". Both exist because both questions get asked, and answering
        the second from the first means an operator watching a sweep sees a
        count go from 4 to 5 with no way to learn which video it was.

        Includes rows that `fail()` returned to `queued` with a backoff, for the
        same reason `stage_report` does: those are the first two attempts of
        every failure, so a panel filtered to `state='failed'` would show
        nothing at all until a video had already failed three times.
        """
        rows = []
        for r in self.conn.execute(
                "SELECT video_key, component, state, attempts, revivals, "
                "last_error, done_at, next_try_at FROM coverage "
                "WHERE state<>? AND last_error IS NOT NULL AND last_error<>''"
                + self._mine()
                + " ORDER BY COALESCE(done_at,0) DESC LIMIT ?",
                (DONE, max(1, min(int(limit), 300)))):
            rows.append({
                "video_key": r["video_key"], "component": r["component"],
                "state": r["state"],
                "retrying": r["state"] not in TERMINAL
                            and (r["attempts"] or 0) < MAX_ATTEMPTS,
                "attempts": r["attempts"] or 0,
                "revivals": r["revivals"] or 0,
                "error": (r["last_error"] or "")[:400],
                "at": r["done_at"] or 0,
                "next_try_at": r["next_try_at"] or 0})
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
    def unsettled(self, video_key: str, components) -> dict:
        """Of these components, the ones whose answer for this video can still
        change. Returns {component: when it may be claimed again}.

        The dependency gate needs this to tell two situations apart that the
        `state` column alone conflates. A need that is `skipped` has declined —
        it will decline again next session, so a consumer that cannot run
        without it should be retired with it. A need that is `failed` with
        revivals left has not declined anything: it is on the retry ladder and
        `revive_failed` will hand it back in four hours. Treating those the same
        is what made one OOM on `shots` permanently retire `keyframes`, `poster`
        and everything downstream for that video — the dependency came back and
        succeeded, and its consumers were already terminal.

        `next_try_at` comes back with it so the caller can wait exactly as long
        as the dependency is going to wait, instead of guessing an interval and
        re-asking the same question forty-eight times in between.
        """
        want = [c for c in (components or []) if c]
        if not want:
            return {}
        qs = ",".join("?" * len(want))
        out = {}
        for r in self.conn.execute(
                f"SELECT component, state, revivals, next_try_at FROM coverage "
                f"WHERE video_key=? AND component IN ({qs})",
                [video_key] + want):
            if r["state"] in (QUEUED, RUNNING):
                out[r["component"]] = float(r["next_try_at"] or 0)
            elif (r["state"] == FAILED
                    and (r["revivals"] or 0) < MAX_REVIVALS):
                # `fail()` already set `next_try_at` to now + REVIVE_AFTER for a
                # row it retired, so this is the revival moment itself, not an
                # offset from it.
                out[r["component"]] = float(r["next_try_at"] or 0)
        return out

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

    # Skip reasons that were never about the video. Used once, on an existing
    # database, to undo declines that a later build made wrong; new code does
    # not create rows this has to match, because an environment fault raises
    # `PassUnavailable` and lands in `failed` where the retry ladder can see it.
    # That split matters: matching on a message is archaeology, and archaeology
    # is the only tool available for rows already written — `state` and
    # `last_error` are the entire record a skip leaves behind.
    #
    # Every pattern below is a string this codebase itself produces, quoted from
    # the raise site, so the list is checkable rather than hopeful:
    UNAVAILABLE_LIKE = (
        "% is not installed%",          # faster-whisper, pyannote.audio, librosa,
                                        #   ultralytics, insightface, paddleocr
        "%are not installed%",          # transformers/torch, torch/Pillow
        "%could be initialised%",       # "no OCR engine could be initialised" —
                                        #   the negation is carried by "no …
                                        #   engine", so this pattern cannot hold a
                                        #   "not". Quoted from the raise site, and
                                        #   `_t_unavailable.py` fails if either
                                        #   side drifts from the other.
        "%could not be loaded%",        # a language model, Florence-2
        "%could not start%",            # insightface could not start
        "%is not on PATH%",             # ffmpeg
        "%no Hugging Face token%",      # pyannote's gated weights
        "%no NIM key%",                 # the cloud narrator
        "%NIM is not reachable%",
        "%NIM unusable%",
        "%needs % GPUs; this machine has %",
    )

    def reclaim_unavailable(self) -> dict:
        """Hand back rows that were declined for something the video never said.

        A one-shot repair for a database written by an earlier build, not a
        recurring rule. The measured damage it exists to undo: before the
        EasyOCR fallback landed on 17 August, the OCR pass raised
        `SkipPass("paddleocr is not installed")` — `paddlepaddle` is deliberately
        absent from requirements.txt — and every one of those became a `skipped`
        row, which `claim`, `candidates`, `revive_failed` and `reconcile` all
        refuse to reconsider. The fallback works; it could only ever reach the
        two rows that had not been retired yet, and those two are exactly the two
        videos in the archive with OCR evidence today. There is no manual escape
        either: the Process tab's Requeue button posts `state:'failed'`
        (process_ui.html:1451), and the Kaggle cell clones fresh with nobody
        watching.

        Only `skipped` rows are touched, and only those whose `last_error`
        matches a string this codebase produces for an environment fault, so a
        genuine decline — "no speech detected", "container reports zero
        duration" — is left exactly where it is. `attempts` is cleared so the
        retry is a real retry; `revivals` is deliberately *not*, so a row that
        has already exhausted its ladder does not get a second full one.
        `last_error` is kept for the same reason `revive_failed` keeps it: if
        this row fails again, the operator needs to see what it said the first
        time.

        Returns {"requeued": n, "components": {cid: n}}. Reporting zero is a
        result, not a silence: it says these rows were never skipped for a
        missing backend, which is the one alternative a log this old cannot
        otherwise rule out.
        """
        where = " OR ".join("last_error LIKE ?" for _ in self.UNAVAILABLE_LIKE)
        args = list(self.UNAVAILABLE_LIKE)
        by: dict = {}
        for r in self.conn.execute(
                f"SELECT component, COUNT(*) n FROM coverage "
                f"WHERE state=? AND ({where}) GROUP BY component",
                [SKIPPED] + args):
            by[r["component"]] = r["n"]
        cur = self.conn.execute(
            f"UPDATE coverage SET state=?, attempts=0, next_try_at=0, "
            f"worker=NULL, token=NULL, lease_until=0 "
            f"WHERE state=? AND ({where})",
            [QUEUED, SKIPPED] + args)
        self.conn.commit()
        return {"requeued": max(cur.rowcount, 0), "components": by}

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
