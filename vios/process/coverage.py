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

SCHEMA = """
CREATE TABLE IF NOT EXISTS coverage (
    video_key   TEXT NOT NULL,
    component   TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'queued',
    attempts    INTEGER DEFAULT 0,
    worker      TEXT,
    token       TEXT,           -- identifies one claim() call, exactly
    lease_until REAL DEFAULT 0,
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
CREATE INDEX IF NOT EXISTS ix_cov_claim ON coverage(component, state, lease_until);
CREATE INDEX IF NOT EXISTS ix_cov_state ON coverage(state, component);
CREATE INDEX IF NOT EXISTS ix_cov_token ON coverage(token);
"""


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
        self.conn.commit()

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
        """
        now = time.time()
        token = uuid.uuid4().hex
        sql = (
            "UPDATE coverage SET state=?, worker=?, token=?, lease_until=?, "
            "started_at=?, attempts=attempts+1 "
            "WHERE rowid IN (SELECT rowid FROM coverage "
            "  WHERE component=? AND ("
            "     state=? OR (state=? AND lease_until < ?) "
            "     OR (state=? AND attempts < ?)) "
            + self._mine() +
            "  ORDER BY attempts, video_key LIMIT ?)")
        self.conn.execute(sql, (
            RUNNING, self.worker, token, now + LEASE_SECONDS, now,
            component, QUEUED, RUNNING, now, FAILED, MAX_ATTEMPTS, n))
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
        """
        if not components:
            return []
        now = time.time()
        marks = ",".join("?" * len(components))
        sql = (
            "SELECT video_key, MIN(attempts) AS a FROM coverage "
            f"WHERE component IN ({marks}) AND ("
            "   state=? OR (state=? AND lease_until < ?) "
            "   OR (state=? AND attempts < ?))"
            + self._mine() +
            " GROUP BY video_key ORDER BY a, video_key LIMIT ?")
        args = list(components) + [QUEUED, RUNNING, now, FAILED,
                                   MAX_ATTEMPTS, int(limit)]
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
            "     state=? OR (state=? AND lease_until < ?) "
            "     OR (state=? AND attempts < ?))"
            + self._mine() + ")")
        args = ([RUNNING, self.worker, token, now + LEASE_SECONDS, now,
                 video_key] + list(components) +
                [QUEUED, RUNNING, now, FAILED, MAX_ATTEMPTS])
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

        A row that has burned its attempts stays `failed` and stops being
        claimed — but it is never deleted, because "this component cannot handle
        this video" is itself a finding, and the failures panel groups them so a
        pattern is visible. Ten OOMs on the ten longest reels is a configuration
        problem, not ten unrelated accidents.
        """
        row = self.conn.execute(
            "SELECT attempts FROM coverage WHERE video_key=? AND component=?",
            (video_key, component)).fetchone()
        attempts = row["attempts"] if row else MAX_ATTEMPTS
        state = FAILED if attempts >= MAX_ATTEMPTS else QUEUED
        self.conn.execute(
            "UPDATE coverage SET state=?, last_error=?, lease_until=0, "
            "worker=NULL, token=NULL WHERE video_key=? AND component=?",
            (state, str(error)[:800], video_key, component))
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
            "state=? OR (state=? AND lease_until < ?) OR "
            "(state=? AND attempts < ?))" + self._mine(),
            (component, QUEUED, RUNNING, now, FAILED, MAX_ATTEMPTS)
        ).fetchone()[0]

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
        return [dict(r) for r in self.conn.execute(
            "SELECT component, last_error, COUNT(*) n, "
            "GROUP_CONCAT(video_key) keys FROM coverage WHERE state=? "
            "GROUP BY component, SUBSTR(last_error,1,80) "
            "ORDER BY n DESC LIMIT ?", (FAILED, limit))]

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
        sql = ("UPDATE coverage SET state=?, attempts=0, last_error=NULL, "
               "worker=NULL, token=NULL, lease_until=0 WHERE state=?")
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
            "UPDATE coverage SET state=?, attempts=0, worker=NULL, token=NULL, "
            "lease_until=0, done_at=NULL, seconds=NULL, last_error=NULL, "
            "checkpoint=NULL WHERE component=?", (QUEUED, component))
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
