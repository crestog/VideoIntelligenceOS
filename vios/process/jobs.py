"""The job plane's broker, and the same broker without Redis.

Why this file exists at all
───────────────────────────
`queue_manager.py` is already a good broker — atomic `BRPOPLPUSH` claims, a
PROCESSING list that survives a worker crash, three retries then a dead-letter
queue, boot-time orphan recovery, and per-queue metrics. The v2 processing plane
wants exactly that. What it cannot want is a *hard dependency* on it: `routes.py`
notes that a laptop mounts these routes with no Redis at all, and the processing
tab has to come up there too.

So the four calls the plane makes go through this shim, which picks its backend
once at `start()` via `wait_for_redis` and then never asks again. With Redis it
is a thin pass-through. Without it, an in-process `queue.Queue` with identical
signatures — single-machine, no crash recovery across a restart, and that is a
truthful degradation rather than a hidden one, because `backend()` says which is
in play and the tab prints it.

The rule the whole plane rests on
─────────────────────────────────
**Redis proposes, sqlite decides.** A job in these lanes is a *hint* about what
to work on next, never a grant of ownership. Ownership is `coverage.claim_for`'s
answer — one atomic UPDATE stamping a unique token, in the database, which cannot
lose or duplicate. Redis can do both.

That inversion is what makes the whole thing safe to build on an ephemeral
broker:

  * A worker that claims a job and gets `[]` back from `claim_for` acks it and
    moves on. Someone else has that video. It is not an error and is not logged
    as one.
  * The two recovery paths cannot double-write. `recover_processing_jobs`
    re-queues a *hint*; `LEASE_SECONDS` expiry frees the *row*. Whichever fires
    first, the loser's `claim_for` returns nothing.
  * No state that matters lives only here. A cold Redis costs one re-plan, not
    one row of evidence.

Nothing in this module touches the database, and nothing in it may raise on the
happy path — a broker that throws while reporting a queue depth would take down
the sweep that was only asking a question.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid

# The lane names live in `queue_manager` even for the memory backend, so that
# one list of names is registered with boot's orphan recovery and the metrics
# panel. Importing them from there is what keeps the two in step.
try:
    from queue_manager import (QUEUE_V2_CLOUD, QUEUE_V2_CPU, QUEUE_V2_GPU,
                               QUEUE_V2_PREP, V2_QUEUES)
except Exception:                                       # noqa: BLE001
    # No `queue_manager` at all (a trimmed checkout, or an import that dragged
    # in `config` and failed). The names are still needed to talk about lanes.
    QUEUE_V2_PREP = "QUEUE_V2_PREP"
    QUEUE_V2_GPU = "QUEUE_V2_GPU"
    QUEUE_V2_CPU = "QUEUE_V2_CPU"
    QUEUE_V2_CLOUD = "QUEUE_V2_CLOUD"
    V2_QUEUES = [QUEUE_V2_PREP, QUEUE_V2_GPU, QUEUE_V2_CPU, QUEUE_V2_CLOUD]

__all__ = ["QUEUE_V2_PREP", "QUEUE_V2_GPU", "QUEUE_V2_CPU", "QUEUE_V2_CLOUD",
           "V2_QUEUES", "Broker", "MAX_RETRIES"]

MAX_RETRIES = 3

# How long a worker blocks on an empty lane before looking up. Short enough that
# a stop request is felt promptly, long enough that four idle workers are not
# spinning on Redis. The stop check happens between claims, so this is also the
# worst-case delay between "stop" and a worker noticing.
CLAIM_TIMEOUT = 2.0


# ══════════════════════════════════════════════════════════════════════════
# The in-process backend
# ══════════════════════════════════════════════════════════════════════════

class _MemoryBroker:
    """`queue.Queue` per lane, with the parts of the Redis contract that matter.

    Deliberately not a full reimplementation. What it keeps is the behaviour the
    plane's correctness depends on — FIFO order, an in-flight list so a crash
    does not silently drop a hint, retry-then-dead-letter, and counters — and
    what it drops is durability across a process restart, which an in-process
    queue cannot have and should not pretend to.
    """

    def __init__(self) -> None:
        self._lanes: dict = {}
        self._processing: dict = {}
        self._dlq: dict = {}
        self._counts: dict = {}
        self._lock = threading.Lock()

    def _lane(self, name: str) -> queue.Queue:
        with self._lock:
            if name not in self._lanes:
                self._lanes[name] = queue.Queue()
                self._processing[name] = []
                self._dlq[name] = []
            return self._lanes[name]

    def _bump(self, name: str, field: str, n: int = 1) -> None:
        with self._lock:
            self._counts[f"{name}:{field}"] = \
                self._counts.get(f"{name}:{field}", 0) + n

    # ── the four calls ───────────────────────────────────────────────────
    def push(self, name: str, payload: dict) -> str:
        job_id = f"job:{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        job = {"id": job_id, "payload": payload, "created_at": time.time(),
               "retries": 0}
        self._lane(name).put(json.dumps(job))
        self._bump(name, "pushed")
        return job_id

    def claim(self, name: str, timeout: float = CLAIM_TIMEOUT):
        lane = self._lane(name)
        try:
            raw = lane.get(timeout=max(0.05, timeout))
        except queue.Empty:
            return None, None
        with self._lock:
            self._processing[name].append(raw)
        job = json.loads(raw)
        job["claimed_at"] = time.time()
        self._bump(name, "claimed")
        return job, raw

    def ack(self, name: str, job: dict, raw: str) -> None:
        with self._lock:
            try:
                self._processing[name].remove(raw)
            except (KeyError, ValueError):
                pass
        self._bump(name, "completed")

    def fail(self, name: str, job: dict, raw: str, error: str) -> str:
        with self._lock:
            try:
                self._processing[name].remove(raw)
            except (KeyError, ValueError):
                pass
        retries = job.get("retries", 0)
        job["last_error"] = str(error)[:200]
        if retries < MAX_RETRIES:
            job["retries"] = retries + 1
            job["status"] = "RETRY"
            self._lane(name).put(json.dumps(job))
            self._bump(name, "retries")
            return "RETRIED"
        job["status"] = "DEAD"
        job["died_at"] = time.time()
        with self._lock:
            self._dlq[name].append(json.dumps(job))
        self._bump(name, "dead")
        return "DEAD"

    def recover(self, name: str) -> int:
        """Re-queue in-flight hints. Only useful after a worker thread died."""
        with self._lock:
            held = list(self._processing.get(name) or [])
            self._processing[name] = []
        for raw in held:
            self._lane(name).put(raw)
        if held:
            self._bump(name, "recovered", len(held))
        return len(held)

    def depth(self, name: str) -> int:
        return self._lane(name).qsize()

    def drain(self, name: str) -> int:
        """Empty a lane and forget its in-flight list. Returns what was dropped.

        The planner calls this when it abandons a cohort — a stop, or a wave
        preemption. Hints for work nobody is going to admit are noise that the
        next cohort's workers would claim and immediately ack.
        """
        lane = self._lane(name)
        n = 0
        while True:
            try:
                lane.get_nowait()
                n += 1
            except queue.Empty:
                break
        with self._lock:
            n += len(self._processing.get(name) or [])
            self._processing[name] = []
        return n

    def metrics(self, names=None) -> dict:
        out = {}
        for name in (names or list(self._lanes)):
            with self._lock:
                proc = len(self._processing.get(name) or [])
                dlq = len(self._dlq.get(name) or [])
                c = dict(self._counts)
            out[name] = {
                "pending_total": self.depth(name),
                "processing": proc,
                "dead_letter": dlq,
                "total_pushed": c.get(f"{name}:pushed", 0),
                "total_claimed": c.get(f"{name}:claimed", 0),
                "total_completed": c.get(f"{name}:completed", 0),
                "total_retries": c.get(f"{name}:retries", 0),
                "total_dead": c.get(f"{name}:dead", 0),
                "total_recovered": c.get(f"{name}:recovered", 0),
            }
        return out


# ══════════════════════════════════════════════════════════════════════════
# The Redis backend
# ══════════════════════════════════════════════════════════════════════════

class _RedisBroker:
    """A pass-through to `queue_manager`, which already does all of this."""

    def __init__(self, qm) -> None:
        self._qm = qm

    def push(self, name: str, payload: dict) -> str:
        return self._qm.push_job(name, payload)

    def claim(self, name: str, timeout: float = CLAIM_TIMEOUT):
        # `int()` because redis-py's blocking pops take whole seconds, and a
        # float 2.0 becomes 2 while a float 0.5 would become 0 — which means
        # *block forever*, and a worker that never looks up at its stop flag.
        return self._qm.claim_job(name, timeout=max(1, int(timeout)))

    def ack(self, name: str, job: dict, raw: str) -> None:
        self._qm.ack_job(name, job, raw)

    def fail(self, name: str, job: dict, raw: str, error: str) -> str:
        return self._qm.fail_job(name, job, raw, error)

    def recover(self, name: str) -> int:
        return self._qm.recover_processing_jobs(name)

    def depth(self, name: str) -> int:
        return self._qm.get_queue_depth(name)

    def drain(self, name: str) -> int:
        r = self._qm.get_redis()
        n = int(r.llen(name) or 0) + int(
            r.llen(self._qm._processing_key(name)) or 0)
        r.delete(name, self._qm._processing_key(name))
        return n

    def metrics(self, names=None) -> dict:
        out = {}
        for name in (names or V2_QUEUES):
            try:
                out[name] = self._qm.get_queue_metrics(name).get(name, {})
            except Exception:                            # noqa: BLE001
                out[name] = {}
        return out


# ══════════════════════════════════════════════════════════════════════════
# What the engine holds
# ══════════════════════════════════════════════════════════════════════════

class Broker:
    """One of the two backends, chosen once, plus a `dead()` roll-up.

    Every method swallows backend faults and returns a neutral answer. A Redis
    that goes away mid-session must not crash the sweep: `claim` returning
    nothing looks exactly like an empty lane, the workers idle, and the planner's
    own `candidates` loop is what actually finds the work. Degraded throughput
    beats a dead engine.
    """

    def __init__(self, log=None) -> None:
        self._log = log or (lambda *_a, **_k: None)
        self._impl = None
        self.kind = "none"

    # ── choosing a backend, once ─────────────────────────────────────────
    def start(self, wait_seconds: float = 5.0) -> str:
        if self._impl is not None:
            return self.kind
        try:
            import queue_manager as qm                   # noqa: PLC0415
            if qm.wait_for_redis(timeout=wait_seconds, label="process/jobs"):
                self._impl = _RedisBroker(qm)
                self.kind = "redis"
                self._log("Job plane on Redis — lanes survive a worker crash "
                          "and a mid-session restart")
                return self.kind
            reason = "Redis did not answer"
        except Exception as exc:                         # noqa: BLE001
            reason = f"{type(exc).__name__}: {str(exc)[:120]}"

        self._impl = _MemoryBroker()
        self.kind = "memory"
        self._log(f"Job plane in-process ({reason}) — workers still run in "
                  f"parallel; hints do not survive a restart, and coverage "
                  f"leases are what recover the work if one does", "warn")
        return self.kind

    def ready(self) -> bool:
        return self._impl is not None

    # ── the calls, each fault-tolerant ───────────────────────────────────
    def push(self, name: str, payload: dict) -> str:
        try:
            return self._impl.push(name, payload)
        except Exception as exc:                         # noqa: BLE001
            self._log(f"queue push to {name} failed: "
                      f"{type(exc).__name__}: {str(exc)[:120]}", "warn")
            return ""

    def claim(self, name: str, timeout: float = CLAIM_TIMEOUT):
        try:
            return self._impl.claim(name, timeout=timeout)
        except Exception as exc:                         # noqa: BLE001
            self._log(f"queue claim on {name} failed: "
                      f"{type(exc).__name__}: {str(exc)[:120]}", "warn")
            time.sleep(1.0)
            return None, None

    def ack(self, name: str, job: dict, raw: str) -> None:
        try:
            self._impl.ack(name, job, raw)
        except Exception:                                # noqa: BLE001
            pass

    def fail(self, name: str, job: dict, raw: str, error: str) -> str:
        try:
            return self._impl.fail(name, job, raw, error)
        except Exception:                                # noqa: BLE001
            return "RETRIED"

    def recover(self, name: str) -> int:
        try:
            return int(self._impl.recover(name) or 0)
        except Exception:                                # noqa: BLE001
            return 0

    def depth(self, name: str) -> int:
        try:
            return int(self._impl.depth(name) or 0)
        except Exception:                                # noqa: BLE001
            return 0

    def drain(self, name: str) -> int:
        try:
            return int(self._impl.drain(name) or 0)
        except Exception:                                # noqa: BLE001
            return 0

    def metrics(self, names=None) -> dict:
        try:
            return self._impl.metrics(names or V2_QUEUES)
        except Exception:                                # noqa: BLE001
            return {}

    def dead(self) -> int:
        """Total dead-lettered across every v2 lane.

        Surfaced on its own because it is the one queue number that is never
        normal. Anything here names a real bug — a job shape a worker cannot
        parse, or a video that crashes a worker three times running.
        """
        return sum(int((m or {}).get("dead_letter", 0) or 0)
                   for m in self.metrics().values())
