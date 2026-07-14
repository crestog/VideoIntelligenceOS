"""
VIOS Queue Manager v3 — Enterprise Reliable Job Queue System

v3 changes:
  - ATOMIC claims: BRPOPLPUSH moves queue→PROCESSING in one Redis op.
    (v2 did blpop + lpush as two ops — a worker crash between them lost the job.)
  - FIFO preserved by LPUSH-producer / RPOP-consumer orientation.
  - QUEUE_ANALYZE (GPU analysis stage) registered in metrics.

Job Lifecycle:
  PUSH → [PRIORITY|DEFAULT] → CLAIM (atomic) → PROCESSING → ACK (completed)
                                                          → FAIL → RETRY (re-queued)
                                                                 → DEAD (DLQ)

Key Schema (Redis):
  {QUEUE}_PRIORITY     — Priority lane (FIFO list)
  {QUEUE}_DEFAULT      — Default lane (FIFO list)
  {QUEUE}_PROCESSING   — In-flight jobs (crash recovery list)
  {QUEUE}_DLQ          — Dead letter queue
  VIOS_METRICS         — Hash of all counters and timestamps
"""

import redis
import json
import time
import uuid

try:
    from config import REDIS_HOST, REDIS_PORT
except Exception:  # standalone usage fallback
    REDIS_HOST, REDIS_PORT = 'localhost', 6379


# ═══════════════════════════════════════════════════════════
# CONNECTION POOL
# ═══════════════════════════════════════════════════════════
redis_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_timeout=30,          # Must be > any blocking-pop timeout (max 5s)
    socket_connect_timeout=5,
    retry_on_timeout=True,
    health_check_interval=15,
)

def get_redis():
    return redis.Redis(connection_pool=redis_pool)


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
PRIORITY_QUEUES = {"QUEUE_VISION"}    # Queues with dual-lane priority routing
ALL_QUEUES = ["QUEUE_VISION", "QUEUE_ANALYZE", "QUEUE_MODELS", "QUEUE_ORACLE", "QUEUE_VISION_EMBED"]
MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════
# KEY SCHEMA HELPERS
# ═══════════════════════════════════════════════════════════
def _priority_key(q):    return f"{q}_PRIORITY"
def _default_key(q):     return f"{q}_DEFAULT"
def _processing_key(q):  return f"{q}_PROCESSING"
def _dlq_key(q):         return f"{q}_DLQ"
def _metrics_key():      return "VIOS_METRICS"
def _pause_key(q):       return f"VIOS_PAUSED:{q}"
def _heartbeat_key(q, job_id): return f"VIOS_HB:{q}:{job_id}"
def _oplock_key(name):   return f"VIOS_OPLOCK:{name}"

HEARTBEAT_TTL_SEC   = 120     # a live worker refreshes well within this window
STALE_JOB_GRACE_SEC = 1800    # jobs with NO heartbeat are considered stale after 30 min in-flight

def _lanes(q):
    """(priority_lane, default_lane) — non-priority queues use one lane."""
    if q in PRIORITY_QUEUES:
        return _priority_key(q), _default_key(q)
    return None, q


# ═══════════════════════════════════════════════════════════
# PUSH — Route a job into the queue system
# FIFO orientation: producers LPUSH (head), consumers pop the TAIL.
# ═══════════════════════════════════════════════════════════
def push_job(queue_name, payload, is_priority=False):
    """Push a job with a unique ID and metadata envelope. Returns the job ID."""
    r = get_redis()
    job_id = f"job:{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

    job = {
        "id": job_id,
        "payload": payload,
        "created_at": time.time(),
        "retries": 0,
    }
    job_data = json.dumps(job)

    prio_lane, default_lane = _lanes(queue_name)
    lane = prio_lane if (is_priority and prio_lane) else default_lane

    pipe = r.pipeline()
    pipe.lpush(lane, job_data)
    pipe.hincrby(_metrics_key(), f"{queue_name}:pushed", 1)
    pipe.execute()

    return job_id


# ═══════════════════════════════════════════════════════════
# CLAIM — Atomic pop → PROCESSING (BRPOPLPUSH)
# ═══════════════════════════════════════════════════════════
def claim_job(queue_name, timeout=2):
    """
    Claim a job with reliable delivery. The move to PROCESSING happens in a
    single atomic Redis command, so a crash at any point leaves the job either
    still queued or safely in PROCESSING (recovered on reboot) — never lost.

    Priority queues drain the PRIORITY lane first (non-blocking), then block
    on the DEFAULT lane for `timeout` seconds.

    Respects the queue's pause flag: while paused, no job is claimed and the
    worker naturally idles (we sleep for `timeout` to avoid a hot loop).

    Returns (job_dict, raw_string) or (None, None).
    """
    r = get_redis()

    if r.exists(_pause_key(queue_name)):
        time.sleep(min(timeout, 2))
        return None, None

    proc_key = _processing_key(queue_name)
    prio_lane, default_lane = _lanes(queue_name)

    job_raw = None
    if prio_lane:
        job_raw = r.rpoplpush(prio_lane, proc_key)          # atomic, non-blocking
    if not job_raw:
        job_raw = r.brpoplpush(default_lane, proc_key, timeout=timeout)  # atomic, blocking

    if not job_raw:
        return None, None

    job = json.loads(job_raw)
    job["claimed_at"] = time.time()
    r.hincrby(_metrics_key(), f"{queue_name}:claimed", 1)

    # Initial heartbeat so the reaper knows this claim is fresh
    try:
        heartbeat_job(queue_name, job.get("id", ""), worker="claim")
    except Exception:
        pass  # heartbeat is best-effort; claiming must never fail because of it

    return job, job_raw


# ═══════════════════════════════════════════════════════════
# HEARTBEAT — workers ping while processing long jobs
# ═══════════════════════════════════════════════════════════
def heartbeat_job(queue_name, job_id, worker="worker"):
    """Refresh the liveness marker for an in-flight job (TTL-based)."""
    if not job_id:
        return
    r = get_redis()
    r.set(_heartbeat_key(queue_name, job_id),
          json.dumps({"worker": worker, "ts": time.time()}),
          ex=HEARTBEAT_TTL_SEC)


class job_heartbeat:
    """
    Context manager that keeps a job's heartbeat alive from a background
    thread while the worker does long processing:

        with job_heartbeat(QUEUE_ANALYZE, job.get("id"), "model_manager"):
            process_analyze_job(payload)

    Best-effort by design — heartbeat failures never interrupt the job.
    """
    def __init__(self, queue_name, job_id, worker="worker", interval_sec=30):
        self.queue_name = queue_name
        self.job_id = job_id or ""
        self.worker = worker
        self.interval = interval_sec
        self._stop = None
        self._thread = None

    def __enter__(self):
        if not self.job_id:
            return self
        import threading
        self._stop = threading.Event()

        def _beat():
            while not self._stop.wait(self.interval):
                try:
                    heartbeat_job(self.queue_name, self.job_id, self.worker)
                except Exception:
                    pass

        try:
            heartbeat_job(self.queue_name, self.job_id, self.worker)
        except Exception:
            pass
        self._thread = threading.Thread(target=_beat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stop is not None:
            self._stop.set()
        return False  # never swallow worker exceptions


# ═══════════════════════════════════════════════════════════
# ACK — Mark job as successfully completed
# ═══════════════════════════════════════════════════════════
def ack_job(queue_name, job, job_raw):
    """Remove the job from PROCESSING and update metrics."""
    r = get_redis()
    pipe = r.pipeline()
    pipe.lrem(_processing_key(queue_name), 1, job_raw)
    pipe.hincrby(_metrics_key(), f"{queue_name}:completed", 1)
    pipe.delete(_heartbeat_key(queue_name, job.get("id", "")))

    if "claimed_at" in job:
        elapsed = time.time() - job["claimed_at"]
        pipe.hset(_metrics_key(), f"{queue_name}:last_duration_sec", f"{elapsed:.2f}")
        pipe.hset(_metrics_key(), f"{queue_name}:last_completed_at", f"{time.time():.2f}")

    pipe.execute()


# ═══════════════════════════════════════════════════════════
# FAIL — Retry or dead-letter the job
# ═══════════════════════════════════════════════════════════
def fail_job(queue_name, job, job_raw, error_msg):
    """Retry (< MAX_RETRIES) or dead-letter. Returns 'RETRIED' or 'DEAD'."""
    r = get_redis()
    r.lrem(_processing_key(queue_name), 1, job_raw)
    r.delete(_heartbeat_key(queue_name, job.get("id", "")))

    retries = job.get("retries", 0)
    _prio, default_lane = _lanes(queue_name)

    if retries < MAX_RETRIES:
        job["retries"] = retries + 1
        job["status"] = "RETRY"
        job["last_error"] = str(error_msg)[:200]
        job["last_failed_at"] = time.time()

        pipe = r.pipeline()
        pipe.lpush(default_lane, json.dumps(job))
        pipe.hincrby(_metrics_key(), f"{queue_name}:retries", 1)
        pipe.execute()
        return "RETRIED"
    else:
        job["status"] = "DEAD"
        job["last_error"] = str(error_msg)[:200]
        job["died_at"] = time.time()

        pipe = r.pipeline()
        pipe.lpush(_dlq_key(queue_name), json.dumps(job))
        pipe.hincrby(_metrics_key(), f"{queue_name}:dead", 1)
        pipe.execute()
        return "DEAD"


# ═══════════════════════════════════════════════════════════
# RECOVERY — Requeue orphaned processing jobs on boot
# ═══════════════════════════════════════════════════════════
def recover_processing_jobs(queue_name):
    """Move orphaned PROCESSING jobs back to the default lane. Returns count."""
    r = get_redis()
    proc_key = _processing_key(queue_name)
    _prio, default_lane = _lanes(queue_name)

    count = 0
    while True:
        orphan = r.rpoplpush(proc_key, default_lane)
        if not orphan:
            break
        count += 1

    if count > 0:
        r.hincrby(_metrics_key(), f"{queue_name}:recovered", count)
    return count


# ═══════════════════════════════════════════════════════════
# STALE-JOB REAPER — recover in-flight jobs whose worker died
# ═══════════════════════════════════════════════════════════
def reap_stale_jobs(queue_name=None, grace_sec=STALE_JOB_GRACE_SEC):
    """
    Scan PROCESSING lists for jobs whose worker heartbeat expired and whose
    claim is older than `grace_sec`. Requeue them (retries incremented so a
    poison job still dead-letters eventually). Returns {queue: reaped_count}.
    """
    r = get_redis()
    queues = [queue_name] if queue_name else list(ALL_QUEUES)
    now = time.time()
    result = {}

    for q in queues:
        proc_key = _processing_key(q)
        _prio, default_lane = _lanes(q)
        reaped = 0

        for job_raw in r.lrange(proc_key, 0, -1):
            try:
                job = json.loads(job_raw)
            except (json.JSONDecodeError, TypeError):
                # Corrupt entry — remove so it can't clog the list forever
                r.lrem(proc_key, 1, job_raw)
                continue

            job_id = job.get("id", "")
            claimed_at = float(job.get("claimed_at", job.get("created_at", now)))
            has_heartbeat = bool(job_id) and r.exists(_heartbeat_key(q, job_id))

            if not has_heartbeat and (now - claimed_at) > grace_sec:
                # Atomically-ish: only requeue if we successfully removed it
                if r.lrem(proc_key, 1, job_raw) == 1:
                    job["retries"] = job.get("retries", 0) + 1
                    job["status"] = "REAPED"
                    job["reaped_at"] = now
                    if job["retries"] > MAX_RETRIES:
                        job["status"] = "DEAD"
                        job["last_error"] = "reaped: exceeded retries after worker loss"
                        r.lpush(_dlq_key(q), json.dumps(job))
                        r.hincrby(_metrics_key(), f"{q}:dead", 1)
                    else:
                        r.lpush(default_lane, json.dumps(job))
                    reaped += 1

        if reaped:
            r.hincrby(_metrics_key(), f"{q}:reaped", reaped)
        result[q] = reaped

    return result


# ═══════════════════════════════════════════════════════════
# PAUSE / RESUME — per-queue flow control
# ═══════════════════════════════════════════════════════════
def pause_queue(queue_name):
    get_redis().set(_pause_key(queue_name), str(time.time()))
    return True

def resume_queue(queue_name):
    get_redis().delete(_pause_key(queue_name))
    return True

def is_paused(queue_name):
    return bool(get_redis().exists(_pause_key(queue_name)))


# ═══════════════════════════════════════════════════════════
# OPERATION LOCKS — prevent conflicting admin actions (multi-tab safe)
# ═══════════════════════════════════════════════════════════
def acquire_op_lock(name, ttl_sec=60):
    """SET NX lock. Returns a lock token, or None if already held."""
    r = get_redis()
    token = uuid.uuid4().hex
    if r.set(_oplock_key(name), token, nx=True, ex=ttl_sec):
        return token
    return None

def release_op_lock(name, token):
    """Release only if we still own it (compare-and-delete)."""
    r = get_redis()
    key = _oplock_key(name)
    if r.get(key) == token:
        r.delete(key)
        return True
    return False


# ═══════════════════════════════════════════════════════════
# METRICS — Full real-time observability
# ═══════════════════════════════════════════════════════════
def get_queue_metrics(queue_name=None):
    """Metrics for one or all queues: pending/processing/completed/DLQ/etc."""
    r = get_redis()
    queues = [queue_name] if queue_name else list(ALL_QUEUES)
    result = {}

    for q in queues:
        if q in PRIORITY_QUEUES:
            p_pending = r.llen(_priority_key(q))
            d_pending = r.llen(_default_key(q))
        else:
            p_pending = 0
            d_pending = r.llen(q)

        result[q] = {
            "pending_priority": p_pending,
            "pending_default": d_pending,
            "pending_total": p_pending + d_pending,
            "processing": r.llen(_processing_key(q)),
            "dead_letter": r.llen(_dlq_key(q)),
            "total_pushed": int(r.hget(_metrics_key(), f"{q}:pushed") or 0),
            "total_claimed": int(r.hget(_metrics_key(), f"{q}:claimed") or 0),
            "total_completed": int(r.hget(_metrics_key(), f"{q}:completed") or 0),
            "total_retries": int(r.hget(_metrics_key(), f"{q}:retries") or 0),
            "total_dead": int(r.hget(_metrics_key(), f"{q}:dead") or 0),
            "total_recovered": int(r.hget(_metrics_key(), f"{q}:recovered") or 0),
            "total_reaped": int(r.hget(_metrics_key(), f"{q}:reaped") or 0),
            "last_duration_sec": r.hget(_metrics_key(), f"{q}:last_duration_sec") or "N/A",
            "last_completed_at": r.hget(_metrics_key(), f"{q}:last_completed_at") or None,
            "paused": bool(r.exists(_pause_key(q))),
        }

    result["_global"] = {
        "dedup_set_size": r.scard("PROCESSED_VIDEOS_SET"),
    }

    return result


def get_queue_depth(queue_name):
    """Quick queue depth check (total pending jobs)."""
    r = get_redis()
    if queue_name in PRIORITY_QUEUES:
        return r.llen(_priority_key(queue_name)) + r.llen(_default_key(queue_name))
    return r.llen(queue_name)


# ═══════════════════════════════════════════════════════════
# DLQ MANAGEMENT
# ═══════════════════════════════════════════════════════════
def replay_dlq(queue_name, count=None):
    """Move dead-lettered jobs back to the default lane. Returns replay count."""
    r = get_redis()
    dlq = _dlq_key(queue_name)
    _prio, default_lane = _lanes(queue_name)
    replayed = 0

    while count is None or replayed < count:
        job_raw = r.rpop(dlq)
        if not job_raw:
            break
        job = json.loads(job_raw)
        job["retries"] = 0
        job["status"] = "PENDING"
        job["replayed_at"] = time.time()
        r.lpush(default_lane, json.dumps(job))
        replayed += 1

    return replayed


def peek_dlq(queue_name, count=10):
    """Inspect dead-lettered jobs without removing them."""
    r = get_redis()
    items = r.lrange(_dlq_key(queue_name), 0, count - 1)
    out = []
    for item in items:
        try:
            out.append(json.loads(item))
        except (json.JSONDecodeError, TypeError):
            out.append({"id": "corrupt", "payload": None, "last_error": "unparseable DLQ entry"})
    return out


def purge_dlq(queue_name):
    """Permanently delete all dead-lettered jobs for a queue. Returns count purged."""
    r = get_redis()
    dlq = _dlq_key(queue_name)
    count = r.llen(dlq)
    r.delete(dlq)
    return count


def peek_processing(queue_name, count=25):
    """Inspect in-flight jobs (with liveness info) without removing them."""
    r = get_redis()
    now = time.time()
    out = []
    for item in r.lrange(_processing_key(queue_name), 0, count - 1):
        try:
            job = json.loads(item)
        except (json.JSONDecodeError, TypeError):
            out.append({"id": "corrupt", "alive": False})
            continue
        job_id = job.get("id", "")
        job["alive"] = bool(job_id) and bool(r.exists(_heartbeat_key(queue_name, job_id)))
        job["in_flight_sec"] = round(now - float(job.get("claimed_at", now)), 1)
        out.append(job)
    return out


# ═══════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY — for model_manager.py boot sequence
# ═══════════════════════════════════════════════════════════
def pop_job(queue_name, timeout=2):
    """
    Legacy blocking pop (no claim/ack safety) used for QUEUE_MODELS boot.
    Handles both old-format payloads and job-envelope payloads.
    """
    r = get_redis()
    if queue_name in PRIORITY_QUEUES:
        job_raw = r.brpop([_priority_key(queue_name), _default_key(queue_name)], timeout=timeout)
    else:
        job_raw = r.brpop(queue_name, timeout=timeout)

    if not job_raw:
        return None

    data = json.loads(job_raw[1])
    if "payload" in data and "id" in data:
        return data["payload"]
    return data
