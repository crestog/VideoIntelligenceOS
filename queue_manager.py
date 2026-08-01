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
import socket
import time
import uuid

from config import REDIS_HOST, REDIS_PORT


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


def _safe_print(msg):
    """
    print() that cannot raise. Console encoding is not guaranteed (a cp1252
    Windows terminal rejects emoji), and this module's whole job on the failure
    path is to report cleanly — a logging call must never become the exception.
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
    except Exception:
        pass


def wait_for_redis(timeout=30.0, label="", probe_interval=1.0):
    """
    Block until Redis accepts a connection, or give up after `timeout` seconds.

    Workers must call this before their first queue op. Without it a worker that
    starts before the broker dies on ECONNREFUSED, the watchdog restarts it 3s
    later, and the whole stack loops forever with no useful diagnostic.

    Uses a raw bounded socket probe rather than redis-py's ping(): the client
    pool applies its own internal retry/backoff under every command, so a
    "30 x 1s" loop built on ping() actually ran for minutes. A plain TCP connect
    is what "is the broker listening?" really means, and it honours the deadline.

    Returns True if Redis is reachable, False otherwise (caller decides whether
    to degrade or exit — this function never raises).
    """
    tag = f"[{label}] " if label else ""
    deadline = time.monotonic() + timeout
    announced = False

    while True:
        try:
            with socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=2) as sock:
                sock.sendall(b"PING\r\n")
                if b"PONG" in sock.recv(64).upper():
                    if announced:
                        _safe_print(f"   ✅ {tag}Redis is up.")
                    return True
        except OSError as e:
            if not announced:
                _safe_print(f"   ⏳ {tag}Waiting for Redis... ({type(e).__name__})")
                announced = True

        if time.monotonic() + probe_interval >= deadline:
            _safe_print(f"   ❌ {tag}Redis unreachable after {timeout:.0f}s "
                        f"({REDIS_HOST}:{REDIS_PORT}).")
            return False
        time.sleep(probe_interval)


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
# Queues with dual-lane priority routing.
# QUEUE_OMNI_*: Telegram-bot uploads ride the PRIORITY lane, Ghost Worker
# harvest jobs ride the DEFAULT lane — the bot always wins.
PRIORITY_QUEUES = {"QUEUE_VISION", "QUEUE_OMNI_VISION", "QUEUE_OMNI_ORACLE"}
ALL_QUEUES = ["QUEUE_VISION", "QUEUE_ANALYZE", "QUEUE_MODELS",
              "QUEUE_OMNI_VISION", "QUEUE_OMNI_ORACLE"]
MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════
# KEY SCHEMA HELPERS
# ═══════════════════════════════════════════════════════════
def _priority_key(q):    return f"{q}_PRIORITY"
def _default_key(q):     return f"{q}_DEFAULT"
def _processing_key(q):  return f"{q}_PROCESSING"
def _dlq_key(q):         return f"{q}_DLQ"
def _metrics_key():      return "VIOS_METRICS"

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

    Returns (job_dict, raw_string) or (None, None).
    """
    r = get_redis()
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

    return job, job_raw


# ═══════════════════════════════════════════════════════════
# ACK — Mark job as successfully completed
# ═══════════════════════════════════════════════════════════
def ack_job(queue_name, job, job_raw):
    """Remove the job from PROCESSING and update metrics."""
    r = get_redis()
    pipe = r.pipeline()
    pipe.lrem(_processing_key(queue_name), 1, job_raw)
    pipe.hincrby(_metrics_key(), f"{queue_name}:completed", 1)

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
def recover_processing_jobs(queue_name, max_recoveries=None):
    """
    Move orphaned PROCESSING jobs back to the default lane. Returns the number
    actually re-queued.

    `max_recoveries` guards against a poison pill. A worker that dies *while*
    holding a job — the correct response to an unrecoverable CUDA fault — leaves
    that job in PROCESSING. Recovering it is right when the crash was
    incidental, but if the job itself is what kills the worker, recovery and
    restart chase each other forever and nothing else in the queue ever runs.
    Past `max_recoveries` the job is dead-lettered instead, so the backlog keeps
    moving and the offender stays inspectable in the DLQ.

    Left as None (boot.py's whole-session sweep) the original unconditional
    behaviour is preserved exactly.
    """
    r = get_redis()
    proc_key = _processing_key(queue_name)
    _prio, default_lane = _lanes(queue_name)

    count = 0
    quarantined = 0
    while True:
        # RPOPLPUSH rather than RPOP: the move is atomic, so a crash mid-recovery
        # can duplicate a job but can never drop one.
        orphan = r.rpoplpush(proc_key, default_lane)
        if not orphan:
            break
        count += 1

        if max_recoveries is None:
            continue

        try:
            job = json.loads(orphan)
        except (ValueError, TypeError):
            continue                      # unparseable envelope — leave it requeued as-is

        job["recoveries"] = job.get("recoveries", 0) + 1
        pipe = r.pipeline()
        pipe.lrem(default_lane, 1, orphan)          # drop the copy just moved
        if job["recoveries"] > max_recoveries:
            job["status"] = "DEAD"
            job["last_error"] = (f"Crashed the worker {job['recoveries']} times — "
                                 f"quarantined to keep the queue moving.")
            job["died_at"] = time.time()
            pipe.lpush(_dlq_key(queue_name), json.dumps(job))
            pipe.hincrby(_metrics_key(), f"{queue_name}:dead", 1)
            count -= 1
            quarantined += 1
        else:
            pipe.lpush(default_lane, json.dumps(job))   # same position, updated count
        pipe.execute()

    if count > 0:
        r.hincrby(_metrics_key(), f"{queue_name}:recovered", count)
    if quarantined > 0:
        _safe_print(f"   ☠️ {queue_name}: quarantined {quarantined} job(s) that "
                    f"repeatedly crashed the worker — see the DLQ.")
    return count


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
            "last_duration_sec": r.hget(_metrics_key(), f"{q}:last_duration_sec") or "N/A",
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
    return [json.loads(item) for item in items]


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
