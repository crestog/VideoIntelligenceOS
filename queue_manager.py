"""
VIOS Queue Manager v2 — Enterprise Reliable Job Queue System

Architecture:
  - Dual-lane priority routing (PRIORITY + DEFAULT lanes)
  - Reliable delivery (claim → processing set → ack/fail)
  - Auto-retry with dead letter queue after MAX_RETRIES
  - Boot crash recovery for orphaned processing jobs
  - Real-time metrics for full observability
  - DLQ management (replay, peek, flush)

Job Lifecycle:
  PUSH → [PRIORITY|DEFAULT] → CLAIM → PROCESSING → ACK (completed)
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


# ═══════════════════════════════════════════════════════════
# CONNECTION POOL
# ═══════════════════════════════════════════════════════════
redis_pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)

def get_redis():
    return redis.Redis(connection_pool=redis_pool)


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
PRIORITY_QUEUES = {"QUEUE_VISION"}   # Queues with dual-lane priority routing
MAX_RETRIES = 3                       # Retry attempts before dead-lettering


# ═══════════════════════════════════════════════════════════
# KEY SCHEMA HELPERS
# ═══════════════════════════════════════════════════════════
def _priority_key(q):    return f"{q}_PRIORITY"
def _default_key(q):     return f"{q}_DEFAULT"
def _processing_key(q):  return f"{q}_PROCESSING"
def _dlq_key(q):         return f"{q}_DLQ"
def _metrics_key():      return "VIOS_METRICS"


# ═══════════════════════════════════════════════════════════
# PUSH — Route a job into the queue system
# ═══════════════════════════════════════════════════════════
def push_job(queue_name, payload, is_priority=False):
    """
    Push a job to the queue system with a unique ID and metadata envelope.
    For priority-enabled queues, routes to PRIORITY or DEFAULT lane.
    Returns the generated job ID.
    """
    r = get_redis()
    job_id = f"job:{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

    job = {
        "id": job_id,
        "payload": payload,
        "created_at": time.time(),
        "retries": 0,
    }
    job_data = json.dumps(job)

    if queue_name in PRIORITY_QUEUES:
        lane = _priority_key(queue_name) if is_priority else _default_key(queue_name)
    else:
        lane = queue_name

    pipe = r.pipeline()
    pipe.rpush(lane, job_data)
    pipe.hincrby(_metrics_key(), f"{queue_name}:pushed", 1)
    pipe.execute()

    return job_id


# ═══════════════════════════════════════════════════════════
# CLAIM — Reliable pop with processing set
# ═══════════════════════════════════════════════════════════
def claim_job(queue_name, timeout=2):
    """
    Claim a job from the queue with reliable delivery.
    
    For priority queues: drains PRIORITY lane first (non-blocking),
    then falls back to DEFAULT lane (blocking with timeout).
    
    The claimed job is copied to the PROCESSING set. If the worker
    crashes before ack/fail, the job survives and can be recovered on reboot.
    
    Returns (job_dict, raw_string) or (None, None) if no jobs available.
    """
    r = get_redis()
    proc_key = _processing_key(queue_name)
    job_raw = None

    if queue_name in PRIORITY_QUEUES:
        # 1. Non-blocking FIFO drain of priority lane
        job_raw = r.lpop(_priority_key(queue_name))
        if not job_raw:
            # 2. Blocking FIFO wait on default lane
            result = r.blpop(_default_key(queue_name), timeout=timeout)
            if result:
                job_raw = result[1]
    else:
        result = r.blpop(queue_name, timeout=timeout)
        if result:
            job_raw = result[1]

    if not job_raw:
        return None, None

    # Move to processing set for crash safety
    r.lpush(proc_key, job_raw)

    job = json.loads(job_raw)
    job["claimed_at"] = time.time()
    r.hincrby(_metrics_key(), f"{queue_name}:claimed", 1)

    return job, job_raw


# ═══════════════════════════════════════════════════════════
# ACK — Mark job as successfully completed
# ═══════════════════════════════════════════════════════════
def ack_job(queue_name, job, job_raw):
    """
    Acknowledge successful job completion.
    Removes the job from the PROCESSING set and updates metrics.
    """
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
    """
    Handle job failure with automatic retry logic.
    
    If retries < MAX_RETRIES: push back to DEFAULT lane with incremented counter.
    If retries >= MAX_RETRIES: move to dead letter queue.
    
    Returns 'RETRIED' or 'DEAD'.
    """
    r = get_redis()
    r.lrem(_processing_key(queue_name), 1, job_raw)

    retries = job.get("retries", 0)

    if retries < MAX_RETRIES:
        job["retries"] = retries + 1
        job["status"] = "RETRY"
        job["last_error"] = str(error_msg)[:200]
        job["last_failed_at"] = time.time()

        target = _default_key(queue_name) if queue_name in PRIORITY_QUEUES else queue_name

        pipe = r.pipeline()
        pipe.rpush(target, json.dumps(job))
        pipe.hincrby(_metrics_key(), f"{queue_name}:retries", 1)
        pipe.execute()
        return "RETRIED"
    else:
        job["status"] = "DEAD"
        job["last_error"] = str(error_msg)[:200]
        job["died_at"] = time.time()

        pipe = r.pipeline()
        pipe.rpush(_dlq_key(queue_name), json.dumps(job))
        pipe.hincrby(_metrics_key(), f"{queue_name}:dead", 1)
        pipe.execute()
        return "DEAD"


# ═══════════════════════════════════════════════════════════
# RECOVERY — Requeue orphaned processing jobs on boot
# ═══════════════════════════════════════════════════════════
def recover_processing_jobs(queue_name):
    """
    Boot-time crash recovery.
    Moves any orphaned jobs from PROCESSING back to the DEFAULT lane.
    These are jobs claimed by a worker that crashed before calling ack/fail.
    Returns the count of recovered jobs.
    """
    r = get_redis()
    proc_key = _processing_key(queue_name)
    target = _default_key(queue_name) if queue_name in PRIORITY_QUEUES else queue_name

    count = 0
    while True:
        orphan = r.rpoplpush(proc_key, target)
        if not orphan:
            break
        count += 1

    if count > 0:
        r.hincrby(_metrics_key(), f"{queue_name}:recovered", count)
    return count


# ═══════════════════════════════════════════════════════════
# METRICS — Full real-time observability
# ═══════════════════════════════════════════════════════════
def get_queue_metrics(queue_name=None):
    """
    Get comprehensive real-time metrics for one or all queues.
    Returns a dict of queue_name → {pending, processing, completed, etc.}
    """
    r = get_redis()
    queues = [queue_name] if queue_name else list(PRIORITY_QUEUES) + ["QUEUE_MODELS"]
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
    """
    Move dead-lettered jobs back to the default queue for reprocessing.
    Resets retry counter to 0. If count is None, replays all.
    Returns the number of replayed jobs.
    """
    r = get_redis()
    dlq = _dlq_key(queue_name)
    target = _default_key(queue_name) if queue_name in PRIORITY_QUEUES else queue_name
    replayed = 0

    while count is None or replayed < count:
        job_raw = r.lpop(dlq)
        if not job_raw:
            break
        job = json.loads(job_raw)
        job["retries"] = 0
        job["status"] = "PENDING"
        job["replayed_at"] = time.time()
        r.rpush(target, json.dumps(job))
        replayed += 1

    return replayed


def peek_dlq(queue_name, count=10):
    """Inspect dead-lettered jobs without removing them."""
    r = get_redis()
    items = r.lrange(_dlq_key(queue_name), 0, count - 1)
    return [json.loads(item) for item in items]


# ═══════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY — for model_manager.py
# ═══════════════════════════════════════════════════════════
def pop_job(queue_name, timeout=2):
    """
    Legacy blocking pop used by model_manager.py for the QUEUE_MODELS boot sequence.
    Does NOT use the reliable claim/ack/fail pattern.
    
    For new consumers, use the claim_job → ack_job → fail_job lifecycle.
    
    Handles both old-format payloads (direct dict) and new job-envelope payloads.
    """
    r = get_redis()
    if queue_name in PRIORITY_QUEUES:
        job_raw = r.blpop([_priority_key(queue_name), _default_key(queue_name)], timeout=timeout)
    else:
        job_raw = r.blpop(queue_name, timeout=timeout)

    if not job_raw:
        return None

    data = json.loads(job_raw[1])
    # Unwrap job envelope for backward compatibility
    if "payload" in data and "id" in data:
        return data["payload"]
    return data
