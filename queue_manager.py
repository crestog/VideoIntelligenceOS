import redis
import json
redis_pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
def get_redis(): return redis.Redis(connection_pool=redis_pool)
def push_job(queue_name, payload): get_redis().lpush(queue_name, json.dumps(payload))
def pop_job(queue_name, timeout=2):
    job_raw = get_redis().blpop(queue_name, timeout=timeout)
    return json.loads(job_raw[1]) if job_raw else None
