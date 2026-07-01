import redis
import json

redis_pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)

def get_redis(): 
    return redis.Redis(connection_pool=redis_pool)

def push_job(queue_name, payload, is_priority=False):
    '''
    DUAL-LANE ROUTER: 
    Both lanes use rpush to maintain strict FIFO chronological order.
    '''
    if queue_name == "QUEUE_VISION":
        target_lane = "QUEUE_VISION_PRIORITY" if is_priority else "QUEUE_VISION_DEFAULT"
        get_redis().rpush(target_lane, json.dumps(payload))
    else:
        get_redis().rpush(queue_name, json.dumps(payload))

def pop_job(queue_name, timeout=2):
    '''
    DUAL-LANE CONSUMER:
    blpop checks PRIORITY lane first. If empty, it checks DEFAULT lane.
    If both empty, it sleeps (0% CPU).
    '''
    if queue_name == "QUEUE_VISION":
        job_raw = get_redis().blpop(["QUEUE_VISION_PRIORITY", "QUEUE_VISION_DEFAULT"], timeout=timeout)
    else:
        job_raw = get_redis().blpop(queue_name, timeout=timeout)
        
    return json.loads(job_raw[1]) if job_raw else None
