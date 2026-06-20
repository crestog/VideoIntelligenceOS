import redis
import json

redis_pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
def get_redis(): return redis.Redis(connection_pool=redis_pool)

def push_job(queue_name, payload):
    get_redis().rpush(queue_name, json.dumps(payload))
