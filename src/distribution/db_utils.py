import redis
import numpy as np
database = redis.Redis(host='localhost', port=6379, decode_responses=True)

database.set('foo', 'bar')
# True
database.get('foo')
# bar

def fetch_state(db: redis.Redis, node: int, id: int) -> np.ndarray:
    