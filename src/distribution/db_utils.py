import redis
import numpy as np
import struct

database = redis.Redis(host='localhost', port=6379, decode_responses=True)

# NOTE numpy arrays MUST be in data type float64

def toRedis(key: str, state: np.ndarray):
   """Store given Numpy array 'a' in Redis under key 'n'"""
   h, w = state.shape
   shape = struct.pack('>II',h,w)
   encoded = shape + state.tobytes()
   
   # Store encoded data in Redis
   database.set(key, encoded)
   return

def fromRedis(key):
   """Retrieve Numpy array from Redis key 'n'"""
   encoded = database.get(key)
   h, w = struct.unpack('>II',encoded[:8])
   # Add slicing here, or else the array would differ from the original
   a = np.frombuffer(encoded[8:]).reshape(h,w)
   return a