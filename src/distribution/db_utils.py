import redis
import numpy as np
import struct

database = redis.Redis(host='localhost', port=6379)
NUM_UNITS = 128

# NOTE numpy arrays MUST be in data type float64

def decode(encoded: bytes) -> np.ndarray:
    height, width = struct.unpack('>II',encoded[:8])
    # Add slicing here, or else the array would differ from the original
    return np.frombuffer(encoded[8:]).reshape(height,width)

def encode(array: np.ndarray) -> bytes:
    height, width = array.shape
    shape = struct.pack('>II',height,width)
    return shape + array.tobytes()


def reset_database():
    """Reset database to its base format by reseting the each card information to defautl values"""
    initial_state = np.zeros((1, NUM_UNITS), dtype=np.float64)
    for key in range(1000):
        if key == 288:
            print("RESETING KEY")
        to_redis(str(key), initial_state)   # set state initialized


def from_redis(key):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = database.get(key)

    height, width = struct.unpack('>II',encoded[:8])
    # Add slicing here, or else the array would differ from the original
    array = np.frombuffer(encoded[8:]).reshape(height,width)
    return array


def to_redis(key: str, state: np.ndarray):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    height, width = state.shape
    shape = struct.pack('>II',height,width)
    encoded = shape + state.tobytes()

    # Store encoded data in Redis
    database.set(key, encoded)


def add_deltas_to_redis(key, deltas: np.ndarray):
    with database.pipeline() as pipe:
        while True:
            try:
                pipe.watch(key)
                current_value = decode(pipe.get(key))
                next_value = np.add(current_value, deltas) + 1
                pipe.multi()
                
                height, width = next_value.shape
                shape = struct.pack('>II',height,width)
                encoded = shape + next_value.tobytes()
                pipe.set(key, encoded)
                
                pipe.execute()
                break
            except redis.WatchError:
                pass
                # show which counter is used by another user
                #print("revoked:", next_value)
                #error_count += 1



