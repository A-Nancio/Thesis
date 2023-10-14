import redis
import numpy as np
import struct
import json



local_address = "127.0.0.1"
database = redis.Redis(host='localhost', port=6379)
NUM_UNITS = 128

# NOTE numpy arrays MUST be in data type float64

def reset_database():
    """Reset database to its base format by reseting the each card information to defautl values"""
    database.flushall()

def delete_states():
    for key in database.keys("delta_*"):
        database.delete(key)

def from_redis(key):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = database.get(key)

    height, width = struct.unpack('>II',encoded[:8])
    # Add slicing here, or else the array would differ from the original
    array = np.frombuffer(encoded[8:]).reshape(height,width)
    return array


def to_redis(key: str, state: np.ndarray):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    state = state.astype('float64')
    height, width = state.shape
    shape = struct.pack('>II',height,width)
    encoded = shape + state.tobytes()

    # Store encoded data in Redis
    database.set(key, encoded, ex=10)


def register_results(thrs, pool_size, performance_results, time_results):
    loss, acc, TP, TN, FP, FN, precision, recall = performance_results
    #_, _, _, _, precision, recall 
    total_time, avg_pass, throughput = time_results
    database.lpush(f"TEST: Loss_thres_{thrs}_pool_{pool_size}", loss)
    database.lpush(f"TEST: Acc_thres_{thrs}_pool_{pool_size}", acc)
    database.lpush(f"TEST: TP_thres_{thrs}_pool_{pool_size}", TP)
    database.lpush(f"TEST: TN_thres_{thrs}_pool_{pool_size}", TN)
    database.lpush(f"TEST: FP_thres_{thrs}_pool_{pool_size}", FP)
    database.lpush(f"TEST: FN_thres_{thrs}_pool_{pool_size}", FN)
    database.lpush(f"TEST: Precision_thres_{thrs}_pool_{pool_size}", precision)
    database.lpush(f"TEST: Recall_thres_{thrs}_pool_{pool_size}", recall)
    database.lpush(f"TEST: TotalTime_thres_{thrs}_pool_{pool_size}", total_time)
    database.lpush(f"TEST: AvgPass_thres_{thrs}_pool_{pool_size}", avg_pass)
    database.lpush(f"TEST: Throughput_thres_{thrs}_pool_{pool_size}", throughput)
