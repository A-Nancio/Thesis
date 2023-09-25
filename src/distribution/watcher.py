import keras
import time

from redis import StrictRedis

from distribution.db_utils import from_redis, to_redis

class PerformanceTracker(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.forward_pass_times = []

    def on_test_begin(self, logs=None):
        self.start_execution_time = time.time()

    def on_test_end(self, logs=None):
        self.end_execution_time = time.time()

    def on_test_batch_begin(self, batch, logs=None):
        self.start_batch = time.time()        
        
    def on_test_batch_end(self, batch, logs=None):
        self.end_batch = time.time()
        self.forward_pass_times.append(self.end_batch - self.start_batch)

    def display_results(self):
        total_execution_time = self.end_execution_time - self.start_execution_time
        average_forward_pass = sum(self.forward_pass_times) / len(self.forward_pass_times)
        throughput = len(self.forward_pass_times) / total_execution_time

        return total_execution_time, average_forward_pass, throughput


class StateWriter(keras.callbacks.Callback):
    def __init__(self, id: int, threshold):
        super().__init__()
        self.id = id
        self.threshold = threshold
        self.version = 0
        self.staleness = 0
    
    def on_test_batch_end(self, batch, logs=None):
        self.staleness += 1
        if self.staleness > self.threshold:
            self.staleness = 0
            self.version += 1

            deltas = self.model.category_gru.deltas.numpy()
            to_redis(f'delta_{self.id}_v{self.version}', deltas)
            
            self.model.category_gru.reset_deltas()


def subscribe(id, model):    
    def event_handler(msg):
        key = msg["data"].decode("utf-8")

        if "delta_" in key and f'delta_{id}' not in key:
            deltas = from_redis(key)
            model.category_gru.weights[0].assign_add(deltas)
            
            # print(f'[WORKER {id}]: Notified {key}')

    redis_server = StrictRedis(host='localhost', port=6379, db=0)
    
    pubsub = redis_server.pubsub()
    pubsub.psubscribe(**{"__keyevent@0__:set": event_handler})
    thread = pubsub.run_in_thread(sleep_time=.01)
    
    print(f"[WORKER {id}]: subscribed")
    return thread