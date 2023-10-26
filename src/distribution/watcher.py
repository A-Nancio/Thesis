import keras
import time
import abc
from redis import StrictRedis
import queue
from distribution.db_utils import from_redis, to_redis, database
import tensorflow as tf
import numpy as np
import sys
QUEUE_LIMIT = 10

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
        total_execution_time = self.end_execution_time - \
                               self.start_execution_time
        average_forward_pass = sum(self.forward_pass_times) / \
            len(self.forward_pass_times)
        throughput = len(self.forward_pass_times) / total_execution_time

        return total_execution_time, average_forward_pass, throughput


# --------------- NOTE ASYNCHRONOUS ----------------

class Asynchronous(keras.callbacks.Callback, abc.ABC):
    def __init__(self, id: int, threshold: int, num_workers: int):
        super().__init__()
        self.id = id
        self.threshold = threshold
        self.num_workers = num_workers

        self.version = 0
        self.staleness = 0
        self.read_time = 0
        self.write_time = 0

        self.avg_count = {}

        self.queue = queue.Queue()

        self.thread = self.subscribe(id)

    def on_test_batch_end(self, batch, logs=None):
        self.staleness += 1
        if self.staleness >= self.threshold:
            self.staleness = 0
            self.version += 1
            self.write_update()
    
    def subscribe(self, id):    
        def event_handler(msg):
            key = msg["data"].decode("utf-8")
            if "delta_" in key and f'delta_{id}' not in key:
                self.queue.put(key)
                self.read_update(self.queue)

        redis_server = StrictRedis(host='localhost', port=6379, db=0)

        pubsub = redis_server.pubsub()
        pubsub.psubscribe(**{"__keyevent@0__:set": event_handler})
        thread = pubsub.run_in_thread(sleep_time=.01)

        print(f"[WORKER {id}]: subscribed")
        return thread

    def display_results(self):
        return self.read_time, self.write_time

    def close_threads(self):
        self.thread.stop()

class AsynchronousBoundSum(Asynchronous):
    name = "async_bound_sum"
    def write_update(self):
        deltas = self.model.category_gru.deltas.numpy()

        start_time = time.time()
        to_redis(f'delta_{self.id}_v{self.version}', deltas)
        self.write_time += time.time() - start_time

        self.model.category_gru.reset_deltas()

    def read_update(self, key):
        start_time = time.time()
        deltas = from_redis(key)
        self.read_time += time.time() - start_time

        deltas = deltas.astype('float32')
        new_states = self.model.category_gru.shared_states.numpy() + deltas
        new_states = np.minimum(new_states, 1)
        new_states = np.maximum(new_states, -1)
        
        self.model.category_gru.shared_states.assign(new_states)

class AsynchronousAverage(Asynchronous):
    name = "async_average"
    def write_update(self):
        states = self.model.category_gru.shared_states.numpy()

        start_time = time.time()
        to_redis(f'delta_{self.id}_v{self.version}', states)
        self.write_time += time.time() - start_time

    def read_update(self, key):
        _, _, version = key.split("_")
        if version not in self.avg_count:
            self.avg_count[version] = 1
        else:
            self.avg_count[version] += 1

        start_time = time.time()
        states = from_redis(key)
        self.read_time += time.time() - start_time

        states = states.astype('float32')
        aux1 = tf.math.multiply(self.model.category_gru.shared_states,
                                self.avg_count[version]/(self.avg_count[version]+1))
        aux2 = tf.math.divide(states, self.avg_count[version]+1)

        res = tf.math.add(aux1, aux2)
        self.model.category_gru.shared_states.assign(res)



# --------------- NOTE SYNCHRONOUS ----------------

class Synchronous(keras.callbacks.Callback, abc.ABC):
    def __init__(self, id: int, threshold: int, num_workers: int):
        super().__init__()
        self.id = id
        self.threshold = threshold
        self.num_workers = num_workers

        self.version = 0
        self.staleness = 0
        self.read_time = 0
        self.write_time = 0

    def close_threads(self):
        return  # Do nothing, since we use no threads
    
    def display_results(self):
        return self.read_time, self.write_time

class SynchronousAverage(Synchronous):
    name = "sync_average"
    def on_test_batch_end(self, batch, logs=None):
        self.staleness += 1
        if self.staleness >= self.threshold:
            self.staleness = 0
            self.version += 1

            sum = self.model.category_gru.shared_states.numpy()

            start_time = time.time()
            to_redis(f'delta_{self.id}_v{self.version}', sum)
            self.write_time += time.time() - start_time

            count = 1      
            for worker_id in range(self.num_workers):
                if worker_id == self.id:
                    continue

                start_time = time.time()
                val = from_redis(f'delta_{worker_id}_v{self.version}')
                self.read_time +=  time.time() - start_time
                if val is None:
                    continue
                sum += val
                count += 1

            self.model.category_gru.shared_states.assign(np.divide(sum, count).astype('float32'))

class SynchronousBoundSum(Synchronous):
    name = "sync_bound_sum"
    def on_test_batch_end(self, batch, logs=None):
        self.staleness += 1
        if self.staleness >= self.threshold:
            self.staleness = 0
            self.version += 1

            start_time = time.time()
            to_redis(f'delta_{self.id}_v{self.version}', self.model.category_gru.deltas.numpy())
            self.write_time += time.time() - start_time
            self.model.category_gru.reset_deltas()

            for worker_id in range(self.num_workers):
                if worker_id == self.id:
                    continue

                start_time = time.time()
                deltas = from_redis(f'delta_{worker_id}_v{self.version}')
                self.read_time += time.time() - start_time
                if deltas is None:
                    continue

                deltas = deltas.astype('float32')
                new_states = self.model.category_gru.shared_states.numpy() + deltas
                new_states = np.minimum(new_states, 1)
                new_states = np.maximum(new_states, -1)
                
                self.model.category_gru.shared_states.assign(new_states)
