import keras
import time

from distribution.db_utils import add_deltas_to_redis, to_redis
import sys

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

    def display_results(self) -> str:
        total_execution_time = self.end_execution_time - self.start_execution_time
        average_forward_pass = sum(self.forward_pass_times) / len(self.forward_pass_times)
        throughput = len(self.forward_pass_times) / total_execution_time

        return f"Total time: {total_execution_time} s, Average forward pass: {average_forward_pass} s, Throughput: {throughput} s"


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
            #TODO write states
            self.staleness = 0
            self.version += 1
            deltas = self.model.card_gru.deltas.numpy()
            to_redis(f'{self.id}_v{self.version}', deltas)


class Example(keras.callbacks.Callback):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))
        print(self.model.get_state())

    def on_test_batch_begin(self, batch, logs=None):
        print(f"Predicted an input: {batch}")
