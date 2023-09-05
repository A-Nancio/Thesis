import keras
import time


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
