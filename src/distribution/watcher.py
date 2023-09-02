import tensorflow as tf
import keras
from db_utils import fromRedis, toRedis

class DistributedPerformance(keras.callbacks.Callback):
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
        print(f"Predicting an input: {batch}")
        fromRedis()
        return
        

    def on_test_batch_begin(self, batch, logs=None):
        return
        #print(f"Predicted an input: {batch}")
