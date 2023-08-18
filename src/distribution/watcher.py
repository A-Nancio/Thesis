import tensorflow as tf
import keras

class PerformanceCallback(keras.callbacks.Callback):
    def __init__(self, model: tf.keras.Model, asynchrony_rate: int):
        self.model = model
        self.asynchrony_rate = asynchrony_rate # rate at which state is fetch and   updated in the database

    def on_test_begin(self, logs=None):
        print("Beginning tests")
    
    def on_test_end(self, logs=None):
        print("\nEnding test")
        # TODO gather overall results from execution
        

    def on_predict_begin(self, logs=None):
        print("predicting")
        
        # TODO start clock to register metrics
        #raise ValueError("Not implemented")
    
    def on_predict_end(self, logs=None):
        return
        #print("here")
        # TODO stop clock to register time taken for prediction

        # TODO start clock to register time to write state
        
        # TODO register model metrics

        # TODO fetch states from redis database

        # TODO merge states
        
        # TODO write new state in redis database

        # TODO stop clock to register time to write state


        # raise ValueError("Not implemented")

class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

