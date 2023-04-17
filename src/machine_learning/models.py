"""Tensorflow model and layer classes"""
import tensorflow as tf
from keras.layers import concatenate, Dense, GRU, Dropout


# Create model
class Stateless(tf.keras.Model):
    """A model which stores no state when processing input"""

    def __init__(self):
        super(Stateless, self).__init__()
        self.layer1 = Dense(64, activation="relu")
        self.layer2 = Dense(64, activation="relu")
        self.out = Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        inputs = self.layer1(inputs)
        inputs = self.layer2(inputs)
        return self.out(inputs)


class GlobalState(tf.keras.Model):
    """A model that stores an internal state relevant to all transactions"""

    def __init__(self, input_shape):
        super(GlobalState, self).__init__()
        self.dropout = Dropout(0.2, input_shape=(input_shape,))
        self.gru = GRU(64, return_sequences=False,
                       dropout=0.3, recurrent_regularizer='l2')
        self.dense = Dense(32, activation='relu')

        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.dropout(inputs)
        var = self.gru(var)
        var = concatenate([inputs, var], axis=1)
        var = self.dense(var)
        return self.out(var)


# class SharedState(GlobalState):
#    def __init__(self):
#        super(SharedState, self).__init__()
#
    # TODO override training_step to change the gru state of the model
