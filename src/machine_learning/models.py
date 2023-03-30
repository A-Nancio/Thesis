import tensorflow as tf
import numpy as np
from keras.layers import concatenate, Dense, GRU

# Create model
class Stateless(tf.keras.Model):
    def __init__(self):
        super(Stateless, self).__init__()
        self.layer1 = Dense(64, activation="relu")
        self.layer2 = Dense(64, activation="relu")
        self.out = Dense(1, activation="softmax")
    
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return self.out(x)

class GlobalState(tf.keras.Model):
    def __init__(self, transactions):
        super(GlobalState, self).__init__()
        self.transactions = transactions
        self.GRU = GRU(64, return_sequences=False, dropout=0.3, recurrent_regularizer='l2')
        self.dense = Dense(32, activation='relu')
        #self.concat = concatenate
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs):

        #inputs.numpy
        x = self.GRU(inputs)
        #x = concatenate(x, inputs)
        x = self.dense(x)
        return self.out(x)


#class SharedState(GlobalState):
#    def __init__(self):
#        super(SharedState, self).__init__()
#
    #TODO override training_step to change the gru state of the model

