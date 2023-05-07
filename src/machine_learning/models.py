"""Tensorflow model and layer classes"""
import tensorflow as tf
from keras.layers import concatenate, Dense, GRU, Dropout


stateless_model = tf.keras.models.Sequential([
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

StatefulModel = tf.keras.models.Sequential([
    Dropout(0.2),
    GRU(64, return_sequences=False, dropout=0.3, recurrent_regularizer='l2'),
    Dense(1, activation="sigmoid"),
    Dense(32, activation='relu'),
    Dense(1,  activation="sigmoid")
])

class SharedState(tf.keras.Model):
    def __init__(self):
        super(GlobalState, self).__init__()
        self.dropout = Dropout(0.2)
        self.card_gru = GRU(64, return_sequences=False,
                       dropout=0.3, recurrent_regularizer='l2')
        self.zip_gru = GRU(64, return_sequences=False,
                       dropout=0.3, recurrent_regularizer='l2')
        self.dense = Dense(32, activation='relu')

        self.out = Dense(1,  activation="sigmoid")
    

    def call(self, inputs):
        var = self.dropout(inputs)
        #if inputs[...] 
        #card_gru.change_state(inputs[new_card])
        var = self.gru(var)
        #var = concatenate([inputs, var], axis=1)
        var = self.dense(var)
        out = self.out(var)
        return out

class CardStateful(tf.keras.Model):
    """A model that stores an internal state relevant to each card"""

    def __init__(self, transactions):
        self.transactions = transactions
        super(GlobalState, self).__init__()
        self.dropout = Dropout(0.2)
        self.gru = GRU(64, return_sequences=False,
                       dropout=0.3, recurrent_regularizer='l2')
        #self.zip_gru = GRU(64, return_sequences=False,
        #               dropout=0.3, recurrent_regularizer='l2')
        self.dense = Dense(32, activation='relu')

        self.out = Dense(1,  activation="sigmoid")
    
    
    def call(self, inputs):
        var = self.dropout(inputs)
        #card_gru.change_state(inputs[new_card])
        var = self.gru(var)
        #var = concatenate([inputs, var], axis=1)
        var = self.dense(var)
        out = self.out(var)
        return out

    def train_step(self, data):
        return
    
    #def test_step(self, data)

# class SharedState(GlobalState):
#    def __init__(self):
#        super(SharedState, self).__init__()
#
    # TODO override training_step to change the gru state of the model
