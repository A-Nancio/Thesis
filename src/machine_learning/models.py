"""Tensorflow model and layer classes"""
import tensorflow as tf
from keras.layers import concatenate, Dense, Dropout

from machine_learning.shared_state import SharedState, AsynchronousSharedState

CARD_ID_COLUMN = 0
CATEOGRY_ID_COLUMN = 1

class FeedzaiProduction(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=128, id_column=CARD_ID_COLUMN)    #NOTE change for sync to evaluate when synchronizing
        self.layer = Dense(units=128, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var, _ = self.card_gru(inputs)
        var = self.layer(var)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)
        
        return out

    def reset_gru(self):
        self.card_gru.reset_states()

    def get_state(self):
        return self.card_gru.weights[0].value().numpy()
    
    def set_state(self, new_state):
        self.card_gru.weights[0].assign(new_state)
        

class DoubleStateProduction(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=128, id_column=CARD_ID_COLUMN)    #NOTE change for sync to evaluate when synchronizing
        self.category_gru = SharedState(units=128, id_column=CARD_ID_COLUMN)
        self.layer = Dense(units=128, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_output, _ = self.card_gru(inputs)
        category_output, _ = self.category_gru(inputs)
        var = concatenate([card_output, category_output])
        var = self.layer(var)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()
        self.category_gru.reset_states()

    def get_state(self):
        return (self.card_gru.weights[0].value().numpy(), self.category_gru.weights[0].value().numpy())
    
    def set_card_state(self, new_state):
        self.card_gru.weights[0].assign(new_state)

    def set_category_state(self, new_state):
        self.category_gru.weights[0].assign(new_state)


class DistributedProducion(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = AsynchronousSharedState(units=128, id_column=CARD_ID_COLUMN)    #NOTE change for sync to evaluate when synchronizing
        self.layer = Dense(units=128, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var, _ = self.card_gru(inputs)
        var = self.layer(var)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)
        
        return out

    def reset_gru(self):
        self.card_gru.reset_states()

    def get_state(self):
        return self.card_gru.weights[0].value().numpy()
    
    def set_state(self, new_state):
        self.card_gru.weights[0].assign(new_state)
