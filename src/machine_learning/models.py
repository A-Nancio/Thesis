"""Tensorflow model and layer classes"""
import tensorflow as tf
from keras.layers import concatenate, Dense, GRU, GRUCell, Dropout, RNN
import numpy as np

CARD_ID_COLUMN = 0
CATEOGRY_ID_COLUMN = ...
BATCH_SIZE=128

class SharedStateCell(GRUCell):
    def __init__(self, 
                 units, 
                 id_column,
                 activation="tanh", 
                 recurrent_activation="hard_sigmoid", 
                 use_bias=True, 
                 kernel_initializer="glorot_uniform", 
                 recurrent_initializer="orthogonal", 
                 bias_initializer="zeros", 
                 kernel_regularizer=None, 
                 recurrent_regularizer=None, 
                 bias_regularizer=None, 
                 kernel_constraint=None, 
                 recurrent_constraint=None, 
                 bias_constraint=None, 
                 dropout=0, 
                 recurrent_dropout=0, 
                 reset_after=False,
                 **kwargs):
        
        super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, reset_after, **kwargs)
        self.shared_states: tf.Variable = tf.Variable(tf.zeros([1000, units]))
        self.id_column = id_column

    def call(self, inputs, states, training=None):
        """
        tensors should be of size:
        inputs: [batch_size (BATCH_SIZE), num_features (18)]
        states: [batch_size (BATCH_SIZE, num_units (64))]
        """
        ids = tf.dtypes.cast(inputs[:, self.id_column], tf.int32) # shape =

        input_states= tf.gather(self.shared_states, ids)
        output, new_states = super().call(inputs, input_states, training)
        
        self.shared_states.scatter_nd_update(ids, new_states)

        return output, new_states
        # NOTE How do I synchronize with batch entries where 2 entries have the same ID, #NOTE this is fine, tensorflow operations come with synchronization techniques

class TrainSingleState(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = RNN(SharedStateCell(units=32, id_column=CARD_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2'))
        self.dropout = Dropout(0.2)
        self.dense = Dense(32, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.card_gru(inputs)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)

        return out
    

class DoubleStateModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = RNN(SharedStateCell(32, id_column=CARD_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2'))
        self.zip_gru = RNN(SharedStateCell(32, id_column=CATEOGRY_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2'))
        self.dropout = Dropout(0.2)
        self.dense = Dense(32, activation='relu')
        self.out = Dense(1,  activation="sigmoid")
    

    def call(self, inputs, training=None, mask=None):
        var = self.card_cell(inputs)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)

        return out
    
class InferenceSingleState(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_cell = SharedStateCell(32, id_column=CARD_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2')
        self.dropout = Dropout(0.2)
        self.dense = Dense(32, activation='relu')
        self.out = Dense(1,  activation="sigmoid")
    

    def call(self, inputs, training=None, mask=None):
        var = self.card_cell(inputs)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)

        return out
    
class InferenceDoubleState(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_cell = SharedStateCell(32, id_column=CARD_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2')
        self.zip_cell = SharedStateCell(32, id_column=CATEOGRY_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2')
        self.dropout = Dropout(0.2)
        self.dense = Dense(32, activation='relu')
        self.out = Dense(1,  activation="sigmoid")
    

    def call(self, inputs, training=None, mask=None):
        var1 = self.card_cell(inputs)
        var2 = self.zip_cell(inputs)
        var = concatenate(var, axis=0)  # FIXME verify if axis is correct
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)

        return out

