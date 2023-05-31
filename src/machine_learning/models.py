"""Tensorflow model and layer classes"""
import tensorflow as tf
from keras.layers import concatenate, Dense, GRU, GRUCell, Dropout, RNN

CARD_ID_COLUMN = 0
CATEOGRY_ID_COLUMN = ...
BATCH_SIZE=256

class SharedStateTrain(GRUCell):
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
        self.shared_states: tf.Variable = tf.Variable(tf.zeros([1000, BATCH_SIZE, units]))
        self.id_column = id_column
        self.batch_ids = tf.expand_dims(tf.range(BATCH_SIZE), axis=1)

    def call(self, inputs, states, training=None):
        """
        tensors should be of size:
        inputs: [batch_size (BATCH_SIZE), num_features (18)]
        states: [batch_size (BATCH_SIZE, num_units (64))]
        """
        ids = tf.expand_dims(tf.dtypes.cast(inputs[:, self.id_column], tf.int32), axis=1)# shape =
        indices = tf.concat([ids, self.batch_ids], axis=1)

        input_states = tf.gather_nd(self.shared_states, indices)
        output, new_states = super().call(inputs, input_states, training)

        self.shared_states.scatter_nd_update(indices, new_states)
        return output, new_states

    
class SharedStateInference(GRUCell):
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

        input_states = tf.gather(self.shared_states, ids)
        output, new_states = super().call(inputs, input_states, training)
        
        self.shared_states.scatter_nd_update(ids, new_states)
        return output, new_states


class Feedzai(tf.keras.Model):
    def __init__(self, training_mode: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if training_mode:
            self.card_gru = RNN(SharedStateTrain(units=32, id_column=CARD_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2'))
        else:
            self.card_gru = SharedStateInference(units=32, id_column=CARD_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2')

        self.dropout = Dropout(0.2)
        self.dense = Dense(32, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.card_gru(inputs)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)

        return out
    
class DoubleSharedState(tf.keras.Model):
    def __init__(self, training_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if training_mode:
            self.card_gru = RNN(SharedStateTrain(units=32, id_column=CARD_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2'))
            self.category_gru = RNN(SharedStateTrain(units=32, id_column=CATEOGRY_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2'))
        else:
            self.card_gru = SharedStateInference(units=32, id_column=CARD_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2')
            self.category_gru = SharedStateInference(units=32, id_column=CATEOGRY_ID_COLUMN, dropout=0.3, recurrent_regularizer='l2')
    
    def call(self, inputs, training=None, mask=None):
        card_output = self.card_gru(inputs)
        category_output = self.category_gru(inputs)
        var = concatenate(card_output, category_output, axis=0)  # FIXME verify if axis is correct
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)
        return out

