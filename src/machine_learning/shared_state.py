"""Shared state classes"""
import tensorflow as tf
from keras.layers import concatenate, Dense, GRUCell, Dropout, Layer, RNN, GRU
from keras.engine import base_layer_utils
from keras import backend
import random 

from distribution.db_utils import from_redis, to_redis, add_deltas_to_redis

BATCH_SIZE=1024
 
class SharedState(GRUCell):
    def __init__(self, units, id_column, activation="tanh", recurrent_activation="hard_sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="zeros", kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0, recurrent_dropout=0, reset_after=False,**kwargs):
        super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, reset_after, **kwargs)
        self.shared_states: tf.Variable = tf.Variable(tf.zeros([1000, self.units]), name='shared_state')
        self.id_column = id_column

    def call(self, inputs, states=None, training=None):
        """
        tensors should be of size:
        inputs: [batch_size (BATCH_SIZE), num_features (18)]
        states: [batch_size (BATCH_SIZE, units)]
        """
        ids = tf.dtypes.cast(inputs[:, self.id_column], tf.int32)
        input_states = tf.gather(self.shared_states, ids)

        output, new_states = super().call(inputs, input_states, training)

        self.shared_states.scatter_nd_update(tf.expand_dims(ids, axis=1), new_states)

        return output, new_states

    def reset_states(self):
        self.shared_states.assign(tf.zeros([1000, self.units]))


class AsynchronousSharedState(GRUCell):
    def __init__(self, units, id_column, activation="tanh", recurrent_activation="hard_sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="zeros", kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0, recurrent_dropout=0, reset_after=False,**kwargs):
        super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, reset_after, **kwargs)
        self.shared_states: tf.Variable = tf.Variable(tf.zeros([1000, self.units]), name='shared_state')
        self.id_column = id_column
        

    def call(self, inputs, states=None, training=None):
        """
        tensors should be of size:
        inputs: [batch_size (BATCH_SIZE), transaction_id + num_features (18)]
        states: [batch_size (BATCH_SIZE, num_units)]
        """
        ids = tf.dtypes.cast(inputs[:, self.id_column], tf.int32)   # in production ids = [id], i.e. numpy array with a single value
        
        # ------------ UPDATE STATE -------------------
        fetch_state = random.random()
        #if fetch_state > 0.80:
        input_states = tf.convert_to_tensor(from_redis(str(ids.numpy()[0])), dtype=tf.float32)
        #else:
            #input_states = tf.gather(self.shared_states, ids)


        # ------------ FORWARD PASS -------------------
        output, new_states = super().call(inputs, input_states, training)
        self.shared_states.scatter_nd_update(tf.expand_dims(ids, axis=1), new_states)


        # ------------ UPDATE DELTAS & TIMESTAMPS -------------------
        add_deltas_to_redis(str(ids.numpy()[0]), tf.subtract(new_states, input_states).numpy())

        return output, new_states


    def reset_states(self):
        self.shared_states.assign(tf.zeros([1000, self.units]))


# NOTE DEPPRECATED, not tested
# class SharedStateSync(GRUCell):
#     def __init__(self, units, id_column,activation="tanh", recurrent_activation="hard_sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="zeros", kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0, recurrent_dropout=0, reset_after=False,**kwargs):
#         super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, reset_after, **kwargs)
#         self.sync_states: tf.Variable = tf.Variable(tf.zeros([1000, BATCH_SIZE, units]))
#         self.id_column = id_column
#         self.batch_ids = tf.expand_dims(tf.range(BATCH_SIZE), axis=1)
#         self.num_units: int = units
# 
#     def call(self, inputs, states=None, training=None):
#         """
#         tensors should be of size:
#         inputs: [batch_size (BATCH_SIZE), num_features (18)]
#         states: [batch_size (BATCH_SIZE), num_units)]
#         """
#         ids = tf.expand_dims(tf.dtypes.cast(inputs[:, self.id_column], tf.int32), axis=1)# shape =
#         indices = tf.concat([ids, self.batch_ids], axis=1)
# 
#         input_states = tf.gather_nd(self.sync_states, indices)
#         # print(input_states.shape)
#         output, new_states = super().call(inputs, input_states, training)
# 
#         self.sync_states.scatter_nd_update(indices, new_states)
#         return output, new_states
# 
#     def reset_states(self):
#         self.sync_states.assign(tf.zeros([1000, BATCH_SIZE, self.num_units]))