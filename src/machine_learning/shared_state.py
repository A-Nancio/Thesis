"""Shared state classes"""
import tensorflow as tf
from keras.layers import GRUCell

BATCH_SIZE=1024  
 
class SharedState(GRUCell):
    def __init__(self, units, id_column, shared_state_size, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="zeros", kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0, recurrent_dropout=0, reset_after=False,**kwargs):
        super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, reset_after, **kwargs)
        self.shared_state_size = shared_state_size
        self.shared_states: tf.Variable = tf.Variable(tf.zeros([self.shared_state_size, self.units]), name='shared_state')
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
        self.shared_states.assign(tf.zeros([self.shared_state_size, self.units]))


class DistributedSharedState(GRUCell):
    def __init__(self, units, id_column, shared_state_size, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="zeros", kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0, recurrent_dropout=0, reset_after=False,**kwargs):
        super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, reset_after, **kwargs)
        self.shared_state_size = shared_state_size
        self.id_column = id_column
        
        self.shared_states: tf.Variable = tf.Variable(tf.zeros([self.shared_state_size , self.units]), name='shared_state')
        self.deltas: tf.Variable = tf.Variable(tf.zeros([self.shared_state_size, self.units]), name='delta')

    def call(self, inputs, states=None, training=None):
        """
        tensors should be of size:
        inputs: [batch_size (BATCH_SIZE), num_features (18)]
        states: [batch_size (BATCH_SIZE, units)]
        """
            
        
        ids = tf.dtypes.cast(inputs[:, self.id_column], tf.int32)
        
        input_states = tf.gather(self.shared_states, ids)
        old_delta = tf.gather(self.deltas, ids)
        
        #tf.print(inputs[:, self.id_column][0])
        #tf.print(input_states, summarize=-1)
        output, new_states = super().call(inputs, input_states, training)
        new_deltas = tf.add(old_delta, tf.subtract(new_states, input_states))

        self.deltas.scatter_nd_update(tf.expand_dims(ids, axis=1), new_deltas)
        self.shared_states.scatter_nd_update(tf.expand_dims(ids, axis=1), new_states)
        return output, new_states

    def reset_states(self):
        self.shared_states.assign(tf.zeros([self.shared_state_size, self.units]))

    def reset_deltas(self):
        self.deltas.assign(tf.zeros([self.shared_state_size, self.units]))

