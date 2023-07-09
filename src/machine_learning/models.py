"""Tensorflow model and layer classes"""
import tensorflow as tf
from keras.layers import concatenate, Dense, GRUCell, Dropout, Layer, RNN
from keras.engine import base_layer_utils
from keras import backend

CARD_ID_COLUMN = 0
CATEOGRY_ID_COLUMN = 1
BATCH_SIZE=1024

class SharedStateSync(GRUCell):
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
        self.sync_states: tf.Variable = tf.Variable(tf.zeros([1000, BATCH_SIZE, units]))
        self.id_column = id_column
        self.batch_ids = tf.expand_dims(tf.range(BATCH_SIZE), axis=1)
        self.num_units: int = units

    def call(self, inputs, states=None, training=None):
        """
        tensors should be of size:
        inputs: [batch_size (BATCH_SIZE), num_features (18)]
        states: [batch_size (BATCH_SIZE, num_units (64))]
        """
        ids = tf.expand_dims(tf.dtypes.cast(inputs[:, self.id_column], tf.int32), axis=1)# shape =
        indices = tf.concat([ids, self.batch_ids], axis=1)

        input_states = tf.gather_nd(self.sync_states, indices)
        # print(input_states.shape)
        output, new_states = super().call(inputs, input_states, training)

        self.sync_states.scatter_nd_update(indices, new_states)
        return output, new_states

    def reset_states(self):
        self.sync_states.assign(tf.zeros([1000, BATCH_SIZE, self.num_units]))
 
class SharedStateAsync(GRUCell):
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
        self.num_units: int = units
        self.shared_states: tf.Variable = tf.Variable(tf.zeros([1000, self.num_units]))
        self.id_column = id_column

    def call(self, inputs, states=None, training=None):
        """
        tensors should be of size:
        inputs: [batch_size (BATCH_SIZE), num_features (18)]
        states: [batch_size (BATCH_SIZE, num_units (64))]
        """
        ids = tf.dtypes.cast(inputs[:, self.id_column], tf.int32)

        input_states = tf.gather(self.shared_states, ids)
        output, new_states = super().call(inputs, input_states, training)
        
        self.shared_states.scatter_nd_update(ids, new_states)
        return output, new_states


    def reset_states(self):
        self.shared_states.assign(tf.zeros([1000, self.num_units]))
#
# ------- FEEDZAI'S MODEL -------
#
class FeedzaiTrainSync(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = RNN(SharedStateSync(units=128, id_column=CARD_ID_COLUMN))    #NOTE change for sync to evaluate
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.card_gru(inputs)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)

        return out
    
    def reset_gru(self):
        self.card_gru.cell.reset_states()   # training model needs to access cell

class FeedzaiTrainAsync(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = RNN(SharedStateAsync(units=128, id_column=CARD_ID_COLUMN))    #NOTE change for sync to evaluate
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.card_gru(inputs)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)

        return out
    
    def reset_gru(self):
        self.card_gru.cell.reset_states()   # training model needs to access cell

class FeedzaiProduction(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedStateAsync(units=128, id_column=CARD_ID_COLUMN)    #NOTE change for sync to evaluate when synchronizing
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var, _ = self.card_gru(inputs)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)

        return out

    def reset_gru(self):
        self.card_gru.reset_states()



#
#   ------- DOUBLE SHARED STATE MODELS -------
#
class DoubleStateTrainSync(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = RNN(SharedStateSync(units=128, id_column=CARD_ID_COLUMN))    #NOTE change for sync to evaluate when synchronizing
        self.category_gru = RNN(SharedStateSync(units=128, id_column=CARD_ID_COLUMN))
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_output = self.card_gru(inputs)
        category_output = self.category_gru(inputs)
        var = concatenate([card_output, category_output])
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)
        return out

    def reset_gru(self):
        self.card_gru.cell.reset_states()   # training model needs to access cell
        self.category_gru.cell.reset_states()   # training model needs to access cell

class DoubleStateTrainAsync(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = RNN(SharedStateAsync(units=128, id_column=CARD_ID_COLUMN))    #NOTE change for sync to evaluate when synchronizing
        self.category_gru = RNN(SharedStateAsync(units=128, id_column=CARD_ID_COLUMN))
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_output = self.card_gru(inputs)
        category_output = self.category_gru(inputs)
        var = concatenate([card_output, category_output])
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)
        return out

    def reset_gru(self):
        self.card_gru.cell.reset_states()   # training model needs to access cell
        self.category_gru.cell.reset_states()   # training model needs to access cell
    

class DoubleStateProduction(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedStateAsync(units=128, id_column=CARD_ID_COLUMN)    #NOTE change for sync to evaluate when synchronizing
        self.category_gru = SharedStateAsync(units=128, id_column=CARD_ID_COLUMN)
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_output, _ = self.card_gru(inputs)
        category_output, _ = self.category_gru(inputs)
        var = concatenate([card_output, category_output])
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()
        self.category_gru.reset_states()