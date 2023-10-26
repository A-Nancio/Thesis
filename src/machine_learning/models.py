
"""Tensorflow model and layer classes"""
import tensorflow as tf
from keras.layers import concatenate, Dense, GRU, RNN

from machine_learning.shared_state import SharedState, DistributedSharedState

CARD_ID_COLUMN = 0
CATEGORY_ID_COLUMN = 2

# ------------------------ FEEDZAI ------------------------
class FeedzaiTrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=48, return_state=True)  
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        out = self.out(card_out)
        return out

class FeedzaiProduction(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=48, id_column=CARD_ID_COLUMN, shared_state_size=1000) 
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        out = self.out(card_out)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()

    def set_weights(self, model: tf.keras.Model):    
        self.card_gru.weights[1].assign(model.card_gru.weights[0])
        self.card_gru.weights[2].assign(model.card_gru.weights[1])
        self.card_gru.weights[3].assign(model.card_gru.weights[2][0])

        self.out.set_weights(model.out.get_weights())


# ------------------------ FEEDZAI EXTRA LAYERS ------------------------
class FeedzaiExtraTrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=48, return_state=True) 
        self.dense = Dense(24, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        var = self.dense(card_out)
        out = self.out(var)
        return out
    
class FeedzaiExtraProduction(tf.keras.Model): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=48, id_column=CARD_ID_COLUMN, shared_state_size=1000)  
        self.dense = Dense(24, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        var = self.dense(card_out)
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()

    def set_weights(self, model: tf.keras.Model):    
        self.card_gru.weights[1].assign(model.card_gru.weights[0])
        self.card_gru.weights[2].assign(model.card_gru.weights[1])
        self.card_gru.weights[3].assign(model.card_gru.weights[2][0])

        self.dense.set_weights(model.dense.get_weights())
        self.out.set_weights(model.out.get_weights())


# ------------------------ FEEDZAI WITH CONCAT ------------------------
class FeedzaiConcatTrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=48, return_state=True)  
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        var = concatenate([inputs[:, -1, :], card_out])
        out = self.out(var)
        return out

class FeedzaiConcatProduction(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=48, id_column=CARD_ID_COLUMN, shared_state_size=1000)  
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        var = concatenate([inputs, card_out])
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()

    def set_weights(self, model: tf.keras.Model):    
        self.card_gru.weights[1].assign(model.card_gru.weights[0])
        self.card_gru.weights[2].assign(model.card_gru.weights[1])
        self.card_gru.weights[3].assign(model.card_gru.weights[2][0])

        self.out.set_weights(model.out.get_weights())



# ------------------------ FEEDZAI EXTRA LAYERS WITH CONCAT ------------------------
class FeedzaiExtraConcatTrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=48, return_state=True) 
        self.dense = Dense(24, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        var = concatenate([inputs[:, -1, :], card_out])
        var = self.dense(var)
        out = self.out(var)
        return out
    
class FeedzaiExtraConcatProduction(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=48, id_column=CARD_ID_COLUMN, shared_state_size=1000)
        self.dense = Dense(24, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        var = concatenate([inputs, card_out])
        var = self.dense(var)
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()

    def set_weights(self, model: tf.keras.Model):    
        self.card_gru.weights[1].assign(model.card_gru.weights[0])
        self.card_gru.weights[2].assign(model.card_gru.weights[1])
        self.card_gru.weights[3].assign(model.card_gru.weights[2][0])

        self.dense.set_weights(model.dense.get_weights())
        self.out.set_weights(model.out.get_weights())













# ------------------------ DOUBLE ------------------------
class DoubleTrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=48, return_state=True)
        self.category_gru = RNN(SharedState(units=48, id_column=CATEGORY_ID_COLUMN, shared_state_size=15), return_state=True)  
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        category_out, _ = self.category_gru(inputs)
        
        var = concatenate([card_out, category_out])
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.category_gru.cell.reset_states()

class DoubleProduction(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=48, id_column=CARD_ID_COLUMN, shared_state_size=1000) 
        self.category_gru = DistributedSharedState(units=48, id_column=CATEGORY_ID_COLUMN, shared_state_size=15) 
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        category_out, _ = self.category_gru(inputs)
        var = concatenate([card_out, category_out])
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()
        self.category_gru.reset_states()

    def set_weights(self, model: tf.keras.Model):    
        self.card_gru.weights[1].assign(model.card_gru.weights[0])
        self.card_gru.weights[2].assign(model.card_gru.weights[1])
        self.card_gru.weights[3].assign(model.card_gru.weights[2][0])

        self.category_gru.weights[2].assign(model.category_gru.weights[1])
        self.category_gru.weights[3].assign(model.category_gru.weights[2])
        self.category_gru.weights[4].assign(model.category_gru.weights[3])

        self.out.set_weights(model.out.get_weights())


# ------------------------ DOUBLE EXTRA LAYERS ------------------------
class DoubleExtraTrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=48, return_state=True)
        self.category_gru = RNN(SharedState(units=48, id_column=CATEGORY_ID_COLUMN, shared_state_size=15), return_state=True) 
        self.dense = Dense(24, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        category_out, _ = self.category_gru(inputs)
        var = concatenate([card_out, category_out])
        var = self.dense(var)
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.category_gru.cell.reset_states()
    
class DoubleExtraProduction(tf.keras.Model): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=48, id_column=CARD_ID_COLUMN, shared_state_size=1000) 
        self.category_gru = DistributedSharedState(units=48, id_column=CATEGORY_ID_COLUMN, shared_state_size=15) 
        self.dense = Dense(24, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        category_out, _ = self.category_gru(inputs)
        var = concatenate([card_out, category_out])
        var = self.dense(var)
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()
        self.category_gru.reset_states()

    def set_weights(self, model: tf.keras.Model):    
        self.card_gru.weights[1].assign(model.card_gru.weights[0])
        self.card_gru.weights[2].assign(model.card_gru.weights[1])
        self.card_gru.weights[3].assign(model.card_gru.weights[2][0])

        self.category_gru.weights[2].assign(model.category_gru.weights[1])
        self.category_gru.weights[3].assign(model.category_gru.weights[2])
        self.category_gru.weights[4].assign(model.category_gru.weights[3])

        self.dense.set_weights(model.dense.get_weights())
        self.out.set_weights(model.out.get_weights())


# ------------------------ DOUBLE WITH CONCAT ------------------------
class DoubleConcatTrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=48, return_state=True)
        self.category_gru = RNN(SharedState(units=48, id_column=CATEGORY_ID_COLUMN, shared_state_size=15), return_state=True)  
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        category_out, _ = self.category_gru(inputs)
        var = concatenate([inputs[:, -1, :], card_out, category_out])
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.category_gru.cell.reset_states()

class DoubleConcatProduction(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=48, id_column=CARD_ID_COLUMN, shared_state_size=1000) 
        self.category_gru = DistributedSharedState(units=48, id_column=CATEGORY_ID_COLUMN, shared_state_size=15) 
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        category_out, _ = self.category_gru(inputs)
        var = concatenate([inputs, card_out, category_out])
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()
        self.category_gru.reset_states()

    def set_weights(self, model: tf.keras.Model):    
        self.card_gru.weights[1].assign(model.card_gru.weights[0])
        self.card_gru.weights[2].assign(model.card_gru.weights[1])
        self.card_gru.weights[3].assign(model.card_gru.weights[2][0])

        self.category_gru.weights[2].assign(model.category_gru.weights[1])
        self.category_gru.weights[3].assign(model.category_gru.weights[2])
        self.category_gru.weights[4].assign(model.category_gru.weights[3])

        self.out.set_weights(model.out.get_weights())



# ------------------------ DOUBLE EXTRA LAYERS WITH CONCAT ------------------------
class DoubleExtraConcatTrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=48, return_state=True)
        self.category_gru = RNN(SharedState(units=48, id_column=CATEGORY_ID_COLUMN, shared_state_size=15), return_state=True) 
        self.dense = Dense(24, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        category_out, _ = self.category_gru(inputs)
        var = concatenate([inputs[:, -1, :], card_out, category_out])
        var = self.dense(var)
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.category_gru.cell.reset_states()

class DoubleExtraConcatProduction(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = SharedState(units=48, id_column=CARD_ID_COLUMN, shared_state_size=1000)
        self.category_gru = DistributedSharedState(units=48, id_column=CATEGORY_ID_COLUMN, shared_state_size=15) 
        self.dense = Dense(24, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_out, _ = self.card_gru(inputs)
        category_out, _ = self.category_gru(inputs)
        var = concatenate([inputs, card_out, category_out])
        var = self.dense(var)
        out = self.out(var)
        return out
    
    def reset_gru(self):
        self.card_gru.reset_states()
        self.category_gru.reset_states()

    def set_weights(self, model: tf.keras.Model):    
        self.card_gru.weights[1].assign(model.card_gru.weights[0])
        self.card_gru.weights[2].assign(model.card_gru.weights[1])
        self.card_gru.weights[3].assign(model.card_gru.weights[2][0])

        self.category_gru.weights[2].assign(model.category_gru.weights[1])
        self.category_gru.weights[3].assign(model.category_gru.weights[2])
        self.category_gru.weights[4].assign(model.category_gru.weights[3])

        self.dense.set_weights(model.dense.get_weights())
        self.out.set_weights(model.out.get_weights())
