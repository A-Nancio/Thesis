import numpy as np
import tensorflow as tf
import pandas as pd

# Create model
class Stateless(tf.keras.Model):
    def __init__(self):
        super(Stateless, self).__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation="relu")
        self.layer2 = tf.keras.layers.Dense(64, activation="relu")
        self.out = tf.keras.layers.Dense(1, activation="softmax")
    
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return self.out(x)

class GlobalState(tf.keras.Model):
    def __init__(self):
        super(GlobalState, self).__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation="relu")
        self.GRU = tf.keras.layers.GRU(128, return_sequences=True, dropout=0.3, recurrent_regularizer='l2',...)
        self.layer2 = tf.keras.layers.Dense(24)
        self.out = tf.keras.layers.Dense(1,  activation="softmax")

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.GRU(x)
        x = self.layer2(inputs)
        return self.out(x)


class SharedState(GlobalState):
    def __init__(self):
        super(SharedState, self).__init__()

    #TODO override training_step to change the gru state of the model

#class SharedGRU(tf.keras.layer)
...


model = Stateless()

dataset = pd.read_csv('../../datasets/modified/modified_sparkov.csv')
print(dataset.dtypes)
labels = dataset['is_fraud']
dataset.drop(['is_fraud', 'cc_num'],axis=1,inplace=True)

model.compile(optimizer='SGD', loss=tf.keras.losses.mean_squared_error, metrics=['accuracy', tf.keras.metrics.Recall()])
model.fit(dataset, labels, validation_split=0.1, epochs=20) 