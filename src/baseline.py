# Train model for one epoch and save weights
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_test_set, get_class_weights
from machine_learning.models import BATCH_SIZE
from keras.layers import concatenate, Dense, GRUCell, Dropout, Layer, RNN, GRU

import tensorflow as tf
import numpy as np
from keras import metrics
SEQUENCE_LENGTH = 100


class Stateless(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = Dense(units=128, activation='relu')    #NOTE change for sync to evaluate when synchronizing
        self.layer2 = Dense(units=128, activation='relu')
        self.dropout = Dropout(0.2)
        self.layer3 = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.layer1(inputs)
        var = self.layer2(var)
        var = self.dropout(var)
        var = self.layer3(var)
        out = self.out(var)
        return out
    
model = Stateless()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])

path = 'src/data'

transactions = np.load(f'{path}/train/transactions.npy')
labels = np.load(f'{path}/train/all_transaction_labels.npy')
train_set = tf.data.Dataset.from_tensor_slices((transactions, labels)).batch(1024)


dataset = np.load(f'{path}/test/transactions.npy')
labels = np.load(f'{path}/test/all_transaction_labels.npy')
test_set = tf.data.Dataset.from_tensor_slices((dataset, labels)).batch(1024)

with tf.device("/gpu:0"):
    
    model.fit(train_set,
                epochs=20, 
                verbose='auto', 
                shuffle=True)
    
    model.evaluate(test_set)

  