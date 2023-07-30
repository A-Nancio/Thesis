# Train model for one epoch and save weights
import os

import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_test_set, get_class_weights
from machine_learning.models import FeedzaiTrainAsync, FeedzaiProduction, BATCH_SIZE
from data_processing.batch_generator import load_train_set, load_test_set, load_pre_data, get_class_weights
from machine_learning.models import DoubleStateTrainAsync, DoubleStateTrainSync, DoubleStateProduction, FeedzaiTrainAsync, FeedzaiTrainSync, FeedzaiProduction, BATCH_SIZE
from machine_learning.pipeline import compile_model
from keras.layers import concatenate, Dense, GRUCell, Dropout, Layer, RNN, GRU
import tensorflow as tf
import tensorflow as tf
import numpy as np
from keras import metrics


SEQUENCE_LENGTH = 100


# IT Trains only on sequences which have only on credit card in the sequence
class SimpleFeedzai(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=128)  
        self.layer = Dense(units=128, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.card_gru(inputs)
        var = self.layer(var)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)

        return out
    
model = SimpleFeedzai()

transactions = np.load(f'src/data/train/transactions.npy')
train_set = tf.data.Dataset.from_tensor_slices(
    (np.load(f'src/data/seq_ids.npy').astype(int), np.load(f'src/data/seq_labels.npy').astype(int))).batch(BATCH_SIZE)

# train_set = load_train_set(BATCH_SIZE)
# pre_test_data = load_pre_data()
test_set = load_test_set()
test_set = tf.data.Dataset.from_tensor_slices(
    (np.load(f'src/data/test/seq_ids.npy'), np.load(f'src/data/test/seq_labels.npy').astype(int))).batch(BATCH_SIZE)
# --------------------

print(f"----------------- {model.name} -----------------")
with tf.device("/gpu:0"):
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])
    
    for epoch in range(20):
        print(f"[EPOCH {epoch}]")
        train_set.shuffle(1000)
        model.reset_metrics()
        for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_set), total=train_set.cardinality().numpy()):
            x_batch_train = transactions[x_batch_train]

            with tf.GradientTape() as tape:
                results = model.train_on_batch(
                    x_batch_train,
                    y_batch_train,
                    reset_metrics=False,
                    return_dict=True
                )
        print(results)

    model.reset_metrics()
    for step, (x_batch_test, y_batch_test) in tqdm(enumerate(train_set), total=train_set.cardinality().numpy()):
        x_batch_test = transactions[x_batch_test]

        with tf.GradientTape() as tape:
            results = model.test_on_batch(
                x_batch_test,
                y_batch_test,
                reset_metrics=False,
                return_dict=True
            )
        print(results)