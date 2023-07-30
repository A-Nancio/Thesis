import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_test_set, load_pre_data, load_train_set
from machine_learning.models import FeedzaiProduction, SimpleDouble, DoubleStateProduction, BATCH_SIZE
from machine_learning.pipeline import compile_model
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras import metrics
import time
from tqdm import tqdm
from keras.layers import concatenate, Dense, GRUCell, Dropout, Layer, RNN, GRU
import sys

tf.random.set_seed(42)


class GlobalState(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gru = GRU(units=128, dropout=0.2, recurrent_dropout=0.1)
        self.layer2 = Dense(units=128, activation='relu')
        self.dropout = Dropout(0.2)
        self.layer3 = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.gru(inputs)
        var = self.layer2(var)
        var = self.dropout(var)
        var = self.layer3(var)
        out = self.out(var)
        return out

class GlobalStateProduction(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gru = GRU(units=128, stateful=True)
        self.layer2 = Dense(units=128, activation='relu')
        self.dropout = Dropout(0.2)
        self.layer3 = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.gru(inputs)
        var = self.layer2(var)
        var = self.dropout(var)
        var = self.layer3(var)
        out = self.out(var)
        return out



train_model =  GlobalState()
production_model = GlobalStateProduction()

# --------------------
#transactions = np.load(f'src/data/train/transactions.npy')

#train_set = tf.data.Dataset.from_tensor_slices(
#    (np.load(f'src/data/seq_ids.npy').astype(int), np.load(f'src/data/seq_labels.npy').astype(int))).batch(BATCH_SIZE)

train_set = load_train_set(BATCH_SIZE)

dataset = np.load(f'src/data/test/transactions.npy')
dataset = np.expand_dims(dataset, axis=1)
print(dataset.shape)
labels = np.load(f'src/data/test/all_transaction_labels.npy')
test_set = tf.data.Dataset.from_tensor_slices((dataset, labels)).batch(1)

# --------------------

print(f"----------------- {train_model.name} -----------------")
with tf.device("/gpu:0"):
    train_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])
    
    
    for epoch in range(10):
        print(f"[EPOCH {epoch}]")
        train_set.shuffle(1000)
        train_model.reset_metrics()
        for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_set), total=train_set.cardinality().numpy()):
            # x_batch_train = transactions[x_batch_train]

            with tf.GradientTape() as tape:
                results = train_model.train_on_batch(
                    x_batch_train,
                    y_batch_train,
                    reset_metrics=False,
                    return_dict=True
                )
        print(results)

        
    with tf.device("cpu"):
        compile_model(production_model, test_set)
        weight_path = f'src/machine_learning/saved_models/{train_model.name}'
        train_model.save_weights(
            filepath=weight_path,
            save_format='h5'
        )
        production_model.load_weights(weight_path, by_name=True)
        production_model.reset_states()

        production_model.evaluate(test_set)
