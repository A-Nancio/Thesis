# Train model for one epoch and save weights
import os

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_test_set, get_class_weights
from machine_learning.models import CARD_ID_COLUMN, CATEOGRY_ID_COLUMN, FeedzaiTrainAsync, FeedzaiProduction, BATCH_SIZE, SharedStateAsync
from data_processing.batch_generator import load_train_set, load_test_set, load_pre_data, get_class_weights
from machine_learning.models import DoubleStateTrainAsync, DoubleStateTrainSync, DoubleStateProduction, FeedzaiTrainAsync, FeedzaiTrainSync, FeedzaiProduction, BATCH_SIZE
from machine_learning.pipeline import compile_model
from keras.layers import concatenate, Dense, GRUCell, Dropout, Layer, RNN, GRU
import tensorflow as tf
import tensorflow as tf
import numpy as np
from keras import metrics


SEQUENCE_LENGTH = 100


class SimpleDouble(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=128, dropout=0.1, recurrent_dropout=0.2)
        self.category_gru = RNN(SharedStateAsync(units=128, dropout=0.1, recurrent_dropout=0.2, id_column=CATEOGRY_ID_COLUMN))
        self.layer = Dense(units=128, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense = Dense(64, activation='relu')
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_output = self.card_gru(inputs)
        category_output = self.category_gru(inputs)
        var = concatenate([card_output, category_output])
        var = self.layer(var)
        var = self.dropout(var)
        var = self.dense(var)
        out = self.out(var)
        return out

    def reset_gru(self):
        self.card_gru.cell.reset_states()   # training model needs to access cell
        self.category_gru.cell.reset_states()   # training model needs to access cell
    

train_model = SimpleDouble()

prototype = DoubleStateProduction()



transactions = np.concatenate((np.load(f'src/data/train/transactions.npy'), np.load(f'src/data/test/transactions.npy')), axis=0)
                              
# BATCHES REDUCED TO 10 FOR SIMPLICITY
train_seq_ids = np.load(f'src/data/train/seq_ids.npy').astype(int)#[0:10]
train_seq_labels = np.load(f'src/data/train/seq_labels.npy').astype(int)#[0:10]
train_set = tf.data.Dataset.from_tensor_slices(
    (train_seq_ids, 
     train_seq_labels)
).batch(4096)

sample_transaction = np.load(f'src/data/test/transactions.npy')[0]
# test_transactions = np.load(f'src/data/test/transactions.npy')
# test_seq_ids = np.load(f'src/data/test/seq_ids.npy').astype(int)#[0:10]
# test_seq_labels = np.load(f'src/data/test/seq_labels.npy').astype(int)#[0:10]
# test_set = tf.data.Dataset.from_tensor_slices(
#     (test_seq_ids, test_seq_labels)).batch(1)


single_trans_set = load_test_set()


prototype.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])

prototype(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model



print(f"----------------- {train_model.name} -----------------")
train_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[metrics.BinaryAccuracy(),
            metrics.TruePositives(), metrics.TrueNegatives(),
            metrics.FalsePositives(), metrics.FalseNegatives()])


for epoch in range(10):
    print(f"[EPOCH {epoch}]")
    train_model.reset_metrics()
    with tf.device("/gpu:0"):
        for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_set), total=train_set.cardinality().numpy()):
            x_batch_train = transactions[x_batch_train]

            with tf.GradientTape() as tape:
                results = train_model.train_on_batch(
                    x_batch_train,
                    y_batch_train,
                    reset_metrics=False,
                    return_dict=True
                )
        print(results)
    
    # Assign weights to the stateful layers
    prototype.card_gru.weights[1].assign(train_model.card_gru.weights[0])
    prototype.card_gru.weights[2].assign(train_model.card_gru.weights[1])
    prototype.card_gru.weights[3].assign(train_model.card_gru.weights[2][0])

    prototype.category_gru.weights[1].assign(train_model.category_gru.weights[1])
    prototype.category_gru.weights[2].assign(train_model.category_gru.weights[2])
    prototype.category_gru.weights[3].assign(train_model.category_gru.weights[3])

    prototype.layer.set_weights(train_model.layer.get_weights())
    prototype.dense.set_weights(train_model.dense.get_weights())
    prototype.out.set_weights(train_model.out.get_weights())

    prototype.evaluate(single_trans_set)
 
    weight_path = f'src/machine_learning/saved_models/Double_{epoch}.keras'
    prototype.save_weights(
        filepath=weight_path,
        save_format='h5'
    )
