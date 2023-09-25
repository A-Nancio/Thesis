# Train model for one epoch and save weights
import os

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_test_set
from machine_learning.models import FeedzaiProduction
from machine_learning.shared_state import BATCH_SIZE
from keras.layers import Dense, Dropout, GRU
import tensorflow as tf
import numpy as np


SEQUENCE_LENGTH = 100


# IT Trains only on sequences which have only on credit card in the sequence
class Feedzaitrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=128)  
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.card_gru(inputs)
        out = self.out(var)

        return out
    

train_model = Feedzaitrain()

prototype = FeedzaiProduction()



transactions = np.concatenate((np.load(f'src/data/train/transactions.npy'), np.load(f'src/data/test/transactions.npy')), axis=0)
                              
# BATCHES REDUCED TO 10 FOR SIMPLICITY
train_seq_ids = np.load(f'src/data/train/seq_ids.npy').astype(int)#[0:10]
train_seq_labels = np.load(f'src/data/train/seq_labels.npy').astype(int)#[0:10]
train_set = tf.data.Dataset.from_tensor_slices(
    (train_seq_ids, 
     train_seq_labels)
).batch(BATCH_SIZE)

sample_transaction = np.load(f'src/data/test/transactions.npy')[0]
# test_transactions = np.load(f'src/data/test/transactions.npy')
# test_seq_ids = np.load(f'src/data/test/seq_ids.npy').astype(int)#[0:10]
# test_seq_labels = np.load(f'src/data/test/seq_labels.npy').astype(int)#[0:10]
# test_set = tf.data.Dataset.from_tensor_slices(
#     (test_seq_ids, test_seq_labels)).batch(1)


single_trans_set = load_test_set()


prototype.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.TruePositives(), 
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(), 
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()])

prototype(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model


print(f"----------------- {train_model.name} -----------------")
train_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.TruePositives(), 
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(), 
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()])


for epoch in range(10):
    print(f"[EPOCH {epoch}]")
    train_model.reset_metrics()
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
    prototype.card_gru.reset_states()
    # Assign weights to the stateful layers
    prototype.card_gru.weights[1].assign(train_model.card_gru.weights[0])
    prototype.card_gru.weights[2].assign(train_model.card_gru.weights[1])
    prototype.card_gru.weights[3].assign(train_model.card_gru.weights[2][0])

    prototype.layer.set_weights(train_model.layer.get_weights())
    prototype.dense.set_weights(train_model.dense.get_weights())
    prototype.out.set_weights(train_model.out.get_weights())

    

    # results = prototype.evaluate(single_trans_set)

    weight_path = f'src/machine_learning/saved_models/feedzai/Feedzai_{epoch}.keras'
    prototype.save_weights(
        filepath=weight_path,
        save_format='h5'
    )
