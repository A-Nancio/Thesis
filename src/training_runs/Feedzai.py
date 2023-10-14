# Train model for one epoch and save weights
import os

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from machine_learning.models import FeedzaiProduction
from machine_learning.shared_state import BATCH_SIZE
from keras.layers import Dense, GRU
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

SEQUENCE_LENGTH = 100
NUM_EPOCHS = 12

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
    

# DATA
transactions = np.concatenate((np.load(f'src/data/train/transactions.npy'), np.load(f'src/data/test/transactions.npy')), axis=0)
train_seq_ids = shuffle(np.load(f'src/data/train/seq_ids.npy').astype(int), random_state=42)
train_seq_labels = shuffle(np.load(f'src/data/train/seq_labels.npy').astype(int), random_state=42)
train_set = tf.data.Dataset.from_tensor_slices(
    (train_seq_ids, 
     train_seq_labels)
).batch(BATCH_SIZE)
sample_transaction = np.load(f'src/data/test/transactions.npy')[0]

# MODEL
train_model = Feedzaitrain()
test_model = FeedzaiProduction()
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
test_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.TruePositives(), 
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(), 
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()])
test_model(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model


for epoch in range(NUM_EPOCHS):
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

test_model.card_gru.reset_states()

# Assign weights to the stateful layers
test_model.card_gru.weights[1].assign(train_model.card_gru.weights[0])
test_model.card_gru.weights[2].assign(train_model.card_gru.weights[1])
test_model.card_gru.weights[3].assign(train_model.card_gru.weights[2][0])

test_model.layer.set_weights(train_model.layer.get_weights())
test_model.dense.set_weights(train_model.dense.get_weights())
test_model.out.set_weights(train_model.out.get_weights())

weight_path = f'src/machine_learning/saved_models/feedzai.keras'
test_model.save_weights(
    filepath=weight_path,
    save_format='h5'
)
