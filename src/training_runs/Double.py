# Train model for one epoch and save weights
import os

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from machine_learning.models import CATEOGRY_ID_COLUMN
from machine_learning.shared_state import SharedState, BATCH_SIZE
from machine_learning.models import DistributedProduction
from keras.layers import concatenate, Dense, RNN, GRU
import tensorflow as tf
import tensorflow as tf
import numpy as np
from keras import metrics
from sklearn.utils import shuffle

SEQUENCE_LENGTH = 100
NUM_EPOCHS = 12

class DoubleTrain(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=128, dropout=0.1, recurrent_dropout=0.2)
        self.category_gru = RNN(SharedState(units=128, dropout=0.1, recurrent_dropout=0.2, id_column=CATEOGRY_ID_COLUMN))
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_output = self.card_gru(inputs)
        category_output = self.category_gru(inputs)
        var = concatenate([card_output, category_output])
        out = self.out(var)
        return out

    def reset_gru(self):
        self.card_gru.cell.reset_states()   # training model needs to access cell
        self.category_gru.cell.reset_states()   # training model needs to access cell
    

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
train_model = DoubleTrain()
test_model = DistributedProduction()
train_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[metrics.BinaryAccuracy(),
            metrics.TruePositives(), metrics.TrueNegatives(),
            metrics.FalsePositives(), metrics.FalseNegatives()])
test_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])
test_model(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model


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
    
# Assign weights to the stateful layers
train_model.card_gru.reset_states()
train_model.category_gru.reset_states()
test_model.card_gru.reset_states()
train_model.category_gru.reset_states()

test_model.card_gru.weights[1].assign(train_model.card_gru.weights[1])
test_model.card_gru.weights[2].assign(train_model.card_gru.weights[2])
test_model.card_gru.weights[3].assign(train_model.card_gru.weights[3])

test_model.category_gru.weights[2].assign(train_model.category_gru.weights[1])
test_model.category_gru.weights[3].assign(train_model.category_gru.weights[2])
test_model.category_gru.weights[4].assign(train_model.category_gru.weights[3])

weight_path = f'src/machine_learning/saved_models/double.keras'
test_model.save_weights(
    filepath=weight_path,
    save_format='h5'
)
