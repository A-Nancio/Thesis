# Train model for one epoch and save weights
import os

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from machine_learning.shared_state import BATCH_SIZE
from keras.layers import Dense, GRU
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

SEQUENCE_LENGTH = 100
NUM_EPOCHS = 18 # NOTE dependant on best performing epoch in K_fold resutls


class GlobalState(tf.keras.Model):
    def __init__(self, stateful, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.gru = GRU(units=128, stateful=stateful)
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.gru(inputs)
        out = self.out(var)
        return out
    

transactions = np.concatenate((np.load(f'src/data/train/transactions.npy'), np.load(f'src/data/test/transactions.npy')), axis=0)
                              
train_seq_ids = shuffle(np.load(f'src/data/train/seq_ids.npy').astype(int), random_state=42)
train_seq_labels = shuffle(np.load(f'src/data/train/seq_labels.npy').astype(int), random_state=42)
train_set = tf.data.Dataset.from_tensor_slices(
    (train_seq_ids, 
     train_seq_labels)
).batch(BATCH_SIZE)

train_model = GlobalState(stateful=False)

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

for epoch in range(NUM_EPOCHS):
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

weight_path = f'src/machine_learning/saved_models/feedzai.keras'
train_model.save_weights(
    filepath=weight_path,
    save_format='h5'
)


