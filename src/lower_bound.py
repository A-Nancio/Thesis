# Train model for one epoch and save weights
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tqdm import tqdm

from machine_learning.models import DoubleProduction, DoubleExtraProduction, DoubleConcatProduction, DoubleExtraConcatProduction
from machine_learning.shared_state import BATCH_SIZE
import tensorflow as tf
import numpy as np
import csv

sample_transaction = np.load(f'src/data/test/transactions.npy')[0]
test_set = tf.data.Dataset.from_tensor_slices(
    (np.load(f'src/data/test/transactions.npy'), 
     np.load(f'src/data/test/all_transaction_labels.npy').astype(float))
).batch(1)

model_list = [DoubleProduction, DoubleExtraProduction, DoubleConcatProduction, DoubleExtraConcatProduction]

class randomizer(tf.keras.callbacks.Callback):
    def on_test_batch_end(self, batch, logs=None):
        self.model.category_gru.shared_states.assign(
            tf.random.uniform(
                self.model.category_gru.shared_states.shape,
                minval=-1,
                maxval=1,
                seed=42
            ))

callback = randomizer()

for model_class in model_list:
    tf.keras.backend.clear_session()

    model = model_class()

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()])
    
    # initialize weights
    model(np.expand_dims(sample_transaction, axis=0))
    model.load_weights(f'src/machine_learning/pre_loaded_models/{model.name}.keras')

    callback.set_model(model)
    lower_bound = model.evaluate(test_set, callbacks=[callback], return_dict=True)

    print(f"[{model.name}]:\n{lower_bound}")
