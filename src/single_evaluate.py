# Train model for one epoch and save weights
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tqdm import tqdm

from machine_learning.models import FeedzaiTrain, FeedzaiProduction, FeedzaiExtraTrain, FeedzaiExtraProduction, FeedzaiConcatTrain, FeedzaiConcatProduction, FeedzaiExtraConcatTrain, FeedzaiExtraConcatProduction, DoubleTrain, DoubleProduction, DoubleExtraTrain, DoubleExtraProduction, DoubleConcatTrain, DoubleConcatProduction, DoubleExtraConcatTrain, DoubleExtraConcatProduction
import tensorflow as tf
import numpy as np
import csv
tf.random.set_seed(42)

sample_transaction = np.load(f'src/data/all_transactions.npy')[0]
test_set = tf.data.Dataset.from_tensor_slices(
    (np.load(f'src/data/test/all_transactions.npy'), 
     np.load(f'src/data/test/all_labels.npy').astype(float))
).batch(1)


model = DoubleExtraConcatProduction()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=[
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.TruePositives(),
                    tf.keras.metrics.TrueNegatives(),
                    tf.keras.metrics.FalsePositives(),
                    tf.keras.metrics.FalseNegatives(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()])
model(np.expand_dims(sample_transaction, axis=0))

for epoch in [19]:
    model.load_weights(f'src/machine_learning/saved_models/{model.name}_{epoch}.keras')
    
    print(f"---------------------- MODEL {model.name} {epoch} ----------------------")
    model.reset_gru()
    results = model.evaluate(test_set)

