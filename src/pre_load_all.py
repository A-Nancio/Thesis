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

train_set = tf.data.Dataset.from_tensor_slices(
    (np.load(f'src/data/train/all_transactions.npy'), 
     np.load(f'src/data/train/all_labels.npy').astype(float))
).batch(1)

test_set = tf.data.Dataset.from_tensor_slices(
    (np.load(f'src/data/test/all_transactions.npy'), 
     np.load(f'src/data/test/all_labels.npy').astype(float))
).batch(1)

model_list = {
    FeedzaiProduction: 11,
    FeedzaiExtraProduction: 11,
    FeedzaiConcatProduction: 10,
    FeedzaiExtraConcatProduction: 14,
    DoubleProduction: 19,
    DoubleExtraProduction: 3,
    DoubleConcatProduction: 7,
    DoubleExtraConcatProduction: 17
}

with open(f'src/training_runs/preloaded_state_evaluation.csv', 'w') as f:
    writer = csv.writer(f)
    header = ['model', 'loss', 'binary_accuracy','TP', 'TN', 'FP', 'FN','precision', 'recall', 'auc']
    writer.writerow(header)
    for model_class, epoch in model_list.items():
        tf.keras.backend.clear_session()

        model = model_class()
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
        model.load_weights(f'src/machine_learning/saved_models/{model.name}_{epoch}.keras')
        
        print(f"---------------------- MODEL {model.name} ----------------------")
        # NOTE preload state
        model.reset_gru()
        model.evaluate(train_set)

        weight_path = f'src/machine_learning/pre_loaded_models/{model.name}.keras'
        model.save_weights(
            filepath=weight_path,
            save_format='h5'
        )

        results = model.evaluate(test_set)
        writer.writerow([model.name] + results)

