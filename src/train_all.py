# Train model for one epoch and save weights
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tqdm import tqdm

from machine_learning.models import FeedzaiTrain, FeedzaiProduction, FeedzaiExtraTrain, FeedzaiExtraProduction, FeedzaiConcatTrain, FeedzaiConcatProduction, FeedzaiExtraConcatTrain, FeedzaiExtraConcatProduction, DoubleTrain, DoubleProduction, DoubleExtraTrain, DoubleExtraProduction, DoubleConcatTrain, DoubleConcatProduction, DoubleExtraConcatTrain, DoubleExtraConcatProduction
from machine_learning.shared_state import BATCH_SIZE
import tensorflow as tf
import numpy as np
import csv
tf.random.set_seed(1234)

SEQUENCE_LENGTH = 100
NUM_EPOCHS = 20

transactions = np.load(f'src/data/all_transactions.npy')

training_set = tf.data.Dataset.from_tensor_slices(
    (np.load(f'src/data/train/all_seq_ids.npy').astype(int), 
     np.load(f'src/data/train/all_seq_labels.npy').astype(float))
     ).batch(BATCH_SIZE)


model_set = [
    (DoubleTrain, DoubleProduction),
    (DoubleExtraTrain, DoubleExtraProduction),
    (DoubleConcatTrain, DoubleConcatProduction),
    (DoubleExtraConcatTrain, DoubleExtraConcatProduction)
    ]

for train_model, production_model in model_set:
    tf.keras.backend.clear_session()

    train_model = train_model()
    production_model = production_model()
    train_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
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
    production_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
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
    
    print(f"---------------------- MODEL {train_model.name} ----------------------")
    
    with open(f'src/training_runs/train_{production_model.name}.csv', 'w') as train_f:
        train_writer = csv.writer(train_f)

        header = ['epoch', 'loss', 'binary_accuracy','TP', 'TN', 'FP', 'FN','precision', 'recall', 'auc']
        
        train_writer.writerow(header)
        
        production_model(np.expand_dims(transactions[0], axis=0)) # a single forward pass to initialize weights for the model
        for epoch in range(NUM_EPOCHS):
            training_set.shuffle(buffer_size=training_set.cardinality())

            # NOTE TRAINING
            for step, (x_batch_train, y_batch_train) in tqdm(enumerate(training_set), total=training_set.cardinality().numpy()):
                x_batch_train = transactions[x_batch_train]
                train_results = train_model.train_on_batch(x_batch_train, y_batch_train, reset_metrics=False)
            train_writer.writerow([epoch] + train_results)
            print(f"{train_model.name} training: epoch {epoch}, results: {train_results}")

            train_model.reset_metrics()
            if "double" in train_model.name:
                train_model.reset_gru()
            
            
            # NOTE SAVE MODEL
            production_model.set_weights(train_model)
            weight_path = f'src/machine_learning/saved_models/{production_model.name}_{epoch}.keras'
            production_model.save_weights(
                filepath=weight_path,
                save_format='h5'
            )
