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
NUM_FOLDS=5
transactions = np.load(f'src/data/all_transactions.npy')


model_set = [
    (FeedzaiTrain, FeedzaiProduction),
    (FeedzaiExtraTrain, FeedzaiExtraProduction),
    (FeedzaiConcatTrain, FeedzaiConcatProduction),
    (FeedzaiExtraConcatTrain, FeedzaiExtraConcatProduction)
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
    production_model(np.expand_dims(transactions[0], axis=0)) # a single forward pass to initialize weights for the model

    print(f"---------------------- MODEL {train_model.name} ----------------------")
    
    with open(f'src/training_runs/train_{production_model.name}.csv', 'w') as train_f:
        with open(f'src/training_runs/validation_{production_model.name}.csv', 'w')  as val_f:
            train_writer = csv.writer(train_f)
            val_writer = csv.writer(val_f) 

            header = ['fold', 'epoch', 'loss', 'binary_accuracy','TP', 'TN', 'FP', 'FN','precision', 'recall', 'auc']
            train_writer.writerow(header)
            val_writer.writerow(header)

            for fold in range(NUM_FOLDS):
                #  --------- NOTE DATASETS ------------
                training_set = tf.data.Dataset.from_tensor_slices(
                    (np.load(f'src/data/train/fold_{fold}/train_seq_ids.npy').astype(int), 
                    np.load(f'src/data/train/fold_{fold}/train_seq_labels.npy').astype(float))
                    ).batch(BATCH_SIZE)

                validation_set = tf.data.Dataset.from_tensor_slices(
                    (np.load(f'src/data/train/fold_{fold}/val_seq_ids.npy').astype(int), 
                    np.load(f'src/data/train/fold_{fold}/val_seq_labels.npy').astype(float))
                    ).batch(BATCH_SIZE)
                
                
                for epoch in range(NUM_EPOCHS):
                    training_set.shuffle(buffer_size=training_set.cardinality())

                    # NOTE TRAINING
                    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(training_set), total=training_set.cardinality().numpy()):
                        x_batch_train = transactions[x_batch_train]
                        train_results = train_model.train_on_batch(x_batch_train, y_batch_train, reset_metrics=False)
                    
                    print(f"{train_model.name} training: [fold {fold}, epoch {epoch}], results: {train_results}")
                    train_model.reset_metrics()
                    if "double" in train_model.name:
                        train_model.reset_gru()

                    # NOTE VALIDATION
                    for step, (x_batch_validation, y_batch_validation) in tqdm(enumerate(validation_set), total=validation_set.cardinality().numpy()):
                        x_batch_validation = transactions[x_batch_validation]
                        validation_results = train_model.test_on_batch(x_batch_validation, y_batch_validation, reset_metrics=False)
                    
                    print(f"{train_model.name} validation: [fold {fold}, epoch {epoch}], results: {validation_results}")
                    train_model.reset_metrics()
                    if "double" in train_model.name:
                        train_model.reset_gru()
                    
                    train_writer.writerow([fold, epoch] + train_results)
                    val_writer.writerow([fold, epoch] + validation_results)

                    # NOTE SAVE MODEL
                    # production_model.set_weights(train_model)
                    # weight_path = f'src/machine_learning/saved_models/{production_model.name}_{epoch}.keras'
                    # production_model.save_weights(
                    #     filepath=weight_path,
                    #     save_format='h5'
                    # )
