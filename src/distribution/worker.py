from machine_learning.models import CARD_ID_COLUMN
import tensorflow as tf
import numpy as np
import keras

from distribution.watcher import PerformanceTracker
from distribution.db_utils import register_results, database

def worker_function(id, model_type, merge, threshold, num_workers):
    # --------- DATASET LOAD -----------
    dataset = np.append(np.load(f'src/data/test/all_transactions.npy'), 
                        np.expand_dims(np.load(f'src/data/test/all_labels.npy'), axis=1), 
                        axis=1)

    mask = (dataset[:, CARD_ID_COLUMN] % num_workers == id)
    set = dataset[mask, :]
 
    dataset = set[:, :18]   
    labels = set[:, -1]
    dataset = tf.data.Dataset.from_tensor_slices((dataset, labels)).batch(1)

    # ---------- INITIALIZE MODEL ----------
    model = model_type()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[keras.metrics.BinaryAccuracy(),
                    keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                    keras.metrics.FalsePositives(), keras.metrics.FalseNegatives(),
                    tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()])

    sample_transaction = np.load(f'src/data/test/all_transactions.npy')[0]
    model(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model

    # ---------- RUN MODEL -------------
    model.load_weights(f'src/machine_learning/pre_loaded_models/{model.name}.keras')

    state_callback = merge(id, threshold, num_workers)
    tracker_callback = PerformanceTracker()
    state_callback.set_model(model)
    
    ml_results = model.evaluate(dataset, callbacks=[tracker_callback, state_callback])

    print(f"FIRST ONE TO FINISH: {id}, writing {state_callback.version}")
    database.set(f'last_version_{id}', str(state_callback.version))
    state_callback.close_threads()
    
    time_results = tracker_callback.display_results()
    read_time, write_time = state_callback.display_results()
    register_results(model, merge, threshold, num_workers, ml_results, time_results, read_time, write_time)
    
    print(f'[WORKER {id}]:  ------------ RESULTS ------------ \
          \n\tTotal time: {time_results[0]} s, Average forward pass: {time_results[1]} s, Throughput: {time_results[2]} \
          \n\tLoss: {ml_results[0]}, Accuracy: {ml_results[1]}, Precision: {ml_results[6]}, Recall: {ml_results[7]}, AUC: {ml_results[8]} \
          \n\tDatabase: read: {read_time}, wrote: {write_time}')
