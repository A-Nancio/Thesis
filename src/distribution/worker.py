
from data_processing.batch_generator import load_test_set
from machine_learning.models import FeedzaiProduction, DoubleStateProduction, DistributedProducion, CARD_ID_COLUMN
from tqdm import tqdm
import tensorflow as tf

import numpy as np
import keras

from distribution.watcher import PerformanceTracker

def worker_function(id):
    # --------- DATASET LOAD -----------
    dataset = np.append(np.load(f'src/data/test/transactions.npy'), 
                        np.expand_dims(np.load(f'src/data/test/all_transaction_labels.npy'), axis=1), 
                        axis=1)
    
    # append indexes of each transaction
    #np.append(dataset,
    #          np.arange(dataset.shape[0]),
    #          axis=1)

    # mask = (dataset[:, CARD_ID_COLUMN] % 3 == id)
    set = dataset#[mask, :]
 
    dataset = set[:, :18]   # one extra column, to include index
    labels = set[:, -1]
    
    dataset = load_test_set()# tf.data.Dataset.from_tensor_slices((dataset, labels)).batch(1)


    # ---------- INITIALIZE MODEL ----------
    model = DistributedProducion()
    sample_transaction = np.load(f'src/data/test/transactions.npy')[0]
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[keras.metrics.BinaryAccuracy(),
                    keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                    keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    model(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model
    
    weight_path = f'src/machine_learning/saved_models/Feedzai_{3}.keras'

    # !!!!!!!!!!!!!!1 BIG NOTE VERIFY IF WEIGHT LOADING WORKS AFTER ADDING DELTA VARIABLES IN MODELS !!!!!!!!!!!!
    model.load_weights(weight_path)
    model.reset_gru()

    # ---------- RUN MODEL -------------
    callback = PerformanceTracker()
    callback.set_model(model)
    
    # NOTE Simpler version, implement callbacks to measure metrics
    results = model.evaluate(dataset, callbacks=callback)
    
    print(f'[WORKER {id}]: {callback.display_results()}')
    