
from data_processing.batch_generator import load_test_set
from machine_learning.models import FeedzaiProduction, DoubleStateProduction, CARD_ID_COLUMN
import redis
from tqdm import tqdm
import tensorflow as tf

import numpy as np
import keras

from distribution.watcher import PerformanceCallback, CustomCallback

def worker_function(id):
    database = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # --------- DATASET LOAD -----------
    dataset = np.append(np.load(f'src/data/test/transactions.npy'), 
                        np.expand_dims(np.load(f'src/data/test/all_transaction_labels.npy'), axis=1), 
                        axis=1)

    mask = (dataset[:, CARD_ID_COLUMN] % 3 == id)
    set = dataset[mask, :]

    dataset = set[:, :18]
    labels = set[:, -1]
    
    dataset = tf.data.Dataset.from_tensor_slices((dataset, labels)).batch(1)


    # ---------- INITIALIZE MODEL ----------
    model = FeedzaiProduction()
    sample_transaction = np.load(f'src/data/test/transactions.npy')[0]
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[keras.metrics.BinaryAccuracy(),
                    keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                    keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])

    model(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model
    
    weight_path = f'src/machine_learning/saved_models/Feedzai_{3}.keras'
    model.load_weights(weight_path)


    # ---------- RUN MODEL -------------
    # NOTE Simpler version, implement callbacks to measure metrics
    results = model.evaluate(dataset) # TODO add callback 
    
    print(f'[WORKER {id}]: {results}')
    

    # NOTE more complex, but it is currently with poor time performance
    #print("beginning training")
    #for step, (x, y) in tqdm(enumerate(dataset), total=dataset.cardinality().numpy()):
    #    y_pred = model(x)
    #    for metric in model.metrics:
    #        if metric.name != "loss":
    #            metric.update_state(y, y_pred)

    #    if step % 10000 == 0:
    #        print(model.metrics)
    
    

