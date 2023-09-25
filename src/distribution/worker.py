
from redis import StrictRedis
from data_processing.batch_generator import load_test_set
from machine_learning.models import FeedzaiProduction, DoubleStateProduction, DistributedProducion, CARD_ID_COLUMN, SimplerDistributedProduction
import tensorflow as tf
import numpy as np
import keras

import csv

from distribution.watcher import PerformanceTracker, StateWriter, subscribe
from distribution.db_utils import from_redis, register_results

def worker_function(id, threshold, num_workers):
    # --------- DATASET LOAD -----------
    dataset = np.append(np.load(f'src/data/test/transactions.npy'), 
                        np.expand_dims(np.load(f'src/data/test/all_transaction_labels.npy'), axis=1), 
                        axis=1)
    

    mask = (dataset[:, CARD_ID_COLUMN] % num_workers == id)
    set = dataset[mask, :]
 
    dataset = set[:, :18]   
    labels = set[:, -1]

    
    # dataset = load_test_set()
    dataset = tf.data.Dataset.from_tensor_slices((dataset, labels)).batch(1)


    # ---------- INITIALIZE MODEL ----------
    model = SimplerDistributedProduction()

    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[keras.metrics.BinaryAccuracy(),
                    keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                    keras.metrics.FalsePositives(), keras.metrics.FalseNegatives(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()])

    
    
    sample_transaction = np.load(f'src/data/test/transactions.npy')[0]
    model(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model

    
    # ---------- RUN MODEL -------------
    model.load_weights(f'src/machine_learning/saved_models/distribution/Simpler_Double_3.keras')

    state_callback = StateWriter(id=id, threshold=threshold)
    tracker_callback = PerformanceTracker()

    state_callback.set_model(model)
    
    subscription = subscribe(id, model)

    ml_results = model.evaluate(dataset, callbacks=[tracker_callback, state_callback])#, verbose=0)
    
    subscription.stop()
    
    time_results = tracker_callback.display_results()
    register_results(threshold, num_workers, ml_results, time_results)
    
    print(f'[WORKER {id}]:  ------------ RESULTS ------------ \
          \n\tTotal time: {time_results[0]} s, Average forward pass: {time_results[1]} s, Throughput: {time_results[2]} \
          \n\tLoss: {ml_results[0]}, Accuracy: {ml_results[1]}, Precision: {ml_results[2]}, Recall: {ml_results[3]}')



    # NOTE CODE TO SET UP A DISTRIBUTED MODEL BEFORE RUNNING
    # helper_model = DoubleStateProduction() 
    # helper_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
    #             optimizer=tf.keras.optimizers.Adam(),
    #             metrics=[keras.metrics.BinaryAccuracy(),
    #                 keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
    #                 keras.metrics.FalsePositives(), keras.metrics.FalseNegatives(),tf.keras.metrics.Precision(),
    #         tf.keras.metrics.Recall()])
    # 
    # helper_model(np.expand_dims(sample_transaction, axis=0))
    # 
    # helper_model.load_weights(f'src/machine_learning/saved_models/simpler_double/Double_3.keras')
    # 
    # model.card_gru.reset_states()
    # model.category_gru.reset_states()
    # helper_model.card_gru.reset_states()
    # helper_model.category_gru.reset_states()
    # 
    # model.card_gru.weights[1].assign(helper_model.card_gru.weights[1])
    # model.card_gru.weights[2].assign(helper_model.card_gru.weights[2])
    # model.card_gru.weights[3].assign(helper_model.card_gru.weights[3])
# 
    # model.category_gru.weights[2].assign(helper_model.category_gru.weights[1])
    # model.category_gru.weights[3].assign(helper_model.category_gru.weights[2])
    # model.category_gru.weights[4].assign(helper_model.category_gru.weights[3])
# 
    # # model.layer.set_weights(helper_model.layer.get_weights())
    # # model.dense.set_weights(helper_model.dense.get_weights())
    # model.out.set_weights(helper_model.out.get_weights())
# 
    # model.save_weights(
    #     filepath='src/machine_learning/saved_models/distribution/Simpler_Double_3.keras',
    #     save_format='h5'
    # )
    # sys.exit()

    # -------------------------------------