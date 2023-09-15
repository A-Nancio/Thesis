
from data_processing.batch_generator import load_test_set
from machine_learning.models import FeedzaiProduction, DoubleStateProduction, DistributedProducion, CARD_ID_COLUMN
from tqdm import tqdm
import tensorflow as tf
import sys
import numpy as np
import keras

from distribution.watcher import PerformanceTracker, StateWriter

def worker_function(id):
    # --------- DATASET LOAD -----------
    dataset = np.append(np.load(f'src/data/test/transactions.npy'), 
                        np.expand_dims(np.load(f'src/data/test/all_transaction_labels.npy'), axis=1), 
                        axis=1)
    
    # append indexes of each transaction
    #np.append(dataset,
    #          np.arange(dataset.shape[0]),
    #          axis=1)

    mask = (dataset[:, CARD_ID_COLUMN] % 10 == id)
    set = dataset[mask, :]
 
    dataset = set[:, :18]   
    labels = set[:, -1]

    
    # dataset = load_test_set()
    dataset = tf.data.Dataset.from_tensor_slices((dataset, labels)).batch(1)


    # ---------- INITIALIZE MODEL ----------
    model = DistributedProducion()

    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[keras.metrics.BinaryAccuracy(),
                    keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                    keras.metrics.FalsePositives(), keras.metrics.FalseNegatives(),tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()])

    
    
    sample_transaction = np.load(f'src/data/test/transactions.npy')[0]
    model(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model
    
    

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
    # helper_model.load_weights(f'src/machine_learning/saved_models/double/Double_15.keras')
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
    # model.layer.set_weights(helper_model.layer.get_weights())
    # model.dense.set_weights(helper_model.dense.get_weights())
    # model.out.set_weights(helper_model.out.get_weights())
# 
    # model.save_weights(
    #     filepath='src/machine_learning/saved_models/distribution/Double_15.keras',
    #     save_format='h5'
    # )
    # sys.exit()

    # -------------------------------------



    
    # ---------- RUN MODEL -------------
    model.load_weights(f'src/machine_learning/saved_models/distribution/Double_15.keras')

    state_callback = StateWriter(id=id, threshold=10000)
    tracker_callback = PerformanceTracker()
    tracker_callback.set_model(model)
    
    # NOTE Simpler version, implement callbacks to measure metrics
    results = model.evaluate(dataset, callbacks=[tracker_callback])
    
    print(f'[WORKER {id}]: {tracker_callback.display_results()}')
    