
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from redis import StrictRedis
from data_processing.batch_generator import load_test_set
from machine_learning.models import FeedzaiProduction, DoubleStateProduction, DistributedProducion, CARD_ID_COLUMN, SimplerDistributedProduction
import tensorflow as tf
import numpy as np
import keras
from tqdm import tqdm

model_0 = SimplerDistributedProduction()
model_1 = SimplerDistributedProduction()


#model_2 = SimplerDistributedProduction()
#model_3 = SimplerDistributedProduction()

dataset = np.append(np.load(f'src/data/test/transactions.npy'), 
                        np.expand_dims(np.load(f'src/data/test/all_transaction_labels.npy'), axis=1), 
                        axis=1)

mask_0 = (dataset[:, CARD_ID_COLUMN] % 2 == 0)
mask_1 = (dataset[:, CARD_ID_COLUMN] % 2 == 1)
#mask_2 = (dataset[:, CARD_ID_COLUMN] % 4 == 2)
#mask_3 = (dataset[:, CARD_ID_COLUMN] % 4 == 3)

#set_0 = dataset[mask_0, :]
#set_1 = dataset[mask_1, :]
# 
#dataset_0 = set_0[:, :18]   
#labels_0 = set_0[:, -1]
# 
# dataset_1 = set_1[:, :18]   
# labels_1 = set_1[:, -1]

#set = dataset

#dataset = set[:, :18]   
#labels = set[:, -1]

dataset_0 = tf.data.Dataset.from_tensor_slices((dataset[mask_0, :18], dataset[mask_0, -1])).batch(1)
dataset_1 = tf.data.Dataset.from_tensor_slices((dataset[mask_1, :18], dataset[mask_1, -1])).batch(1)
#dataset_2 = tf.data.Dataset.from_tensor_slices((dataset[mask_2, :18], dataset[mask_2, -1])).batch(1)
#dataset_3 = tf.data.Dataset.from_tensor_slices((dataset[mask_3, :18], dataset[mask_3, -1])).batch(1)


    
model_0.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[keras.metrics.BinaryAccuracy(),
                keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                keras.metrics.FalsePositives(), keras.metrics.FalseNegatives(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()])

model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[keras.metrics.BinaryAccuracy(),
                keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                keras.metrics.FalsePositives(), keras.metrics.FalseNegatives(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()])

# model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
#             optimizer=tf.keras.optimizers.Adam(),
#             metrics=[keras.metrics.BinaryAccuracy(),
#                 keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
#                 keras.metrics.FalsePositives(), keras.metrics.FalseNegatives(),
#                 tf.keras.metrics.Precision(),
#                 tf.keras.metrics.Recall()])
# 
# model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
#             optimizer=tf.keras.optimizers.Adam(),
#             metrics=[keras.metrics.BinaryAccuracy(),
#                 keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
#                 keras.metrics.FalsePositives(), keras.metrics.FalseNegatives(),
#                 tf.keras.metrics.Precision(),
#                 tf.keras.metrics.Recall()])

sample_transaction = np.load(f'src/data/test/transactions.npy')[0]
model_0(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model
model_1(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model
#model_2(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model
#model_3(np.expand_dims(sample_transaction, axis=0)) # a single forward pass to initialize weights for the model


model_0.load_weights(f'src/machine_learning/saved_models/distribution/Simpler_Double_3.keras')
model_1.load_weights(f'src/machine_learning/saved_models/distribution/Simpler_Double_3.keras')
#model_2.load_weights(f'src/machine_learning/saved_models/distribution/Simpler_Double_3.keras')
#model_3.load_weights(f'src/machine_learning/saved_models/distribution/Simpler_Double_3.keras')


#with tf.device("/gpu:0"):
step = 0
for entry_0, entry_1 in tqdm(zip(enumerate(dataset_0), enumerate(dataset_1)), total=dataset_1.cardinality().numpy()):
    # tf.print(entry_1[1])

    res_0 = model_0.test_on_batch(
        entry_0[1][0],
        entry_0[1][1],
        reset_metrics=False,
        return_dict=True
    )
    res_1 = model_1.test_on_batch(
        entry_1[1][0],
        entry_1[1][1],
        reset_metrics=False,
        return_dict=True
    )
    # res_2 = model_2.test_on_batch(
    #     entry_2[1][0],
    #     entry_2[1][1],
    #     reset_metrics=False,
    #     return_dict=True
    # )
    # res_3 = model_3.test_on_batch(
    #     entry_3[1][0],
    #     entry_3[1][1],
    #     reset_metrics=False,
    #     return_dict=True
    # )

    
    model_0.category_gru.shared_states.assign_add(
        model_1.category_gru.deltas, use_locking=True)
    #model_0.category_gru.shared_states.assign_add(
    #    model_2.category_gru.deltas, use_locking=True)
    #model_0.category_gru.shared_states.assign_add(
    #    model_3.category_gru.deltas, use_locking=True)

    model_1.category_gru.shared_states.assign_add(
        model_0.category_gru.deltas, use_locking=True)
    #model_1.category_gru.shared_states.assign_add(
    #    model_2.category_gru.deltas, use_locking=True)
    #model_1.category_gru.shared_states.assign_add(
    #    model_3.category_gru.deltas, use_locking=True)

    #model_2.category_gru.shared_states.assign_add(
    #    model_0.category_gru.deltas, use_locking=True)
    #model_2.category_gru.shared_states.assign_add(
    #    model_1.category_gru.deltas, use_locking=True)
    #model_2.category_gru.shared_states.assign_add(
    #    model_3.category_gru.deltas, use_locking=True)
    
    #model_3.category_gru.shared_states.assign_add(
    #    model_0.category_gru.deltas, use_locking=True)
    #model_3.category_gru.shared_states.assign_add(
    #    model_1.category_gru.deltas, use_locking=True)
    #model_3.category_gru.shared_states.assign_add(
    #    model_2.category_gru.deltas, use_locking=True)
    
    model_0.category_gru.reset_deltas()
    model_1.category_gru.reset_deltas()
    #model_2.category_gru.reset_deltas()
    #model_3.category_gru.reset_deltas()

    step += 1
    if step == 1000:
        step = 0
        tf.print(model_0.category_gru.shared_states[2], summarize=-1)
        # tf.print(res_1)
        tf.print("-----------")
    # with tf.GradientTape() as tape:
    #     results = train_model.train_on_batch(
    #         x_batch_train,
    #         y_batch_train,
    #         reset_metrics=False,
    #         return_dict=True
    #     )
    # print(results)