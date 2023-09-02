# Train model for one epoch and save weights
import os

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_test_set, get_class_weights
from machine_learning.models import CARD_ID_COLUMN, CATEOGRY_ID_COLUMN, FeedzaiTrainAsync, FeedzaiProduction, BATCH_SIZE, SharedStateAsync
from machine_learning.models import DoubleStateProduction
import tensorflow as tf
import numpy as np
from keras import metrics


model = FeedzaiProduction()

single_trans_set = load_test_set()

transactions = np.load(f'src/data/test/transactions.npy')[0:1000]


model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])

#print(np.expand_dims(tr_1, axis=0).shape)
model(np.expand_dims(transactions[0], axis=0)) # a single forward pass to initialize weights for the model
tf.print(model.weights[0].read_value(), summarize=-1)


weight_path = f'src/machine_learning/saved_models/Feedzai_{3}.keras'
model.load_weights(weight_path)
model.reset_gru()




tf.print(model.weights[0].read_value(), summarize=-1)
for trans in transactions:
    model(np.expand_dims(trans, axis=0))

print("TESTING  --------------------------------------------------------------------\n--------------------------------------------------------------")
tf.print(model.weights[0].read_value(), summarize=-1)

#model(np.expand_dims(tr_2, axis=0))
#
#model(np.expand_dims(tr_3, axis=0))
## print(model.card_gru.weights[0])
#model(np.expand_dims(sample_transaction, axis=0))
# print(model.card_gru.weights[0])

#model.evaluate(single_trans_set)
