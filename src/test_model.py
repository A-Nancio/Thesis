# Train model for one epoch and save weights
import os

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_test_set, get_class_weights
from machine_learning.models import FeedzaiProduction
from machine_learning.models import DoubleStateProduction
import tensorflow as tf
import numpy as np
from keras import metrics


model = DoubleStateProduction()

single_trans_set = load_test_set()

transactions = np.load(f'src/data/test/transactions.npy')


model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])

#print(np.expand_dims(tr_1, axis=0).shape)
model(np.expand_dims(transactions[0], axis=0)) # a single forward pass to initialize weights for the model


weight_path = f'src/machine_learning/saved_models/Double_{6}.keras'
model.load_weights(weight_path)
model.reset_gru()

model.evaluate(single_trans_set)
