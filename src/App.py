import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_train_set, load_test_set
from machine_learning.models import TrainSingleState, InferenceSingleState, BATCH_SIZE
import tensorflow as tf
from keras import metrics


SEQUENCE_LENGTH = 100
NUM_EPOCHS = 20

train_model = TrainSingleState()

train_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])

total = 1000000
neg = 800000
pos = 7506

weight_0 = (1 / neg) * (total / 2.0)
weight_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_0, 1: weight_1}

# TRAIN THE MODEL
train_set = load_train_set(sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)

train_model.fit(train_set, epochs=NUM_EPOCHS, class_weight=class_weight, verbose='auto', shuffle=True)

# MODEL IN PRODUCTION
production_model = InferenceSingleState()
production_model.set_weights(train_model.get_weights())

test_set = load_test_set()
production_model.evaluate(test_set)

#TODO store results



