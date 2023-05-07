import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_train_set
from machine_learning.models import StatefulModel
from machine_learning.train import train_model, test_model
import tensorflow as tf
from keras import metrics


#model.compile(
#    optimizer=keras.optimizers.Adam(),
#    loss=keras.losses.BinaryCrossentropy(from_logits=True),
#    metrics=(metrics.BinaryAccuracy(),
#               metrics.TruePositives(), metrics.TrueNegatives(),
#               metrics.FalsePositives(), metrics.FalseNegatives()))
SEQUENCE_LENGTH = 100
BATCH_SIZE = 128
NUM_EPOCHS = 5

StatefulModel.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[metrics.BinaryAccuracy(),
               metrics.TruePositives(), metrics.TrueNegatives(),
               metrics.FalsePositives(), metrics.FalseNegatives()])

total = 1296675
neg = 1289169
pos = 7506

weight_0 = (1 / neg) * (total / 2.0)
weight_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_0, 1: weight_1}


#ratios = ((16,16), (15,17), (14,18), (13,19), (12,20), (11,21))
#ratios = ((28, 4),)


#metrics = {}
#test_set, test_labels, test_transactions = import_test_set()

#for ratio in ratios:

train_set = load_train_set(sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)

StatefulModel.fit(train_set, epochs=NUM_EPOCHS, class_weight=class_weight, verbose='auto', shuffle=True)
#for epoch in range(NUM_EPOCHS):
#    metrics[ratios] = ()
#    print(f"\nStart of epoch {epoch}")
#    print("------- TRAINING LOOP -------")
#    train_model(model, train_set)
#    print("------- TESTING LOOP -------")
#    validation_results = test_model(model, test_set, test_labels, test_transactions)
#    metrics[ratios] += (validation_results,)
#
#    #model.reset
#    #break
#
#    
#print(metrics)
#