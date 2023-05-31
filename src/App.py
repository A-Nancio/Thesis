import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_train_set, load_test_set
from machine_learning.models import Feedzai, BATCH_SIZE, CARD_ID_COLUMN, CATEOGRY_ID_COLUMN
import tensorflow as tf
from keras import metrics
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

SEQUENCE_LENGTH = 100
NUM_EPOCHS = 5

train_model = Feedzai(training_mode=True)
production_model = Feedzai(training_mode=False)

train_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])

production_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
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

train_set = load_train_set(sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
test_set = load_test_set()
pre_inference_data = np.load(f'data/train/transactions.npy')[-200:,:]   # Last 200 transaction from training to prepare the inference model state


results = np.ndarray(shape=(0,0), dtype=float)
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch}:")
    
    # train the model
    train_model.fit(train_set, epochs=NUM_EPOCHS, class_weight=class_weight, verbose='auto', shuffle=True)

    #transfer the weights to production model to evaluate
    production_model.set_weights(train_model.get_weights())

    # Prepare the state of the model
    print('Preparing state: ')
    production_model.evaluate(pre_inference_data, batch_size=1)

    # Evaluate the model
    results = np.append(results, production_model.evaluate(test_set, batch_size=1))

    print(results)


#mpl.rcParams['figure.figsize'] = (25, 6)
#mpl.rcParams['axes.grid'] = False
#
#plt.title("Single shared state model Evaluation") 
#plt.xlabel("input indexes") 
#plt.ylabel("predictions (degrees celcius)") 
#
#plt.plot(indexes, new, label = "sequence length 1") 
#plt.legend()
#plt.show()
#TODO store results



