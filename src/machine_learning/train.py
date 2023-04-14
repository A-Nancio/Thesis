import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf
import os

from tensorflow import keras
from keras import layers
from keras import metrics
from keras.losses import BinaryCrossentropy

import time

from models import Stateless, GlobalState

NUM_FEATURES = 18
CUTOFF_LENGTH = 100
BATCH_SIZE = 100
NUM_EPOCHS = 5
TEST_SIZE = 200000

transactions = np.load('datasets/modified/modified_sparkov.npy')
sequences = np.load('datasets/modified/sparkov_sequences.npy')
labels = np.load('datasets/modified/sparkov_labels.npy')

train_sequences, val_sequences = np.split(sequences, [int(0.67 * len(sequences))])
train_labels, val_labels = np.split(labels, [int(0.67 * len(labels))])

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_dataset = val_dataset.batch(BATCH_SIZE)

# to save memory, delete no longer used variables
del sequences, labels, train_sequences, train_labels, val_sequences, val_labels


#model = GlobalState(transactions.shape[1])
model = GlobalState()
#model.fit(dataset, epochs=3, batch_size=32)
optimizer = keras.optimizers.Adam()
loss_fn = BinaryCrossentropy(from_logits=True)

val_metrics = (metrics.BinaryAccuracy(), 
               metrics.TruePositives(), metrics.TrueNegatives(), 
               metrics.FalsePositives(), metrics.FalseNegatives())

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    for metric in val_metrics:
        metric.update_state(y, val_logits)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


for epoch in range(NUM_EPOCHS):
    print("\nStart of epoch %d/%d" % (epoch+1,NUM_EPOCHS))
    start_time = time.time()

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):        
        # Convert transaction IDs to actual transactions to be fed to the model
        x_batch_train = transactions[x_batch_train]
        loss_value = train_step(x_batch_train, y_batch_train)
        #log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.7f"
                % (step, float(loss_value)))
        if step == 2000:
            break      

    # display metrics at the end of each epoch.
    #results = [float(metric.result()) for metric in train_metrics]
    #print("Metrics over epoch: " + str(results[0]) + " Acc, " + str(results[1]) + " Precision, " + str(results[2]) +" Recall, "+ str(results[3]) +" AUC")
    #for metric in train_metrics: 
    #    metric.reset_states()
    
    true = 0
    false = 0
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset: 
    #    print(1 in y_batch_val)
    #    if (1 in y_batch_val):
    #        true +=1
    #    else: 
    #        false +=1
        x_batch_val = transactions[x_batch_val]
        test_step(x_batch_val, y_batch_val)

    results = [float(metric.result()) for metric in val_metrics]
    print("Validation acc: " + str(results[0]) + " Acc, " + str(results[1]) + " TP, " + str(results[2]) +" TN, "+ str(results[3]) +" FP, "+ str(results[4]) +" FN")
    print("Time taken: %.2fs" % (time.time() - start_time))