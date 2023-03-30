import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.metrics import AUC, Accuracy, Precision, Recall
from keras.losses import binary_crossentropy

import time

from models import Stateless, GlobalState

#TODO ENABLE GRAPH MODE FOR FASTER EXECUTION
#TODO ENABLE GPU USAGE, AND IF NECESSARY LIMIT IF IT USES ALL OF GPU MEMORY

NUM_FEATURES = 19
CUTOFF_LENGTH = 100
BATCH_SIZE = 100
NUM_EPOCHS = 10
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


model = GlobalState(transactions=transactions)

#model.fit(dataset, epochs=3, batch_size=32)
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

train_metrics = (Accuracy(), Precision(), Recall(), AUC())
val_metrics = (Accuracy(), Precision(), Recall(), AUC())


def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for metric in train_metrics:
        metric.update_state(y, logits)
    return loss_value

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
        print(
            "Training loss (for one batch) at step %d: %.7f"
            % (step, float(loss_value)))
        #if step % 200 == 0:
        #    
        #    )

    # display metrics at the end of each epoch.
    print("Metrics over epoch: %.4f Acc, %.4f Precision, %.4f Recall, %.4f AUC" % (float(x.result()) for x in train_metrics))
    for metric in train_metrics: 
        metric.reset_states()
    
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)
    
    print("Validation acc: %.4f, Precision %.4f, Recall %.4f AUC, %.4f" % (float(x.result()) for x in val_metrics))
    for metric in val_metrics: metric.reset_states()
    print("Time taken: %.2fs" % (time.time() - start_time))