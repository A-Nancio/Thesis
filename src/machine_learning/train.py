"""Tensorflow components to be used"""
import time
import tensorflow as tf
from tensorflow import keras
from keras import metrics
from keras.losses import BinaryCrossentropy

optimizer = keras.optimizers.Adam()
loss_fn = BinaryCrossentropy()

total = 1296675
neg = 1289169
pos = 7506

weight_0 = (1 / neg) * (total / 2.0)
weight_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_0, 1: weight_1}


class ModelManager():
    def __init__(self, model: tf.keras.Model, dataHandler) -> None:
        self.model = model
        self.dataHandler = dataHandler
        

validation_metrics = (metrics.BinaryAccuracy(),
               metrics.TruePositives(), metrics.TrueNegatives(),
               metrics.FalsePositives(), metrics.FalseNegatives())


@tf.function
def train_step(model, batch, labels):
    """Perform a training step for a batch of training data"""
    with tf.GradientTape() as tape:
        logits = model(batch, training=True)
        loss_value = loss_fn(labels, logits,)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


@tf.function
def test_step(model, batch, labels):
    """Analyse the performance of the model by predicitng a batch"""
    val_logits = model(batch, training=False)
    for metric in validation_metrics:
        metric.update_state(labels, val_logits)


def train_model(model, train_data, train_labels, train_transactions):
    """Train a model through a set of training data"""
    start_time = time.time()
    step = 0

    for batch, label in zip(train_data, train_labels):
        # Convert transaction IDs to actual transactions to be fed to the model
        batch = train_transactions[batch]
        loss_value = train_step(model, batch, label)
        # log every 200 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for one batch) at step {step}: {loss_value}")
        step += 1

    return loss_value, time.time() - start_time


def test_model(model, test_data, test_labels, test_transactions):
    """Validate model through a set of validation data"""
    start_time = time.time()

    step = 1
    for input, label in zip(test_data, test_labels):
        input = test_transactions[input]
        test_step(model, input, label)
        
        results = [float(metric.result()) for metric in validation_metrics]
        if step % 200 == 0:
            print(f"Validation metrics: {results[0]} acc, {results[1]} TP, {results[2]} TN, {results[3]} FP, {results[4]} FN", end='\r', flush=True)

        step += 1
    
    print(f"\nTime taken: {time.time() - start_time}")
    results = [float(metric.result()) for metric in validation_metrics]
    for metric in validation_metrics: metric.reset_state()
    return results


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
