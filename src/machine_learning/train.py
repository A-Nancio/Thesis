"""Tensorflow components to be used"""
import time
import tensorflow as tf

from tensorflow import keras
from keras import metrics
from keras.losses import BinaryCrossentropy


optimizer = keras.optimizers.Adam()
loss_fn = BinaryCrossentropy(from_logits=True)

validation_metrics = (metrics.BinaryAccuracy(),
               metrics.TruePositives(), metrics.TrueNegatives(),
               metrics.FalsePositives(), metrics.FalseNegatives())


@tf.function
def train_step(model, batch, labels):
    """Perform a training step for a batch of training data"""
    with tf.GradientTape() as tape:
        logits = model(batch, training=True)
        loss_value = loss_fn(labels, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


@tf.function
def test_step(model, batch, labels):
    """Analyse the performance of the model by predicitng a batch"""
    val_logits = model(batch, training=False)
    for metric in validation_metrics:
        metric.update_state(labels, val_logits)


def train_model(model, train_data, train_labels):
    """Train a model through a set of training data"""
    start_time = time.time()
    step = 0

    for batch, label in zip(train_data, train_labels):
        # Convert transaction IDs to actual transactions to be fed to the model
        loss_value = train_step(model, batch, label)
        # log every 200 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for one batch) at step {step}: {loss_value}")
        step += 1

    return loss_value, time.time() - start_time


def validate_model(model, val_data, val_labels):
    """Validate model through a set of validation data"""
    start_time = time.time()

    for batch, label in zip(val_data, val_labels):
        test_step(model, batch, label)

    results = [float(metric.result()) for metric in validation_metrics]
    print(
        f'''Validation metrics: {results[0]} acc, {results[1]} TP, 
        {results[2]} TN, {results[3]} FP, {results[4]} FN''')
    print(f"Time taken: {time.time() - start_time}")

    for metric in validation_metrics: metric.reset_state()
    return results


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
