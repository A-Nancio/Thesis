"""Use numpy to generate batches"""
import sys
import numpy as np
import tensorflow as tf
from machine_learning.models import BATCH_SIZE

SEQUENCE_LENGTH = 100

def load_train_set(sequence_length, batch_size):
    path = 'data/train'
    transactions = np.load(f'{path}/transactions.npy')[-10*batch_size:]
    labels = np.load(f'{path}/all_transaction_labels.npy')[-10*batch_size:]

    return tf.keras.utils.timeseries_dataset_from_array(
        transactions,
        labels,
        sequence_length=sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=False,
        seed=None,
        start_index=None,
        end_index=None
    )

def load_test_set():
    path = 'data/test'
    dataset = np.load(f'{path}/transactions.npy')[-3000:]
    labels = np.load(f'{path}/all_transaction_labels.npy')[-3000:]

    return tf.data.Dataset.from_tensor_slices((dataset, labels)).batch(1)   # needs batch size 1 to have all sequential transactions

def load_pre_data():
    # Last 200 transaction from training to prepare the inference model state
    pre_inference_data = np.load(f'data/train/transactions.npy')[-200:]   
    pre_inference_labels = np.load(f'data/train/all_transaction_labels.npy')[-200:]

    return tf.data.Dataset.from_tensor_slices((pre_inference_data, pre_inference_labels)).batch(1)

def get_class_weights():
    # fetch class weights for model to handle highly imbalanced data
    total = 607506
    neg = 600000
    pos = 7506
    # FIXME change after to original value

    weight_0 = (1 / neg) * (total / 2.0)
    weight_1 = (1 / pos) * (total / 2.0)

    return {0: weight_0, 1: weight_1}
