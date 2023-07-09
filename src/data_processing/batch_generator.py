"""Use numpy to generate batches"""
import sys
import numpy as np
import tensorflow as tf
from machine_learning.models import BATCH_SIZE

SEQUENCE_LENGTH = 100
path = 'data'
def load_train_sample(sample_size: int):
    
    transactions = np.load(f'{path}/train/transactions.npy')[:-sample_size]
    labels = np.load(f'{path}/train/all_transaction_labels.npy')[:-sample_size]

    return tf.keras.utils.timeseries_dataset_from_array(
        transactions,
        labels,
        sequence_length=SEQUENCE_LENGTH,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=1,
        shuffle=False,
        seed=None,
        start_index=None,
        end_index=None
    )

def load_train_set(batch_size):
    # For complete dataset to align with batch size, the first 64 elements need to be excluded 
    transactions = np.load(f'{path}/train/transactions.npy')[64:] 
    labels = np.load(f'{path}/train/all_transaction_labels.npy')[64:]

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        transactions,
        labels,
        sequence_length=SEQUENCE_LENGTH,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=False,
        seed=None,
        start_index=None,
        end_index=None
    )
        
    return dataset

def load_test_set():
    dataset = np.load(f'{path}/test/transactions.npy')
    labels = np.load(f'{path}/test/all_transaction_labels.npy')

    return tf.data.Dataset.from_tensor_slices((dataset, labels)).batch(1)   # needs batch size 1 to have all sequential transactions

def load_pre_data():
    # Last 200 transaction from training to prepare the inference model state
    pre_inference_data = np.load(f'data/train/transactions.npy')[-200:]   
    pre_inference_labels = np.load(f'data/train/all_transaction_labels.npy')[-200:]

    return tf.data.Dataset.from_tensor_slices((pre_inference_data, pre_inference_labels)).batch(1)

def get_class_weights():
    total = 1000000
    neg = 900000
    pos = 100000

    weight_0 = (1 / neg) * (total / 2.0)
    weight_1 = (1 / pos) * (total / 2.0)

    return {0: weight_0, 1: weight_1}
