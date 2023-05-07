"""Use numpy to generate batches"""
import sys
import numpy as np
import tensorflow as tf

def load_train_set(sequence_length, batch_size):
    path = 'data/train'
    transactions = np.load(f'{path}/transactions.npy')
    labels = np.load(f'{path}/all_transaction_labels.npy')
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
    dataset = np.load(f'{path}/transactions.npy')
    labels = np.load(f'{path}/all_transaction_labels.npy')
    return tf.data.Dataset.from_tensor_slices(dataset, labels)



SEQUENCE_LENGTH = 100

def import_training_set(ratio: tuple[int, int]):
    """Load .npy sequences from disk, generate batches and return dataset for training"""

    path = 'data/train'
    transactions = np.load(f'{path}/transactions.npy')
    non_frauds = np.load(f'{path}/sequences/non_fraud_train_seq.npy')
    frauds = np.load(f'{path}/sequences/fraud_train_seq.npy')

    batch_size = ratio[0] + ratio[1]

    batch = np.empty(shape=(0, SEQUENCE_LENGTH), dtype=np.int64)
    dataset = np.empty(shape=(0, batch_size, SEQUENCE_LENGTH), dtype=np.int64)
    labels = np.empty(shape=(0, batch_size), dtype=np.int64)
    np.random.shuffle(non_frauds)
    #np.random.shuffle(frauds)

    original_frauds = np.copy(frauds)
    num_batches = 0
    sys.stdout.write(f"Generating batches for {ratio[0]} non fraud and {ratio[1]} fraud transactions: ")
    sys.stdout.flush()

    while len(non_frauds) > ratio[0]:
        np.random.shuffle(frauds)

        
        if num_batches % 100 == 0:
            print(num_batches)
        num_batches += 1

        batch = np.append(non_frauds[0:ratio[0]],
                          frauds[0:ratio[1]], axis=0)[np.newaxis, :, :]
        dataset = np.append(dataset, batch, axis=0)

        # Remove sampled transactions
        #frauds = frauds[ratio[1]:]
        non_frauds = non_frauds[ratio[0]:]

        # Add labels
        batch_labels = np.append(
            np.zeros((ratio[0],), dtype=int), np.ones((ratio[1],), dtype=int))
        labels = np.append(labels, batch_labels[np.newaxis, :], axis=0)

    print(f"\nGenerated {num_batches} batches")
    print(
        f"Generated train dataset with shape {dataset.shape} and labels {labels.shape}")
    
    return dataset, labels, transactions

def import_full_training_set():
    path = 'data/train'
    transactions = np.load(f'{path}/transactions.npy')
    non_frauds = np.load(f'{path}/sequences/non_fraud_train_seq.npy')
    frauds = np.load(f'{path}/sequences/fraud_train_seq.npy')


def import_test_set():
    path = 'data/test'
    dataset = np.load(f'{path}/all_test_seq.npy')[:,np.newaxis, :]
    labels = np.load(f'{path}/all_test_seq_labels.npy')[:,np.newaxis]
    transactions = np.load(f'{path}/transactions.npy')

    print(
        f"Generated test dataset with shape {dataset.shape} and labels {labels.shape}")
    return dataset, labels, transactions
    # transaction_labels = np.load(f'{path}/all_transactions_labels.npy')

