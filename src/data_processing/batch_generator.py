"""Use numpy to generate batches"""
import sys
import numpy as np

FRAUD_RATIO = 16  # Num of frauds in a batch
NON_FRAUD_RATIO = 16    # Num of non_frauds in a batch
BATCH_SIZE = FRAUD_RATIO + NON_FRAUD_RATIO
SEQUENCE_LENGTH = 100


def import_sequences():
    """Load .npy sequences from disk"""

    non_fraud_sequences = np.load('data/modified/sequences/non_fraud_sequences.npy')
    fraud_sequences = np.load('data/modified/sequences/fraud_sequences.npy')

    return fraud_sequences, non_fraud_sequences


def import_transactions():
    """Load .npy transactions from disk"""

    transactions = np.load('data/modified/transactions/all_transactions.npy')
    transaction_labels = np.load('data/modified/transactions/all_transaction_labels.npy')

    return transactions, transaction_labels


def generate_batches(frauds: np.ndarray, non_frauds: np.ndarray):
    """Generate batchs according to a given ration defined above"""

    batch = np.empty(shape=(0, SEQUENCE_LENGTH), dtype=np.int64)
    dataset = np.empty(shape=(0, BATCH_SIZE, SEQUENCE_LENGTH), dtype=np.int64)
    labels = np.empty(shape=(0, BATCH_SIZE), dtype=np.int64)
    np.random.shuffle(non_frauds)
    np.random.shuffle(frauds)

    num_batches = 0
    sys.stdout.write("Generating batches")
    sys.stdout.flush()

    while len(frauds) > FRAUD_RATIO:
        if num_batches % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

        num_batches += 1

        batch = np.append(non_frauds[0:NON_FRAUD_RATIO],
                          frauds[0:FRAUD_RATIO], axis=0)[np.newaxis, :, :]
        dataset = np.append(dataset, batch, axis=0)

        # Remove sampled transactions
        frauds = frauds[FRAUD_RATIO:]
        non_frauds = non_frauds[NON_FRAUD_RATIO:]

        # Add labels
        batch_labels = np.append(
            np.zeros((NON_FRAUD_RATIO,), dtype=int), np.ones((FRAUD_RATIO,), dtype=int))
        labels = np.append(labels, batch_labels[np.newaxis, :], axis=0)

    print(f"\nGenerated {num_batches} batches")
    print(
        f"Generated dataset with shape {dataset.shape} and labels {labels.shape}")

    return dataset, labels


def split_dataset(dataset: np.ndarray, labels: np.ndarray, ratio: float):
    """Creat training and validation datasets"""

    data_sets = np.split(dataset, [int(ratio * len(dataset))], axis=0)
    label_sets = np.split(labels, [int(ratio * len(labels))], axis=0)

    print(f"Split training and validation sets with {data_sets[0].shape[0]} and {data_sets[1].shape[0]} batches each")
    return (data_sets[0], label_sets[0]), (data_sets[1], label_sets[1])


def convert_ids(batch: np.ndarray, transaction_list: np.ndarray):
    """Convert the IDs of transactions to actual transactions to serve as input
    to the model"""
    return transaction_list[batch]
