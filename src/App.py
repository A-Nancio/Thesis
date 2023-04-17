from data_processing.batch_generator import import_sequences, import_transactions, generate_batches, split_dataset, convert_ids
from machine_learning.models import GlobalState

fraud_sequences, non_fraud_sequences = import_sequences()
transactions, transaction_labels = import_transactions()

full_data, labels = generate_batches(fraud_sequences, non_fraud_sequences)
training_set, validation_set = split_dataset(full_data, labels, ratio=.8)

# clean data for memory saving
del fraud_sequences, non_fraud_sequences, full_data, labels

model = GlobalState(transactions.shape[1])
