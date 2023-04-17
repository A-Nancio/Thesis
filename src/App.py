import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import import_sequences, import_transactions, generate_batches, split_dataset, convert_ids
from machine_learning.models import GlobalState
from machine_learning.train import train_model, validate_model

fraud_sequences, non_fraud_sequences = import_sequences()
transactions, transaction_labels = import_transactions()

full_data, labels = generate_batches(fraud_sequences, non_fraud_sequences)
training_set, validation_set = split_dataset(full_data, labels, ratio=.8)

# clean data for memory saving
del fraud_sequences, non_fraud_sequences, full_data, labels


model = GlobalState(transactions.shape[1])

NUM_EPOCHS = 20

loss_data = ()
for epoch in range(NUM_EPOCHS):
    print(f"\nStart of epoch {epoch}")

    training_data = convert_ids(training_set[0], transactions)
    train_loss, epoch_time = train_model(model, training_data, training_set[1])
    loss_data += (train_loss,)

    validation_data = convert_ids(validation_set[0], transactions)
    validation_results = validate_model(model, validation_data, validation_set[1])
