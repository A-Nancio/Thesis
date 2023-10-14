# Train model for one epoch and save weights
import os

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from machine_learning.models import CATEOGRY_ID_COLUMN
from machine_learning.shared_state import SharedState, BATCH_SIZE
from keras.layers import concatenate, Dense, RNN, GRU
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import csv

SEQUENCE_LENGTH = 100

# IT Trains only on sequences which have only on credit card in the sequence
class Feedzaitrain(tf.keras.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=128)  
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.card_gru(inputs)
        out = self.out(var)

        return out
    
class DoubleTrain(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_gru = GRU(units=128, dropout=0.1, recurrent_dropout=0.2)
        self.category_gru = RNN(SharedState(units=128, dropout=0.1, recurrent_dropout=0.2, id_column=CATEOGRY_ID_COLUMN))
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        card_output = self.card_gru(inputs)
        category_output = self.category_gru(inputs)
        var = concatenate([card_output, category_output])
        out = self.out(var)
        return out

    def reset_gru(self):
        self.card_gru.cell.reset_states()   # training model needs to access cell
        self.category_gru.cell.reset_states()   # training model needs to access cell
    
class GlobalState(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.gru = GRU(units=128, stateful=False)
        self.out = Dense(1,  activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        var = self.gru(inputs)
        out = self.out(var)
        return out
    

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# BATCHES REDUCED TO 10 FOR SIMPLICITY
transactions = np.concatenate((np.load(f'src/data/train/transactions.npy'), np.load(f'src/data/test/transactions.npy')), axis=0)
train_seq_ids = np.load(f'src/data/train/seq_ids.npy').astype(int)#[0:10]
train_seq_labels = np.load(f'src/data/train/seq_labels.npy').astype(float)#[0:10]


model_list = [GlobalState, Feedzaitrain, DoubleTrain]
for model_class in model_list:
    model: tf.keras.Model = model_class()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[
                        tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.TruePositives(), 
                        tf.keras.metrics.TrueNegatives(),
                        tf.keras.metrics.FalsePositives(), 
                        tf.keras.metrics.FalseNegatives(),
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.F1Score()])
    
    
    with open(f'src/training_runs/{model.name}.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['fold', 'epoch', 'loss', 'binary_accuracy', 'TP', 'TN', 'FP', 'FN', 'precision', 'recall', 'f1_score']
        writer.writerow(header)

        model(np.expand_dims(transactions[train_seq_ids[0]], axis=0)) # initialize_weights
        model.save_weights('model.h5')

        count = 1
        for train_indices, val_indices in kfold.split(train_seq_ids):
            model.load_weights('model.h5') # Reset weights
            
            x_train, x_val = train_seq_ids[train_indices], train_seq_ids[val_indices]
            y_train, y_val = train_seq_labels[train_indices], train_seq_labels[val_indices]

            training_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
            validation_set = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

            for epoch in range(20):
                model.reset_metrics()

                for step, (x_batch_train, y_batch_train) in tqdm(enumerate(training_set), total=training_set.cardinality().numpy()):
                    x_batch_train = transactions[x_batch_train]
                    model.train_on_batch(x_batch_train, y_batch_train)

                for step, (x_batch_val, y_batch_val) in tqdm(enumerate(validation_set), total=validation_set.cardinality().numpy()):
                    x_batch_val = transactions[x_batch_val]
                    results = model.test_on_batch(x_batch_val, y_batch_val, reset_metrics=False)
                
                writer.writerow([count, epoch] + results)
                print(f"{model.name}: fold {count}, epoch {epoch}, results: {results}")
            count += 1


        
    

