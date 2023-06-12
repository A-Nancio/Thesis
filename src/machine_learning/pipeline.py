"""Tensorflow components to be used"""
import tensorflow as tf
from keras import metrics
import sys

MAX_EPOCHS=3
metric_names = ['loss', 'binary_accuracy', 'true_positives', 
                'true_negatives', 'false_positives', 'false_negatives']

def compile_model(model: tf.keras.Model, input: tf.data.Dataset):
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[metrics.BinaryAccuracy(),
                metrics.TruePositives(), metrics.TrueNegatives(),
                metrics.FalsePositives(), metrics.FalseNegatives()])
  
  # initialize model weights by providing an initial input
  model(list(input.take(1))[0][0])

def fit_model(model, train_data: tf.data.Dataset, class_weights, model_name):
    checkpoint_path = 'src/machine_learning/saved_models/Feedzai.ckpt'
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    history = model.fit(train_data, 
                        epochs=MAX_EPOCHS, 
                        class_weight=class_weights, 
                        verbose='auto', 
                        shuffle=True,
                        callbacks=[cp_callback])
    return history

def fit_cycle(training_model, production_model, 
              train_dataset: tf.data.Dataset, pre_test_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, 
              class_weights: dict):
    """ Perform one training and validation cycle """
    training_model.reset_gru()
    sys.stdout.write("\tTrain: ")
    train_results = list(training_model.fit(train_dataset, 
                        epochs=1, 
                        class_weight=class_weights, 
                        verbose='auto', 
                        shuffle=True).history.values())
    
    
    production_model.set_weights(training_model.get_weights())
    production_model.reset_gru()
    sys.stdout.write("\tLoad state: ")
    production_model.evaluate(pre_test_dataset, batch_size=1)

    sys.stdout.write("\tTest: ")
    evaluation_results = production_model.evaluate(test_dataset, batch_size=1)

    return train_results, evaluation_results
