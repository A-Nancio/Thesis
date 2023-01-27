from genericpath import exists
import pandas as pd
import matplotlib.pyplot as plt
from zmq import has
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib as mpl
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from buildBatches import *
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from timeit import default_timer as timer
import copy

isTest = False



# mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

cutoff_lenght = 5
batch_size = 1

#one dict for each gru layer
states_dict = {}
#states_dict1 = {}
#states_dict2 = {}
#states_dict3 = {}

"""option 1 - build batches from datasets"""
#result,validation,test = buildBatches(batch_size,cutoff_lenght)

"""option 2 - read batches previously saved"""
#result, validation, test = readSmallBatches()
result, validation, test = readBatches()


n_batches = int(result.shape[0]/(cutoff_lenght*batch_size))


y_original = np.array(result['fraud'], dtype='float')

result = result.iloc[: , 1:]
#result.drop(['fraud'], inplace=True, axis=1)
result = result.drop(columns= ['customer', 'age', 'gender', 'merchant', 'category','step'])
print(result.shape)
n_column = len(result.columns)
def batchTreat(result):  
    

    aux = int(result.shape[0]/cutoff_lenght)



    X_interim = np.zeros([n_batches*batch_size*cutoff_lenght,n_column]) 

    print(X_interim.shape)
    print(X_interim[0])
    y = []

    for i in range(aux):
        beg = i*cutoff_lenght
        end = beg+cutoff_lenght
        result.at[end-1, 'fraud'] = -1
        lastNA = result[beg:end].copy()
        s = np.array(lastNA, dtype='float')
        if i == 0:
            print(lastNA)
        
        X_interim[((i)*cutoff_lenght):((i+1)*cutoff_lenght),:] = s
        #print((i)*aux_batch_size+cutoff_lenght*(j+1))
        y.append(y_original[end-1])
        #print(i)

    y = np.array(y, dtype='float')
    print(np.shape(X_interim))


    print('x int shape:' + str(X_interim.shape))
    print('y size', len(y))
    print(y)
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        X_interim, 
        y_original[cutoff_lenght-1:],
        sequence_length=cutoff_lenght,
        sampling_rate = 1,
        shuffle = False,
        sequence_stride = cutoff_lenght,
        #batch_size = 1
        batch_size=batch_size
    )
    return dataset, X_interim, result, y_original

dataset_train, X_interim, result_finish, y_original = batchTreat(result)

customersPerBatch = []

i = 0
for pack in dataset_train:
    inputs, targets = pack
    customer = inputs[0][0][-1].numpy()
    customersPerBatch.append(customer)
    i = i +1

#y = np.array(y, dtype='float')
print(np.shape(X_interim))
#X_interim = X_interim[:,1::]
print(np.shape(X_interim))



n_batches_Val = int(validation.shape[0]/(cutoff_lenght*batch_size))
validation = validation.iloc[: , 1:]
validation = validation.drop(columns= ['customer', 'age', 'gender', 'merchant', 'category', 'step'])

y_original_Val = np.array(validation['fraud'], dtype='float')

#validation.drop(['fraud'], inplace=True, axis=1)


aux_Val = int(validation.shape[0]/cutoff_lenght)
X_interim_Val = np.zeros([(n_batches_Val*batch_size*cutoff_lenght),n_column])

y_Val = []
for i in range(aux_Val):
    beg = i*cutoff_lenght
    end = beg+cutoff_lenght
    validation.at[end-1, 'fraud'] = -1
    lastNA = validation[beg:end].copy()
    s = np.array(lastNA, dtype='float')
    if i == 0:
        print(lastNA)
    
    X_interim_Val[((i)*cutoff_lenght):((i+1)*cutoff_lenght),:] = s
    #print((i)*aux_batch_size+cutoff_lenght*(j+1))
    y_Val.append(y_original_Val[end-1])
    #print(i)
    

print('y_val', len(y_Val))
#print(y_Val)
dataset_Val = tf.keras.utils.timeseries_dataset_from_array(
    X_interim_Val, 
    y_original_Val[cutoff_lenght-1:],
    sequence_length=cutoff_lenght,
    sampling_rate = 1,
    shuffle = False,
    sequence_stride = cutoff_lenght,
    #batch_size = 1
    batch_size=batch_size
)

customersPerBatch_Val = []

for pack in dataset_Val:
    inputs, targets = pack
    customer = inputs[0][0][-1].numpy()
    customersPerBatch_Val.append(customer)
    i = i +1


#############################
THRESHOLD = 0.05
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),#, thresholds=THRESHOLD),
      keras.metrics.Recall(name='recall'),#, thresholds=THRESHOLD),
      #keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def create_model(inputs):

    rnn_cells = [tf.keras.layers.LSTMCell(128) for _ in range(8)]
    stacked_gru = tf.keras.layers.StackedRNNCells(rnn_cells)
    gru_layer = tf.keras.layers.RNN(stacked_gru, stateful = True)


    inputs = keras.Input(batch_shape = (batch_size, inputs.shape[1], inputs.shape[2]))
    #print(inputs)
    #z = keras.layers.RNN(tf.keras.layers.GRUCell(2)) (inputs)
    #drop = tf.keras.layers.Dropout(0.2)(inputs)
    z = keras.layers.GRU(128, dropout=0.2, recurrent_regularizer='l2', stateful= True) (inputs, training=True)
    #z3 = keras.layers.GRU(128,return_sequences=True, dropout=0.3, recurrent_regularizer='l2', stateful= True) (z, training=True)
    #z4 = keras.layers.GRU(128,return_sequences=True, dropout=0.3, recurrent_regularizer='l2', stateful= True) (z3, training=True)
    #z2 = keras.layers.GRU(128, dropout=0.3, recurrent_regularizer='l2', stateful = True) (z4, training=True)
    # pp = tf.keras.layers.GRU(128, stateful = True) (z)
    #z = gru_layer(inputs)
    #concat = tf.keras.layers.Concatenate()([inputs, z2])
    #output = tf.keras.layers.Dense(1, activation='tanh')(z2)
    #softmax = tf.keras.layers.Softmax(output)
    #output4 = tf.keras.layers.Dense(8, activation='sigmoid')(z2)
    output3 = tf.keras.layers.Dense(24)(z)
    
    output2 = tf.keras.layers.Dense(1)(output3)
    #acti = tf.keras.activations.softmax(output2, axis=-1)

    acti = tf.keras.layers.Activation("sigmoid")(output2)
    #print(output.shape)

    

    model1 = keras.Model(inputs=[inputs],outputs=[acti])
    #model2 = keras.Model(inputs=[inputs],outputs=[rnn,sh,sc])

    return model1

learning_rate = 0.001
model1 = create_model(inputs)
keras.utils.plot_model(model1, "model1.png", show_shapes=True)

model1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    #loss = 'mean_squared_error',
    loss=tf.keras.losses.BinaryCrossentropy(),#from_logits=True),
    metrics = METRICS
)
model1.summary()

"""callbacks"""
class StatePerCard(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, data_layer):
        super(StatePerCard, self).__init__()
        self.customer = None
        self.data = data_layer[0]
        self.data_val = data_layer[1]
        self.state = None
        # self.state1 = None
        # self.state2 = None
        # self.state3 = None

    def on_batch_begin(self, batch, logs=None ):
        """ 1. Get customer id
            2. Get customer state from dict
            3. Load state of the customer into model"""
        #get customer
        gru_layer = self.model.get_layer('gru')
        # gru_layer1 = self.model.get_layer('gru_1')
        # gru_layer2 = self.model.get_layer('gru_2')
        # gru_layer3 = self.model.get_layer('gru_3')
        self.customer = self.data[batch]
        global states_dict
        # global states_dict1
        # global states_dict2
        # global states_dict3

        
        #print("on batch  begin")
        if self.customer in states_dict:
            # print(type(states_dict[self.customer][0]))
            # print("estado no dicionario")
            # print(states_dict[self.customer][0])
            self.state = copy.deepcopy(states_dict[self.customer])
            gru_layer.reset_states(states = self.state[0].numpy())

            # #gru n2
            # self.state1 = copy.deepcopy(states_dict1[self.customer])
            # gru_layer1.reset_states(states = self.state1[0].numpy())

            # #gru n3
            # self.state2 = copy.deepcopy(states_dict2[self.customer])
            # gru_layer2.reset_states(states = self.state2[0].numpy())

            # #gru n4
            # self.state3 = copy.deepcopy(states_dict3[self.customer])
            # gru_layer3.reset_states(states = self.state3[0].numpy())
            
        else:
            #reset state and store it
            gru_layer.reset_states()
            # gru_layer1.reset_states()
            # gru_layer2.reset_states()
            # gru_layer3.reset_states()
        
    
        
    def on_batch_end(self, batch, logs=None):
        """ 1. Get customer id
            2. Get customer state from model
            3. Store state of the customer into dict"""
        #get customer
        self.customer = self.data[batch]
        global states_dict
        # global states_dict1
        # global states_dict2
        # global states_dict3

        gru_layer = self.model.get_layer('gru')
        # gru_layer1 = self.model.get_layer('gru_1')
        # gru_layer2 = self.model.get_layer('gru_2')
        # gru_layer3 = self.model.get_layer('gru_3')
        

        
        if isTest:
            
            if self.customer in states_dict:
                print("\n on batch  end")
                #print(gru_layer.states)
                print("customer in dict")
                #print(states_dict[self.customer])
                print(np.array_equal(states_dict[self.customer][0].numpy(),gru_layer.states[0].numpy()))
                if np.array_equal(states_dict[self.customer][0].numpy(),gru_layer.states[0].numpy()):
                    print(gru_layer.states)
                
                    print(states_dict[self.customer])
                #print(states_dict[self.customer]==gru_layer.states)
            
        states_dict[self.customer] = copy.deepcopy(gru_layer.states)
        # states_dict1[self.customer] = copy.deepcopy(gru_layer1.states)
        # states_dict2[self.customer] = copy.deepcopy(gru_layer2.states)
        # states_dict3[self.customer] = copy.deepcopy(gru_layer3.states)
        

    def on_test_batch_begin(self, batch, logs=None):
        """ 1. Get customer id
            2. Get customer state from dict
            3. Load state of the customer into model"""
         #get customer
        gru_layer = self.model.get_layer('gru')
        # gru_layer1 = self.model.get_layer('gru_1')
        # gru_layer2 = self.model.get_layer('gru_2')
        # gru_layer3 = self.model.get_layer('gru_3')
        self.customer = self.data_val[batch]
        global states_dict
        # global states_dict1
        # global states_dict2
        # global states_dict3

        
        #print("on batch  begin")
        if self.customer in states_dict:
            # print(type(states_dict[self.customer][0]))
            # print("estado no dicionario")
            # print(states_dict[self.customer][0])
            self.state = copy.deepcopy(states_dict[self.customer])
            gru_layer.reset_states(states = self.state[0].numpy())

            # #gru n2
            # self.state1 = copy.deepcopy(states_dict1[self.customer])
            # gru_layer1.reset_states(states = self.state1[0].numpy())

            # #gru n3
            # self.state2 = copy.deepcopy(states_dict2[self.customer])
            # gru_layer2.reset_states(states = self.state2[0].numpy())

            # #gru n4
            # self.state3 = copy.deepcopy(states_dict3[self.customer])
            # gru_layer3.reset_states(states = self.state3[0].numpy())
            
        else:
            #reset state and store it
            gru_layer.reset_states()
            # gru_layer1.reset_states()
            # gru_layer2.reset_states()
            # gru_layer3.reset_states()
        


    def on_test_batch_end(self, batch, logs=None):
        """ 1. Get customer id
            2. Get customer state from model
            3. Store state of the customer into dict"""
        #get customer
        self.customer = self.data_val[batch]
        global states_dict
        # global states_dict1
        # global states_dict2
        # global states_dict3

        gru_layer = self.model.get_layer('gru')
        # gru_layer1 = self.model.get_layer('gru_1')
        # gru_layer2 = self.model.get_layer('gru_2')
        # gru_layer3 = self.model.get_layer('gru_3')
        

        
        if isTest:
            
            if self.customer in states_dict:
                # print("\n on batch  end")
                # print(gru_layer.states)
                # print("customer in dict")
                #print(states_dict[self.customer])
                print(np.array_equal(states_dict[self.customer][0].numpy(),gru_layer.states[0].numpy()))
                if np.array_equal(states_dict[self.customer][0].numpy(),gru_layer.states[0].numpy()):
                    print(gru_layer.states)
                
                    print(states_dict[self.customer])
                #print(states_dict[self.customer]==gru_layer.states)
            
        states_dict[self.customer] = copy.deepcopy(gru_layer.states)
        # states_dict1[self.customer] = copy.deepcopy(gru_layer1.states)
        # states_dict2[self.customer] = copy.deepcopy(gru_layer2.states)
        # states_dict3[self.customer] = copy.deepcopy(gru_layer3.states)
        
        
    # def on_train_begin(self, batch, logs=None):


    # def on_epoch_end(self, epoch, logs=None):


    # def on_train_end(self, logs=None):



es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=20)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10)

checkpoint_filepath = '/tmp/StateperCard1F_v1_seq.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

csvLog = tf.keras.callbacks.CSVLogger("training_log_1F_StateperCard_v1_seq.csv", 
                             separator=',', 
                             append=False)

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()

print(dataset_Val.take(1))
for batch in dataset_Val.take(1):
    inputs2, targets2 = batch
    print("Input val shape:", inputs2.numpy().shape)

cp = ModelCheckpoint(filepath="fortest.h5",
                               save_best_only=True,
                               verbose=0)


tensorboard_callback = TensorBoard(log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)

#isTest = True

history = model1.fit(
    dataset_train,
    #dataset_train,
    epochs=200, 
    validation_data = dataset_Val,
    #batch_size= (card_Seq-cutoff_lenght)+batch_size,
    callbacks = [es_callback, reduce_lr, cb, model_checkpoint_callback, csvLog, StatePerCard([customersPerBatch, customersPerBatch_Val])],
    class_weight = {0 : 1., 1: 10.},
    #verbose = 2
    #shuffle = True
)


plt.plot(history.history['loss'], linewidth=2, label='Train')
plt.plot(history.history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(ymin=0.70,ymax=1)
#plt.show()
plt.savefig("StateperCard1F_v1_seq.png")

"""test phase"""




n_batches_test = int(test.shape[0]/(cutoff_lenght*batch_size))
test = test.iloc[: , 1:]
test = test.drop(columns= ['customer', 'age', 'gender', 'merchant', 'category', 'step'])

y_original_test = np.array(test['fraud'], dtype='float')

#test.drop(['fraud'], inplace=True, axis=1)

aux_test = int(test.shape[0]/cutoff_lenght)

X_interim_test = np.zeros([(n_batches_test*batch_size*cutoff_lenght),n_column])

y_test = []
for i in range(aux_test):
    beg = i*cutoff_lenght
    end = beg+cutoff_lenght
    test.at[end-1, 'fraud'] = -1
    lastNA = test[beg:end].copy()
    s = np.array(lastNA, dtype='float')
    
    X_interim_test[((i)*cutoff_lenght):((i+1)*cutoff_lenght),:] = s
    #print((i)*aux_batch_size+cutoff_lenght*(j+1))
    y_test.append(y_original_test[end-1])
    #print(i)
    

#y = np.array(y, dtype='float')

print('y_test', len(y_test))
#print(y_test)
dataset_test = tf.keras.utils.timeseries_dataset_from_array(
    X_interim_test,
    #None, 
    y_original_test[cutoff_lenght-1:],
    sequence_length=cutoff_lenght,
    sampling_rate = 1,
    shuffle = False,
    sequence_stride = cutoff_lenght,
    #batch_size = 1
    batch_size=batch_size
)

test_features = tf.keras.utils.timeseries_dataset_from_array(
    X_interim_test,
    None, 
    #y_original_test[cutoff_lenght-1:],
    sequence_length=cutoff_lenght,
    sampling_rate = 1,
    shuffle = False,
    sequence_stride = cutoff_lenght,
    #batch_size = 1
    batch_size=batch_size
)

test_labels = tf.keras.utils.timeseries_dataset_from_array(
    #X_interim_test,
    y_original_test[cutoff_lenght-1:],
    None,
    sequence_length=1,
    sampling_rate = 1,
    shuffle = False,
    sequence_stride = cutoff_lenght,
    #batch_size = 1
    batch_size=batch_size
)
real_test_labels = []
for i in range(0, len(y_original_test[cutoff_lenght-1:]), 5):
  real_test_labels.append((y_original_test[cutoff_lenght-1:])[i])

train_features = tf.keras.utils.timeseries_dataset_from_array(
    X_interim,
    None, 
    #y_original_test[cutoff_lenght-1:],
    sequence_length=cutoff_lenght,
    sampling_rate = 1,
    shuffle = False,
    sequence_stride = cutoff_lenght,
    #batch_size = 1
    batch_size=batch_size
)

customersPerBatch_test = []

for pack in dataset_test:
    inputs, targets = pack
    customer = inputs[0][0][-1].numpy()
    customersPerBatch_test.append(customer)
    i = i +1

print("this is the interseption")
a = set(customersPerBatch)
print(len(a.intersection(set(customersPerBatch_test))))

#isTest = True

#train_predict = model1.predict(X_train)
#test_predict = model1.predict(dataset_train[0])
model1.load_weights(checkpoint_filepath)
test_scores = model1.evaluate(dataset_test, callbacks=[csvLog,StatePerCard([customersPerBatch, customersPerBatch_test])])

print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

print(cb.logs)
print(sum(cb.logs))

test_predictions_baseline = model1.predict(test_features, callbacks=StatePerCard([customersPerBatch, customersPerBatch_test]))

#print(type(test_predictions_baseline))
np.savetxt("foo.csv", test_predictions_baseline, delimiter=",")

### test AUC ###
from sklearn import metrics 
test_np = np.stack(list(test_labels))
# fpr, tpr, thresholds = metrics.roc_curve(y_train, train_predict, pos_label=1)
# print('TRAIN | AUC Score: ' + str((metrics.auc(fpr, tpr))))
#fpr, tpr, thresholds = metrics.roc_curve(test_np, test_predictions_baseline, pos_label=1)
#print('TEST | AUC Score: ' + str((metrics.auc(fpr, tpr))))
#divideDatasetTrain()
#Train_Validation_Test_Split()

def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend();

plot_metrics(history)

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions >p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))

def plot_roc_auc(y_test, preds):
    '''
    Takes actual and predicted(probabilities) as input and plots the Receiver
    Operating Characteristic (ROC) curve
    '''
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

for name, value in zip(model1.metrics_names, test_scores):
  print(name, ': ', value)
print()

print("this is the test labels")
print(test_labels)
print("this is the test predictions baseline")
print(test_predictions_baseline)
plot_cm(real_test_labels, test_predictions_baseline)

