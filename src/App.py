import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_processing.batch_generator import load_train_set, load_test_set, load_pre_data, get_class_weights
from machine_learning.models import FeedzaiTrain, FeedzaiProduction, BATCH_SIZE
from machine_learning.pipeline import fit_cycle, compile_model, MAX_EPOCHS
from pandas import DataFrame

SEQUENCE_LENGTH = 100

train_model = FeedzaiTrain()
production_model = FeedzaiProduction()

train_set = load_train_set(sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
pre_test_data = load_pre_data()
test_set = load_test_set()

# Metrics to store
train_metrics = {'loss': [], 'binary_accuracy': [], 'true_positives': [], 
               'true_negatives': [], 'false_positives': [], 'false_negatives': []}

val_metrics = {'loss': [], 'binary_accuracy': [], 'true_positives': [], 
               'true_negatives': [], 'false_positives': [], 'false_negatives': []}

for epoch in range(MAX_EPOCHS):
    print(f"[EPOCH {epoch}]")
    compile_model(train_model, train_set)
    compile_model(production_model, test_set)
    train_results, val_results = fit_cycle(training_model=train_model, 
              production_model=production_model,
              train_dataset=train_set,
              pre_test_dataset=pre_test_data,
              test_dataset=test_set,
              class_weights=get_class_weights()
              )
    
    for metric, i in zip(train_metrics.keys(), range(len(train_results))):
        train_metrics[metric].append(train_results[i][0])
        val_metrics[metric].append(val_results[i])


train_metrics = DataFrame.from_dict(train_metrics)
val_metrics = DataFrame.from_dict(val_metrics)

train_metrics.to_csv(f"analysis/results/{train_model.name}_results.csv")
val_metrics.to_csv(f"analysis/results/{production_model.name}_results.csv")


    
    


#mpl.rcParams['figure.figsize'] = (25, 6)
#mpl.rcParams['axes.grid'] = False
#
#plt.title("Single shared state model Evaluation") 
#plt.xlabel("input indexes") 
#plt.ylabel("predictions (degrees celcius)") 
#
#plt.plot(indexes, new, label = "sequence length 1") 
#plt.legend()
#plt.show()
#TODO store results