import pandas as pd
import math
import numpy as np

#NOTE REDUNDANT FUNCTION, CAN BE DONE IN A SINGLE LINE
#def convert_numerical_feat(column: pd.Series, features):
#    """Applies z-score scaling to selected features of the dataset
#    """
#    for feat in features:
#        dataset[feat].apply(zscore)
#    return dataset

#TODO function to have time delta between the last transaction
def convert_bucket_feat(column: pd.Series):
    """Applies percentile bucketing to a panda Series
    """
    initial_column = column.copy()  #without it the quantiles are wrongly calculated

    #NOTE verify if values are placed onto the right buckets
    lower_quantile = 0
    for upper_quantile in np.arange(0.01, 1.01, 0.01):
        lower_limit = initial_column.quantile(lower_quantile)
        upper_limit = initial_column.quantile(upper_quantile)
        
        column[(column >= lower_limit) & (column <= upper_limit)] = (upper_quantile-0.01) * 100
        lower_quantile = upper_quantile

    column.fillna(100)
    return column#convert_categorical_feat(dataset, features, 25) #max_index proposed is 25 since there are 100 buckets

def convert_categorical_feat(column: pd.Series, max_index: int):
    """Converts all possible values of a categorical feature by mapping each
    possible value into a integer based on the number of occurrences
    """
    occurence_list = column.value_counts()
    l = 1
    for index, value in occurence_list.items():
        column = column.replace(index, l)
        l = l + 1

    return column


def convert_time(feature_column, format: int): 
    """Transform hour feature and transform into the sine and 
    cosine of its projection

    format: indicator if to convert either hour of the day, 
    day of the week or the day of the month
    """
    sin_projection = feature_column.apply(lambda x: math.sin(x *(2 * math.pi / format)))
    cos_projection = feature_column.apply(lambda x: math.cos(x *(2 * math.pi / format)))

    return sin_projection, cos_projection

if __name__ == '__main__':
    print("beginning")
    #TODO process each dataset
        #TODO if a dataset contains a timestamp, add the time delta feature between entities
        # 
    #dataset_list = json.loads(open('../data/dataset_list.json', "r").read())

    #print(dataset_list)

    #for dataset in dataset_list:
    #    # process each datasetd
    #    # Process training and test set
    #    print(dataset)

