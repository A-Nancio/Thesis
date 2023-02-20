import pandas as pd
import math
from scipy import zscore

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
    #TODO TO BE TESTED
    percentile, lower_limit = 0.01, column.min()
    while percentile <= 1:
        upper_limit = column.quantile(percentile)
        
        mask = lower_limit <= column < upper_limit
        column.loc[mask] = (percentile-0.01) * 100
        
        lower_limit = upper_limit
        percentile += 0.01

    return column#convert_categorical_feat(dataset, features, 25) #max_index proposed is 25 since there are 100 buckets

def convert_categorical_feat(column: pd.Series, max_index: int):
    """Converts all possible values of a categorical feature by mapping each
    possible value into a integer based on the number of occurrences
    """
    #TODO BE TESTED
    occurence_list = column.value_counts()
    l = 1
    for index, value in occurence_list.items():
        column = column.replace(index, l)
        if l < max_index:
            l = l + 1

    return column


def convert_time(feature_column, format: int): 
    """Transform hour feature and transform into the sine and 
    cosine of its projection

    format: indicator if to convert either hour of the day, 
    day of the week or the day of the month
    """
    #TODO BE TESTED
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

