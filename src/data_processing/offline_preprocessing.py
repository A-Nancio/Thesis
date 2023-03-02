import pandas as pd
import math
import numpy as np
from scipy.stats import zscore

#TODO function to have time delta between the last transaction of a given client(card)

def convert_bucket_feat(column: pd.Series):
    """Applies percentile bucketing to a panda Series
    """
    initial_column = column.copy()  #without it the quantiles are wrongly calculated

    #NOTE verify if values are placed onto the right buckets
    #NOTE NEED TO STORE THE QUNATILE VALUES FOR TRANSFORMATION OF TEST DATASET
    
    lower_quantile = 0
    for upper_quantile in np.arange(0.01, 1.01, 0.01):
        lower_limit = initial_column.quantile(lower_quantile)
        upper_limit = initial_column.quantile(upper_quantile)
        
        column.loc[(column >= lower_limit) & (column <= upper_limit)] = (upper_quantile-0.01) * 100
        lower_quantile = upper_quantile

    column.fillna(100)
    return column#convert_categorical_feat(dataset, features, 25) #max_index proposed is 25 since there are 100 buckets

def convert_categorical_feat(column: pd.Series, max_index: int):
    """Converts all possible values of a categorical feature by mapping each
    possible value into a integer based on the number of occurrences
    """
    #NOTE NEED TO STORE THE VALUE COUNTS FOR TRANSFORMATION OF TEST DATASET
    occurence_list = column.value_counts()
    l = 1
    for index, value in occurence_list.items():
        column = column.replace(index, l)
        if l < max_index:
            l = l + 1

    return column


def time_projection(feature_column, format: int): 
    """Transform hour feature and transform into the sine and 
    cosine of its projection

    format: indicator if to convert either hour of the day, 
    day of the week or the day of the month
    """
    sin_projection = feature_column.apply(lambda x: math.sin(x *(2 * math.pi / format)))
    cos_projection = feature_column.apply(lambda x: math.cos(x *(2 * math.pi / format)))

    return sin_projection, cos_projection

def zscore_clipping(feature_column: pd.Series):
    """Z-scoring with outlier clipping for features with distributions
    that are not very skewed"""
    feature_column = pd.Series(zscore(feature_column.to_numpy()))
    feature_column = feature_column.apply(lambda x: max(min(x, 3), -3))
    return feature_column
