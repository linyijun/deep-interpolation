import copy

import datetime
import pandas as pd
import numpy as np


def construct_time_series_vector():
    pass


def construct_lag_feature_vector(data, features, lag_offsets):
    """
    Construct the lag feature vector, uausally lag offsets is 2 or 3

    """

    # define new feature names with lag information
    lag_features = []
    for i in range(lag_offsets):
        lag_features += ['{}_lag_{}'.format(f, lag_offsets - i) for f in features]

    n_data, n_times, n_features = data.shape[0], data.shape[1], data.shape[2]

    # define an empty mask
    mask = np.empty((n_data, lag_offsets, n_features))
    mask[:] = np.nan
    lag_data = np.concatenate([mask, data], axis=1)

    x_list = []
    for i in range(lag_offsets, n_times + lag_offsets):
        x = lag_data[:, (i - lag_offsets): i, :]
        x = x.reshape(n_data, -1)
        x_list.append(x)

    lag_feature_vector = np.stack(x_list, axis=1)
    return lag_feature_vector, lag_features


def construct_geo_feature_vector(geo_data, column_set):
    """
    Construct the raw geographic data into geo feature vector

    """

    id_column = column_set[0]
    feature_type_column = column_set[1]
    geo_feature_column = column_set[2]
    value_column = column_set[3]

    locations = list(geo_data[id_column].drop_duplicates())

    geo_feature_type_data = copy.copy(geo_data)
    geo_feature_type_column = 'feature_name'
    geo_feature_type_data[geo_feature_type_column] = geo_feature_type_data[geo_feature_column] + '_' \
                                                     + geo_feature_type_data[feature_type_column]

    feature_vector_list = []

    for loc in locations:
        loc_geo_data = geo_feature_type_data[geo_feature_type_data[id_column] == loc]
        loc_geo_data = loc_geo_data.set_index(geo_feature_type_column)[value_column]
        feature_vector_list.append(loc_geo_data)

    feature_vector = pd.concat(feature_vector_list, axis=1)
    feature_vector = feature_vector.fillna(0.0)

    geo_feature_data = feature_vector.values.T
    geo_feature_types = list(feature_vector.index)

    return geo_feature_data, locations, geo_feature_types


