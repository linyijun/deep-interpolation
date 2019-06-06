import copy
import random
import numpy as np
import torch
import torch.utils.data as dat


class DataLoader:

    def __init__(self, X, y, neighbor_X, config, mode='train'):

        self.X = X
        self.y = y
        self.neighbor_X = neighbor_X
        self.config = config
        self.mode = mode

        self.feature_dim = self.X.shape[1]
        self.label_dim = self.y.shape[1]
        self.n_data = self.X.shape[0]

        print('Feature dimension: {}.'.format(self.feature_dim))
        print('Label dimension: {}.'.format(self.label_dim))
        print('Number of records/data: {}.'.format(self.n_data))

        self.mask = self.generate_mask()

        data_list = [self.X, self.y, self.neighbor_X]
        result_data_list = self.filter_nan(data_list)

        self.new_X = self.X[self.mask, ...]
        self.new_y = self.y[self.mask, ...]
        self.new_neighbor_X = self.neighbor_X[self.mask, ...]

        self.new_X, _, _ = self.standard_norm(self.new_X)
        self.new_neighbor_X, _, _ = self.standard_norm(self.new_neighbor_X)

        if self.mode == 'train':
            self.new_y, self.y_mean, self.y_std = self.standard_norm(self.new_y)

        print('The shape of input {} X: {}.'.format(mode, self.new_X.shape))
        print('The shape of input {} Y: {}.'.format(mode, self.new_y.shape))
        print('The shape of input neighboring {} X: {}.'.format(mode, self.new_neighbor_X.shape))

        self.data_loader = self.construct_data_loader()

    # construct data loader for the NN
    def construct_data_loader(self):

        x_tensor = torch.FloatTensor(self.new_X)
        y_tensor = torch.FloatTensor(self.new_y)
        neighbor_x_tensor = torch.FloatTensor(self.new_neighbor_X)

        dataset = dat.TensorDataset(x_tensor, y_tensor, neighbor_x_tensor)
        data_loader = dat.DataLoader(dataset=dataset,
                                     batch_size=self.config['model']['batch_size'],
                                     shuffle=self.config['model']['shuffle'])
        return data_loader

    def generate_mask(self):

        mask = np.ones([self.n_data, ], dtype=bool)
        mask = mask & (~np.isnan(self.X).any(axis=1).reshape(-1, ))
        tmp = self.neighbor_X.reshape(self.n_data, -1)
        mask = mask & (~np.isnan(tmp).any(axis=1).reshape(-1, ))
        return mask

    def standard_norm(self, x):
        x_tmp = x.reshape(-1, x.shape[-1])
        mean, std = np.nanmean(x_tmp, axis=0), np.nanstd(x_tmp, axis=0)
        x_norm = np.divide(x_tmp - mean, std, out=np.zeros_like(x_tmp - mean), where=std != 0.0)
        x_norm = x_norm.reshape(x.shape)
        return x_norm, mean, std


# split all available locations into training, validation, and testing locations
def split_train_val_test_locations(locations, val_ratio, test_ratio):

    train_ratio = 1 - val_ratio - test_ratio
    n_locations = len(locations)

    locations_copy = copy.copy(locations)
    random.shuffle(locations_copy)

    train_loc = locations_copy[: int(train_ratio * n_locations)]
    val_loc = locations_copy[int(train_ratio * n_locations):int((train_ratio + val_ratio) * n_locations)]
    test_loc = locations_copy[int((train_ratio + val_ratio) * n_locations):]

    return train_loc, val_loc, test_loc


def construct_input_data(data, loc_list, config, mode='train'):

    if_include_static = config['other_settings']['if_include_static']

    time_len = len(data.time_list)
    loc_len = len(loc_list)

    def concat_data(data_dict):
        data_list, feature_list = [], []
        for _, dt in data_dict.items():
            dt_locations = dt.locations
            idx = [dt_locations.index(loc) for loc in loc_list]
            data_list.append(dt.data[idx, ...])
            feature_list += dt.features
        return np.concatenate(data_list, axis=-1), feature_list

    dynamic_data, dynamic_features = concat_data(data.dynamic_data_dict)

    if if_include_static:
        static_data, static_features = concat_data(data.static_data_dict)
        static_data = np.repeat(static_data, time_len, axis=0)
        static_data = static_data.reshape(loc_len, time_len, -1)
        X = np.concatenate([dynamic_data, static_data], axis=-1)
        features = dynamic_features + static_features

    else:
        X = dynamic_data
        features = dynamic_features

    if mode == 'train':
        y, _ = concat_data(data.target_data_dict)
        return X, y, features
    else:
        return X


# construct neighbor lag data to support training data
def construct_neighbor_lag_data(data, loc_list, neighbor_grids_list_dict, config):

    id_column = data.id_column
    time_len = data.time_len

    neighbor_loc_list = []
    for loc in loc_list:
        neighbor_loc_list += neighbor_grids_list_dict[loc]
    neighbor_loc_list = list(set(neighbor_loc_list))

    neighbor_data, _, _, feature_columns = construct_input_data(data, neighbor_loc_list, config)

    neighbor_lag_step = config['other_settings']['neighbor_lag_step']
    mask = np.empty((neighbor_lag_step, len(feature_columns)))
    mask[:] = np.nan

    all_neighbor_lag_data = []

    for loc in loc_list:

        neighbor_lag_data = []
        this_loc_neighbors = neighbor_grids_list_dict[loc]

        for n in this_loc_neighbors:
            this_neighbor_data = neighbor_data[neighbor_data[id_column] == n]
            this_neighbor_data = this_neighbor_data[feature_columns].values
            this_neighbor_data = np.concatenate([mask, this_neighbor_data])
            neighbor_lag_data.append(this_neighbor_data[ : time_len])

        neighbor_lag_data = np.array(neighbor_lag_data)
        neighbor_lag_data = neighbor_lag_data.reshape(time_len, len(this_loc_neighbors), -1)
        all_neighbor_lag_data.append(neighbor_lag_data)

    all_neighbor_lag_data = np.concatenate(all_neighbor_lag_data)
    return all_neighbor_lag_data