from preprocess.data_loader import *
from preprocess.data_reader import *
from lib.spatial_neighbors_generator import *

import os

def main():

    # define file path
    base_path = '/Users/yijunlin/JupyterNotebook/data/beijing/'
    resolution = 1000

    # import configuration
    config = load_config('/Users/yijunlin/PycharmProjects/deep-interpolation/data/model_config.json')

    # for testing and debugging
    config['target']['file_name'] = os.path.join(base_path, 'beijing_{}m_grid_aq.csv'.format(resolution))
    config['dynamic']['meo']['file_name'] = os.path.join(base_path, 'beijing_{}m_grid_meo.csv'.format(resolution))
    config['static']['geo']['file_name'] = os.path.join(base_path, 'beijing_{}m_grid_geo_feature.csv'.format(resolution))
    config['static']['coord']['file_name'] = os.path.join(base_path, 'beijing_{}m_grid_coord.csv'.format(resolution))

    # read data
    data = DataReader(config)
    print()

    # add lag features for meo
    lag_offsets = config['other_settings']['lag_offsets']
    meo_dd = data.dynamic_data_dict['meo']
    print('>>> Generating lag features for meo data...')
    lag_feature_vector, lag_features = construct_lag_feature_vector(meo_dd.data, meo_dd.features, lag_offsets)
    meo_lag_dd = DataDict(lag_feature_vector, 'meo_lag', meo_dd.locations, lag_features)
    data.dynamic_data_dict['meo_lag'] = meo_lag_dd
    print()

    # generate spatial neighboring grids
    neighbor_step = config['other_settings']['n_neighbor_step']
    coord_dd = data.static_data_dict['coord']
    print('>>> Generating the neighbours for {} grids.'.format(len(coord_dd.locations)))
    neighbor_grids_dict, neighbor_grids_list = get_neighboring_grids(coord_dd.data, coord_dd.locations, neighbor_step)
    print()

    # select training locations and testing locations
    locations = data.target_data_dict['aq'].locations
    train_loc, _, test_loc = split_train_val_test_locations(locations, val_ratio=0.0, test_ratio=0.2)
    print('Training locations: {}'.format(train_loc))
    print('Testing locations: {}'.format(test_loc))
    print()

    train_X, train_y, features = construct_input_data(data, train_loc, config)
    print()


if __name__ == '__main__':
    main()