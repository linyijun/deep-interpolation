from preprocess.feature_constructor import *

import json


class DataDict:

    def __init__(self, data, data_name, locations, features):

        assert len(data), 'The {} data is empty'.format(data_name)

        self.data = data
        self.data_name = data_name
        self.locations = locations
        self.features = features


class DataReader:

    def __init__(self, config):

        self.config = config

        assert config.get('time_range') is not None, 'time_range should be defined'
        self.time_range = config['time_range']
        self.time_list = self.get_time_list()

        self.target_config = self.config['target']
        self.dynamic_config = self.config['dynamic']
        self.static_config = self.config['static']

        self.target_data_dict = self.load_target_data()
        self.dynamic_data_dict = self.load_dynamic_data()
        self.static_data_dict = self.load_static_data()

        # other configuration settings
        self.id_column = self.target_config['column_set'][0]
        self.time_column = self.target_config['column_set'][1]

        # add missing locations to geo feature data
        self.add_missing_geo_feature_vector()

    def load_target_data(self):
        target_data_dict = {}
        data_name = self.target_config['name']
        data, locations, label_name = load_time_series_data_from_csv(data_name, self.target_config, self.time_list,
                                                                     fill_missing_timestamp=True)
        target_data_dict[data_name] = DataDict(data, data_name, locations, label_name)
        return target_data_dict

    def load_dynamic_data(self):
        dynamic_data_dict = {}
        for data_name in self.dynamic_config['names']:
            data_config = self.dynamic_config[data_name]
            data, locations, features = load_time_series_data_from_csv(data_name, data_config, self.time_list)
            dynamic_data_dict[data_name] = DataDict(data, data_name, locations, features)
        return dynamic_data_dict

    def load_static_data(self):
        static_data_dict = {}
        for data_name in self.static_config['names']:
            data_config = self.static_config[data_name]
            data, locations, features = load_static_data_from_csv(data_name, data_config)
            static_data_dict[data_name] = DataDict(data, data_name, locations, features)
        return static_data_dict

    def get_time_list(self):
        min_time = self.time_range[0]
        max_time = self.time_range[1]
        time_list = pd.date_range(start=min_time, end=max_time, closed='left', freq='1H')
        return time_list

    def add_missing_geo_feature_vector(self):
        geo_feature_data = self.static_data_dict['geo'].data
        locations = self.static_data_dict['geo'].locations
        full_locations = self.static_data_dict['coord'].locations
        missing_locations = [loc for loc in full_locations if loc not in locations]
        self.static_data_dict['geo'].locations = locations + missing_locations

        missing_geo_feature_vector = np.zeros((len(missing_locations), geo_feature_data.shape[1]), dtype=float)
        new_geo_feature_data = np.concatenate([geo_feature_data, missing_geo_feature_vector], axis=0)
        self.static_data_dict['geo'].data = new_geo_feature_data


def load_static_data_from_csv(name, config):

    print(">>> Loading {} data...".format(name))

    assert config.get('file_name') and config.get('column_set'), 'file_name and column_set should be defined'
    file_name, column_set = config['file_name'], config['column_set']

    assert config.get('features'), 'should provide features for the static data'
    features = config['features']

    id_column = column_set[0]
    data = load_csv_file(file_name, column_set)

    if name == 'coord':
        data = data.round({'lon': 12, 'lat': 12})
        locations = list(data[id_column])
        data = data[features].values

    elif name == 'geo':
        geo_data = load_geo_data(data, features, config)
        data, locations, features = construct_geo_feature_vector(geo_data, column_set)

    else:
        locations = list(data[id_column])
        data = data[features].values

    return data, locations, features


def load_geo_data(geo_data, features, config):

    column_set = config['column_set']
    feature_type_column = column_set[1]  # this should be fixed
    geo_feature_column = column_set[2]   # this should be fixed

    geo_data = geo_data[geo_data[geo_feature_column].isin(features)]

    exempt_feature_types = config['exempt_feature_types'] if config.get('exempt_feature_types') else []
    if len(exempt_feature_types) > 0:
        geo_data = geo_data[~geo_data[feature_type_column].isin(exempt_feature_types)]

    return geo_data


def load_time_series_data_from_csv(name, config, times, fill_missing_timestamp=False):

    print(">>> Loading {} data...".format(name))

    assert config.get('file_name') and config.get('column_set'), 'file_name and column_set should be defined'
    file_name, column_set = config['file_name'], config['column_set']

    assert config.get('features'), 'should provide features for the time series data'
    features = config['features']

    id_column = column_set[0]  # this should be fixed
    time_column = column_set[1]  # this should be fixed

    data = load_csv_file(file_name, column_set, time_column)
    data = data[data[time_column].isin(times)]

    locations = list(data[id_column].drop_duplicates())

    time_series_data_list = []

    if fill_missing_timestamp: # fill in missing hours
        times_pd = pd.DataFrame(times, columns=[time_column])
        for loc in locations:
            loc_time_series = data[data[id_column] == loc]
            loc_time_series = times_pd.merge(loc_time_series, how='left', on=time_column)
            time_series_data_list.append(loc_time_series[features].values.T)
        time_series_data = np.vstack(time_series_data_list)

    else:
        time_series_data = data[features].values.reshape(len(locations), len(times), len(features))

    return time_series_data, locations, features


def load_csv_file(file, col_names, if_parse_dates=None):
    if if_parse_dates:
        return pd.read_csv(file, names=col_names, parse_dates=[if_parse_dates], skiprows=1)
    else:
        return pd.read_csv(file, names=col_names, skiprows=1)


def load_config(file):
    model_config = load_json_file(file)
    data_config_file = model_config['data_config']
    data_config = load_json_file(data_config_file)
    config = dict(model_config, **data_config)
    print_configuration(config)
    return config


def print_configuration(config):
    print('CONFIG: if_include_static = {}.'.format(config['other_settings'].get('if_include_static')))
    print('CONFIG: lag_offset = {}.'.format(config['other_settings'].get('lag_offsets')))
    # print('Number of lag steps for the neighbors: {}.'.format(config['other_settings'].get('neighbor_lag_step')))
    print()


def load_json_file(json_file):
    data = open(json_file).read()
    json_data = json.loads(data)
    return json_data