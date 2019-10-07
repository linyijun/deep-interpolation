import copy
import numpy as np

import matplotlib.pyplot as plt
import folium
import geopandas as gpd
from shapely.wkt import loads


def convert_df_to_geo_df(data):
    geo_data = copy.copy(data)
    geo_data['geom'] = geo_data['geom'].apply(lambda x: loads(x))
    geo_data  = gpd.GeoDataFrame(geo_data, geometry='geom')
    geo_data.crs = {'init': 'epsg:4326'}
    return geo_data


def plot_time_series_comparison_for_given_locations(ground_truth, predictions, given_loc, locations, times):
    for loc in given_loc:
        print('Now plotting location = {}.\n'.format(loc))
        i = locations.index(loc)
        plot_time_series_comparison(ground_truth[:, i], predictions[:, i], times)


def plot_time_series_comparison(ground_truth, predictions, times):
    plt.figure(figsize=(15, 8))
    plt.plot(times, ground_truth, marker='.', color='r', label='Ground Truth')
    plt.plot(times, predictions, marker='.', color='olive', label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('Air Quality Values')
    plt.legend(prop={'size': 15})
    plt.show()


def plot_base_map(coord_df, zoom_start=11):
    min_lat, max_lat = min(coord_df['lat']), max(coord_df['lat'])
    min_lon, max_lon = min(coord_df['lon']), max(coord_df['lon'])
    start_lat, start_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    m = folium.Map(location=[start_lat, start_lon], zoom_start=zoom_start)
    return m


def plot_numeric_geo_data(data, m, gid, feature):
    folium.Choropleth(
        data,
        data=data,
        columns=[gid, feature],
        key_on='feature.properties.{}'.format(gid),
        fill_color='BuPu',
        fill_opacity=0.7,
        line_opacity=0.5,

    ).add_to(m)
    return m


def plot_marker(data, lon, lat, m, color):
    folium.Marker(
        location=[lat, lon],
        tooltip = data,
        icon = folium.Icon(color=color),

    ).add_to(m)
    return m
