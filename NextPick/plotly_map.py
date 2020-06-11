import plotly.graph_objs as go
import numpy as np

from geopy.geocoders import Nominatim
from geopy.distance import distance


def create_plot(df):
    '''
    :param df: dataframe output for get_top5_distance
    :return: html string
    '''
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=df['latitude'], lon=df['longitude'], hovertext=df['display']))
    fig.update_layout(template='plotly_dark')
    fig.update_geos(showcountries=True)
    div = fig.to_html()

    return div


def get_input_latlon(location):
    '''
    :param location: string input from web app
    :return: latitude and longitude pair of the location from GeoPy
    '''
    geolocator = Nominatim()
    location = geolocator.geocode(location)
    latlon = (location.latitude, location.longitude)
    return latlon


def get_distances(input_latlon, df):
    '''
    :param input_latlon: output from get_input_latlon
    :param df: output from create_df_for_map_plot
    :return:
    '''
    dist = []
    for i, row in df.iterrows():
        dist.append(distance(df.iloc[i]['latlon'], input_latlon).km)
    df['dist'] = np.around(dist)
    return df


def get_top5_distance(df, closest=True):
    '''
    :param df: dataframe output from get_distances
    :param closest: boolean
    :return:
    '''
    if closest:
        print('...Ascending')
        df5 = df.nsmallest(5, 'dist')
        df5 = df5.sort_values(by=['cos_diff'], ascending=True)
        df5 = df5.reset_index(drop=True)
    else:
        print('...Descending')
        df5 = df.nlargest(5, 'dist')
        df5 = df5.sort_values(by=['cos_diff'], ascending=False)
        df5 = df5.reset_index(drop=True)
    return df5