import plotly.graph_objects as go
import numpy as np

from geopy.geocoders import Nominatim
from geopy.distance import distance


def create_plot(df, input_latlon):
    '''
    :param df: dataframe output for get_top5_distance
    :param input_latlon: output for get_input_latlon
    :return: html string
    '''
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=df['latitude'], lon=df['longitude'],
                                hovertext=df['display'], marker=dict(size=10)))

    sliders = [go.layout.Slider(dict(
        active=2,
        steps=[
            dict(label='Near'),
            dict(label='Far')
        ])
    )]
    fig.update_layout(template='plotly_dark',
                      width=1100,
                      height=600,
                      sliders=sliders)
    fig.update_geos(showcountries=True)

    fig.add_trace(go.Scattergeo(lat=[input_latlon[0]], lon=[input_latlon[1]],
                                hovertext="You are here", marker=dict(size=10)))
    fig.update_layout(showlegend=False)

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


def get_top5_distance(df, prox):
    '''
    :param df: dataframe output from get_distances
    :param prox: string, one of "near", "far", or "wherever"
    :return:
    '''
    if prox == "near":
        # returns nearest 5 results
        print('...Ascending')
        df5 = df.nsmallest(5, 'dist')
        df5 = df5.sort_values(by=['cos_diff'], ascending=True)
        df5 = df5.reset_index(drop=True)
    elif prox == "far":
        # returns furthest 5 results
        print('...Descending')
        df5 = df.nlargest(5, 'dist')
        df5 = df5.sort_values(by=['cos_diff'], ascending=False)
        df5 = df5.reset_index(drop=True)
    elif prox == "wherever":
        # returns all results
        print('...wherever')
        df5 = df.sort_values(by=['cos_diff'], ascending=False)
        df5 = df5.reset_index(drop=True)
    else:
        print("...Invalid input in PROX")
        df = None
    return df5