import plotly
import plotly.graph_objs as go
import numpy as np

from geopy.geocoders import Nominatim
from geopy.distance import distance
import json


def create_plot(df):

    # fig = go.Figure()
    # fig.add_trace(go.Scattergeo(lat=df['latitude'], lon=df['longitude']))
    # fig.update_layout(dict(template='plotly_dark'))

    fig = go.Scattergeo(lat=df['latitude'], lon=df['longitude'], hovertext=df['display'])
    data = [fig]
    # graph = [
    #     dict(
    #         data=[go.Scattergeo(lat=df['latitude'], lon=df['longitude'], hovertext=df['display'])],
    #         layout=dict(
    #             geo=dict(showcountries=True),
    #             plot_bgcolor="black"
    #         )
    #     )
    # ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


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
        df5 = df.sort_values('dist', ascending=True).head(5)
        df5 = df5.reset_index(drop=True)
    else:
        print('...Descending')
        df5 = df.nlargest(5, 'dist')
        df5 = df5.reset_index(drop=True)
    return df5