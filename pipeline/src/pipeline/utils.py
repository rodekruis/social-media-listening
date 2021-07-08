import preprocessor as tp
import pandas as pd
import numpy as np
import os
from google.cloud import translate_v2 as translate
from apiclient import discovery
from google.oauth2 import service_account
import preprocessor as tp
from time import sleep
from requests.exceptions import ReadTimeout, ConnectionError
from shapely.geometry import Polygon, Point
import geopandas as gpd
import json
import enchant
import nltk
import ast
from azure.storage.blob import BlobServiceClient, BlobClient
from tqdm import tqdm
tqdm.pandas()
tp.set_options(tp.OPT.URL, tp.OPT.EMOJI, tp.OPT.MENTION)
en_dict = enchant.Dict("en_US")
count_translate = 0
nltk.download('punkt')


def get_blob_service_client(blob_path):
    with open("../credentials/blobstorage_secrets.json") as file:
        blobstorage_secrets = json.load(file)
    blob_service_client = BlobServiceClient.from_connection_string(blobstorage_secrets['connection_string'])
    container = blobstorage_secrets['container']
    return blob_service_client.get_blob_client(container=container, blob=blob_path)


def match_location(sentence, gdf):
    tokens = nltk.word_tokenize(sentence)
    tokens = [x.lower() for x in tokens]
    match_token, match_geo = np.nan, np.nan
    for token in tokens:
        gdf_match = gdf[gdf['ADM_EN'] == token]
        if len(gdf_match) > 0:
            match_token = token
            match_geo = gdf_match['geometry'].values[0]
    return match_geo, match_token


def extract_coordinates(x):
    """ extract coordinates from tweet's place field """
    if not pd.isna(x):
        x = ast.literal_eval(x)
        bbox = Polygon(x['bounding_box']['coordinates'][0])
        centroid = bbox.centroid.coords
        return Point(centroid)
    else:
        return np.nan


def extract_location(x):
    """ extract locations from tweet's place field """
    if not pd.isna(x):
        x = ast.literal_eval(x)
        location = x['name']
        return location
    else:
        return np.nan


def point_to_xy(x):
    """ extract x,y coordinates from shapely point """
    if not pd.isna(x):
        return x.x, x.y
    else:
        return np.nan, np.nan


def geolocate_dataframe(df_tweets, location_file, adm0_file, location_columns, tw_place_column):

    gdf = gpd.read_file(location_file, encoding='utf8')
    gdf['is_english'] = gdf['ADM_EN'].apply(en_dict.check)
    gdf = gdf[~gdf['is_english']].drop(columns=['is_english'])
    gdf['ADM_EN'] = gdf['ADM_EN'].str.lower()

    # parse locations
    if tw_place_column in df_tweets.columns:
        df_tweets['coord'] = df_tweets[tw_place_column].apply(extract_coordinates)
        df_tweets['location'] = df_tweets[tw_place_column].apply(extract_location)
    else:
        df_tweets['coord'] = np.nan
        df_tweets['location'] = np.nan

    for location_column in location_columns:
        df_tweets['temp_coord'], df_tweets['temp_location'] = zip(*df_tweets[location_column].progress_apply(lambda x: match_location(x, gdf)))
        df_tweets['coord'] = df_tweets['coord'].fillna(df_tweets['temp_coord'])
        df_tweets['location'] = df_tweets['location'].fillna(df_tweets['temp_location'])

    df_tweets = df_tweets.drop(columns=['temp_coord', 'temp_location'])
    count_geolocated = df_tweets.coord.count()

    # filter by country
    gdf_tweets = gpd.GeoDataFrame(df_tweets[~pd.isna(df_tweets.coord)], geometry='coord')
    gdf_country = gpd.read_file(adm0_file, encoding='utf8')
    gdf_tweets = gdf_tweets[gdf_tweets.geometry.within(gdf_country.geometry[0])]
    count_geolocated_filtered = gdf_tweets.coord.count()
    if count_geolocated_filtered < count_geolocated:
        df_tweets = df_tweets[df_tweets['id'] not in gdf_tweets.id.tolist()]

    df_tweets['longitude'], df_tweets['latitude'] = zip(*df_tweets['coord'].apply(point_to_xy))
    df_tweets = df_tweets.drop(columns=['coord'])
    return df_tweets


def clean_text(row_, text_column):
    if 'lang' in row_.keys():
        if row_['lang'] == 'en':
            text_clean = tp.clean(row_[text_column]).lower()
            return text_clean
        else:
            return row_[text_column]
    else:
        text_clean = tp.clean(row_[text_column]).lower()
        return text_clean


def html_decode(row_, text_column):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    """
    htmlCodes = (
            ("'", '&#39;'),
            ('"', '&quot;'),
            ('>', '&gt;'),
            ('<', '&lt;'),
            ('&', '&amp;')
        )
    text = row_[text_column]
    for code in htmlCodes:
        text = text.replace(code[1], code[0])
    return text


def translate_string(row_, translate_client, text_field):
    text = row_[text_field]
    if 'lang' in row_.keys():
        lang = row_['lang']
    else:
        lang = 'unknown'
    if lang != 'en':
        try:
            result = translate_client.translate(text, target_language="en")
        except ReadTimeout or ConnectionError:
            sleep(60)
            try:
                result = translate_client.translate(text, target_language="en")
            except ReadTimeout or ConnectionError:
                sleep(60)
                result = translate_client.translate(text, target_language="en")
        trans = result["translatedText"]
        global count_translate
        count_translate += 1
        return trans
    else:
        return text


def translate_dataframe(df_tweets, text_column):
    global count_translate
    count_translate = 0

    # get Google API credentials
    service_account_info = "../credentials/google_service_account_secrets.json"
    credentials = service_account.Credentials.from_service_account_file(service_account_info)
    translate_client = translate.Client(credentials=credentials)

    df_tweets = df_tweets.dropna(subset=[text_column])
    df_texts = df_tweets.drop_duplicates(subset=[text_column])

    # translate to english
    df_texts[text_column] = df_texts.progress_apply(lambda x: translate_string(x, translate_client, text_column), axis=1)

    for ix, row in df_tweets.iterrows():
        df_texts_ = df_texts[df_texts['id'] == row['id']]
        if len(df_texts_) > 0:
            df_tweets.at[ix, text_column] = df_texts_[text_column].values[0]
        df_tweets.at[ix, 'lang'] = 'en'

    # clean text
    df_tweets[text_column] = df_tweets.apply(clean_text, args=(text_column,), axis=1)

    df_tweets[text_column] = df_tweets[text_column].str.lower()
    return df_tweets


def filter_by_keywords(df_tweets, text_columns, keywords):
    df_tweets['is_conflict'] = False
    for text_column in text_columns:
        df_tweets.at[df_tweets[text_column].str.contains('|'.join(keywords)), 'is_conflict'] = True
    df_tweets = df_tweets[df_tweets['is_conflict']].drop(columns=['is_conflict'])
    return df_tweets



