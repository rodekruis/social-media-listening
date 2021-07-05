import datetime
import os
from pipeline.get_data import get_twitter, get_youtube
from pipeline.parse_data import parse_data
from pipeline.predict_sentiment_tweets import predict_sentiment_tweets
from pipeline.predict_topic_tweets import predict_topic_tweets
from pipeline.prepare_final_dataset import prepare_final_dataset
from azure.storage.blob import BlobServiceClient, BlobClient
import pandas as pd
import logging
logging.basicConfig()
ch = logging.getLogger()
ch.setLevel(logging.INFO)


def main():
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    # load configuration
    df_twitter_users_to_track = pd.read_csv('config/tweets_to_track.csv')
    twitter_users_to_track = df_twitter_users_to_track.dropna()['user_id'].tolist()

    df_youtube_channels_to_track = pd.read_csv('config/youtube_to_track.csv')
    youtube_channels_to_track = df_youtube_channels_to_track.dropna()['channel_id'].tolist()

    df_keywords = pd.read_csv('config/keywords.csv')
    keywords = df_keywords.dropna()['keyword'].tolist()

    # execute pipeline
    try:
        get_twitter(twitter_users_to_track)
    except Exception as e:
        logging.error('Error in get_twitter:')
        logging.error(e)

    try:
        get_youtube(youtube_channels_to_track)
    except Exception as e:
        logging.error('Error in get_youtube:')
        logging.error(e)

    try:
        parse_data(keywords)
    except Exception as e:
        logging.error('Error in parse_data:')
        logging.error(e)

    logging.info('Python timer trigger function ran at %s', utc_timestamp)


if __name__ == "__main__":
    main()