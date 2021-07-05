import datetime
import os
from pipeline.get_data import get_twitter, get_youtube
from pipeline.parse_data import parse_data
import pandas as pd
import logging
import click
logging.basicConfig()
ch = logging.getLogger()
ch.setLevel(logging.INFO)


@click.command()
@click.option('--translate', is_flag=True, help='translate text with Google API')
@click.option('--datalake', is_flag=True, help='upload to Azure datalake')
def main(translate, datalake):

    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    # load configuration
    df_twitter_users_to_track = pd.read_csv('config/tweets_to_track.csv')
    twitter_users_to_track = df_twitter_users_to_track.dropna()['user_id'].tolist()

    df_youtube_channels_to_track = pd.read_csv('config/youtube_to_track.csv')
    youtube_channels_to_track = df_youtube_channels_to_track.dropna()['channel_id'].tolist()

    df_keywords = pd.read_csv('config/keywords.csv')
    keywords = df_keywords.dropna()['keyword'].tolist()

    use_datalake = datalake

    # execute pipeline
    try:
        get_twitter(twitter_users_to_track, use_datalake)
    except Exception as e:
        logging.error('Error in get_twitter:')
        logging.error(e)

    try:
        get_youtube(youtube_channels_to_track, use_datalake)
    except Exception as e:
        logging.error('Error in get_youtube:')
        logging.error(e)

    try:
        parse_data(keywords, use_datalake, translate)
    except Exception as e:
        logging.error('Error in parse_data:')
        logging.error(e)

    logging.info('Python timer trigger function ran at %s', utc_timestamp)


if __name__ == "__main__":
    main()