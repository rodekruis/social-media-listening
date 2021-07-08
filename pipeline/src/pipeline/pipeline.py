import datetime
import os
from pipeline.get_data import get_twitter, get_youtube
from pipeline.parse_data import parse_data
import pandas as pd
import logging
import click
logging.root.handlers = []
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='ex.log')
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


@click.command()
@click.option('--notranslate', is_flag=True, help='translate text with Google API')
@click.option('--nogeolocate', is_flag=True, help='geo-locate text')
@click.option('--nodatalake', is_flag=True, help='upload to Azure datalake')
def main(notranslate, nogeolocate, nodatalake):

    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    # load configuration
    df_twitter_users_to_track = pd.read_csv('../config/tweets_to_track.csv')
    twitter_users_to_track = df_twitter_users_to_track.dropna()['user_id'].tolist()

    df_youtube_channels_to_track = pd.read_csv('../config/youtube_to_track.csv')
    youtube_channels_to_track = df_youtube_channels_to_track.dropna()['channel_id'].tolist()

    df_keywords = pd.read_csv('../config/keywords.csv')
    keywords = df_keywords.dropna()['keyword'].tolist()

    skip_datalake = nodatalake

    # execute pipeline
    try:
        get_twitter(twitter_users_to_track, skip_datalake)
    except Exception as e:
        logging.error(f"in get_twitter: {e}")

    try:
        get_youtube(youtube_channels_to_track, skip_datalake)
    except Exception as e:
        logging.error(f"in get_youtube: {e}")

    try:
        parse_data(keywords, skip_datalake, notranslate, nogeolocate)
    except Exception as e:
        logging.error(f"in parse_data: {e}")

    logging.info('Python timer trigger function ran at %s', utc_timestamp)


if __name__ == "__main__":
    main()