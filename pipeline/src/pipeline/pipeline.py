import datetime
import os, traceback, sys
from pipeline.get_data import get_twitter, get_youtube
from pipeline.parse_data import parse_twitter, parse_youtube, merge_sources
import pandas as pd
import logging
import click
import json
from dotenv import load_dotenv
logging.root.handlers = []
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='ex.log')
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)


@click.command()
@click.option('--config', default="namibia.json", help='configuration file (json)')
def main(config):

    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    # load configuration
    with open(f"../config/{config}") as file:
        config = json.load(file)

    # load credentials
    load_dotenv(f"../credentials/.env")

    data_to_merge = []

    # execute pipeline
    if config["track-twitter-queries"] or config["track-twitter-users"]:
        try:
            get_twitter(config)
        except Exception as e:
            logging.error(f"in getting twitter data: {e}")
            traceback.print_exception(*sys.exc_info())
        try:
            data_twitter = parse_twitter(config)
            data_to_merge.append(data_twitter)
        except Exception as e:
            logging.error(f"in parsing twitter data: {e}")
            traceback.print_exception(*sys.exc_info())

    if config["track-youtube-channels"]:
        try:
            get_youtube(config)
        except Exception as e:
            logging.error(f"in getting youtube data: {e}")
            traceback.print_exception(*sys.exc_info())
        try:
            data_youtube = parse_youtube(config)
            data_to_merge.append(data_youtube)
        except Exception as e:
            logging.error(f"in parsing youtube data: {e}")
            traceback.print_exception(*sys.exc_info())

    try:
        merge_sources(data_to_merge, config)
    except Exception as e:
        logging.error(f"in merging data: {e}")
        traceback.print_exception(*sys.exc_info())

    logging.info('Python timer trigger function ran at %s', utc_timestamp)


if __name__ == "__main__":
    main()