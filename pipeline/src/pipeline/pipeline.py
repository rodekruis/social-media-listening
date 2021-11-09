import datetime
import os, traceback, sys
from pipeline.get_data import get_twitter, get_youtube, get_kobo
from pipeline.parse_data import parse_twitter, parse_youtube, parse_kobo, merge_sources
import logging
import click
import json
import yaml
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
        if config.endswith('json'):
            config = json.load(file)
        elif config.endswith('yaml'):
            config = yaml.load(file, Loader=yaml.FullLoader)

    # load credentials
    if all(x in os.environ for x in ["AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID"]):
        pass
    else:
        load_dotenv(f"../credentials/.env")

    data_to_merge = []

    # execute pipeline
    if config["track-kobo-form"]:
        try:
            get_kobo(config)
        except Exception as e:
            logging.error(f"in getting kobo data: {e}")
            traceback.print_exception(*sys.exc_info())
        try:
            data_kobo = parse_kobo(config)
            data_to_merge.append(data_kobo)
        except Exception as e:
            logging.error(f"in parsing kobo data: {e}")
            traceback.print_exception(*sys.exc_info())

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

    if len(data_to_merge) > 0:
        try:
            merge_sources(data_to_merge, config)
        except Exception as e:
            logging.error(f"in merging data: {e}")
            traceback.print_exception(*sys.exc_info())
    else:
        logging.warning('No data to merge, skipping')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)


if __name__ == "__main__":
    main()