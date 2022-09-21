import datetime
import os, traceback, sys
import pandas as pd
from pipeline.get_data import get_twitter, get_youtube, get_kobo, get_facebook, get_telegram
from pipeline.parse_data import parse_twitter, parse_youtube, parse_kobo, parse_facebook, parse_azure_table, \
    merge_sources, parse_telegram, prepare_final_dataset
from pipeline.utils import get_table_service_client
from azure.data.tables import UpdateMode
from tqdm import tqdm
import logging
import click
import json
import yaml
from dotenv import load_dotenv

logging.root.handlers = []
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.DEBUG,
    filename="ex.log",
)
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)


@click.command()
@click.option("--config", default="romania.yaml", help="configuration file (yaml)")
@click.option("--days", default=14, help="number of days to be scraped (14 by default)")
@click.option("--keep", default="", help="keep first N messages (all if empty)")
def main(config, days, keep):

    utc_timestamp = (
        datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
    )

    # load configuration
    with open(f"../config/{config}") as file:
        if config.endswith("json"):
            config = json.load(file)
        elif config.endswith("yaml"):
            config = yaml.load(file, Loader=yaml.FullLoader)

    # load credentials
    if all(x in os.environ for x in ["AZURE_CLIENT_ID",
                                     "AZURE_CLIENT_SECRET",
                                     "AZURE_TENANT_ID",
                                     "START_DATE",
                                     "END_DATE"]):
        pass
    else:
        load_dotenv(f"../credentials/.env")

    data_to_merge = []

    # execute pipeline
    if config["track-azure-table"]:
        try:
            table_client = get_table_service_client(config["azure-table-name"], config)
            df = pd.DataFrame(
                table_client.query_entities(
                    "Timestamp gt datetime'2000-01-01T00:00:00Z'"
                )
            )
            if keep != "":
                df = df[: int(keep)]
            df_start = df.copy()
        except Exception as e:
            logging.error(f"in getting azure table data: {e}")
            traceback.print_exception(*sys.exc_info())
        try:
            df = parse_azure_table(df, config)
        except Exception as e:
            logging.error(f"in parsing azure table data: {e}")
            traceback.print_exception(*sys.exc_info())

        try:
            logging.info("updating azure table")
            table_client = get_table_service_client(config["azure-table-name"], config)
            df = df.dropna(subset=["topic"])  # drop nan on topic

            # drop what was classified already
            len_before = len(df)
            df = df.drop(
                index=df_start[
                    df_start["topic"].isin(df["topic"].unique().tolist())
                ].index
            )
            logging.info(
                f"updating entities with missing topic: {len(df)} (out of {len_before})"
            )

            # update entities in table
            df_test, cnt = pd.DataFrame(), 0
            for ix, row in df.iterrows():
                if (cnt % 1000) == 0:
                    logging.info(f"{cnt}/{len(df)}")
                cnt += 1
                replaced = table_client.get_entity(
                    partition_key=row["PartitionKey"], row_key=row["RowKey"]
                )
                replaced["topic"] = row["topic"]
                df_test = df_test.append(pd.Series(replaced), ignore_index=True)
                table_client.update_entity(mode=UpdateMode.MERGE, entity=replaced)
        except Exception as e:
            logging.error(f"in updating azure table data: {e}")
            traceback.print_exception(*sys.exc_info())

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

    if config["track-facebook-groups"]:
        try:
            get_facebook(config)
        except Exception as e:
            logging.error(f"in getting facebook data: {e}")
            traceback.print_exception(*sys.exc_info())
        try:
            data_fb = parse_facebook(config)
            data_to_merge.append(data_fb)
        except Exception as e:
            logging.error(f"in parsing facebook data: {e}")
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

    if config["track-telegram-groups"]:
        if config["get-data"]:
            try:
                get_telegram(config, days)
            except Exception as e:
                logging.error(f"in getting telegram data: {e}")
                traceback.print_exception(*sys.exc_info())
        if config["parse-data"]:
            try:
                data_telegram = parse_telegram(config)
                data_to_merge.append(data_telegram)
            except Exception as e:
                logging.error(f"in parsing telegram data: {e}")
                traceback.print_exception(*sys.exc_info())

    # if len(data_to_merge) > 0:
    #     try:
    #         merge_sources(data_to_merge, config)
    #     except Exception as e:
    #         logging.error(f"in merging data: {e}")
    #         traceback.print_exception(*sys.exc_info())
    # else:
    #     logging.warning("No data to merge, skipping")

    logging.info("Python timer trigger function ran at %s", utc_timestamp)


if __name__ == "__main__":
    main()
