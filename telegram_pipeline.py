import os.path
from datetime import datetime, timedelta
from sml.pipeline import Pipeline
from sml.secrets import Secrets
import yaml
import click
import logging
import sys
import asyncio
import pandas as pd
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.functions.channels import GetFullChannelRequest
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)


@click.command()
@click.option("--country", type=str, required=True,
              help="Specify one country name : bulgaria, poland, slovakia, ukraine")
def run_sml_pipeline(country):
    if os.path.exists('config/config.yaml'):
        with open('config/config.yaml', 'r') as f:
            settings = yaml.safe_load(f)
    else:
        settings = yaml.safe_load(os.environ["CONFIG"])

    start_date = datetime.today() - timedelta(days=14)
    end_date = datetime.today()
    country_code = settings[country]['country-code']

    # todo: add reading az keyvault 
    # if os.path.exists("credentials/.env"):
    #     pipe = Pipeline(secrets=Secrets("credentials/.env"))
    # else:
    #     print('Azure Key Vault not found, try with local env')
    pipe = Pipeline(secrets=Secrets("env"))

    logging.info(f"scraping messages")
    pipe.extract.set_source("telegram")
    messages = pipe.extract.get_data(
        start_date=start_date,
        country=country_code,
        channels=settings[country]['channels-to-track'],
        store_temp=False
    )
    logging.info(f"found {len(messages)} messages!")

    pipe.transform.set_translator(model="Microsoft",
                                  from_lang="",  # empty string means auto-detect language
                                  to_lang="en")
    pipe.transform.set_classifier(type="setfit",
                                  model="rodekruis/sml-ukr-message-classifier",
                                  lang="en")
    messages = pipe.transform.process_messages(messages, translate=True, classify=True)
    logging.info(f"processed {len(messages)} messages!")

    pipe.load.set_storage("Azure SQL Database")
    pipe.load.save_messages(messages)
    pipe.load.push_to_argilla(
        messages=messages,
        dataset_name=f"{country_code.lower()}-{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}",
        tags={"Country": country_code}
    )
    logging.info(f"saved {len(messages)} messages!")
    
    # add member counts
    telegram_client = TelegramClient(
        StringSession(os.getenv('STRING_SESSION')),
        os.getenv('API_ID'),
        os.getenv('API_HASH')
    )
    telegram_client.loop.run_until_complete(
        save_membercount(
            telegram_client,
            settings,
            country,
            end_date
        )
    )


async def save_membercount(telegram_client, settings, country, end_date):
    df_member_counts = pd.DataFrame()
    
    await telegram_client.connect()
    
    for channel in settings[country]['channels-to-track']:
        channel_entity = await telegram_client.get_entity(channel)
        channel_full_info = await telegram_client(GetFullChannelRequest(channel=channel_entity))
        idx = len(df_member_counts)
        df_member_counts.at[idx, 'source'] = channel
        member_count = channel_full_info.full_chat.participants_count
        df_member_counts.at[idx, 'member_count'] = member_count
        df_member_counts.at[idx, 'date'] = end_date.strftime("%m %d %Y")
        df_member_counts.at[idx, 'country'] = settings[country]['country-code'].lower()
        df_member_counts.at[idx, 'source_id'] = channel
    
    # Add index column
    df_member_counts.reset_index(inplace=True)
    df_member_counts['id'] = df_member_counts.index
    
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('BLOBSTORAGE_CONNECTION_STRING'))
    blob_client = blob_service_client.get_blob_client(container='membercount', blob='membercount.csv')
    with open('membercount.csv', "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    df_member_counts_old = pd.read_csv('membercount.csv')
    df_member_counts_all = pd.concat([df_member_counts_old, df_member_counts]).reset_index(drop=True)
    df_member_counts_all = df_member_counts_all.drop(columns=['index', 'id'])
    df_member_counts_all.to_csv('membercount.csv', index=False, encoding="utf-8")
    with open('membercount.csv', "rb") as upload_file:
        blob_client.upload_blob(upload_file, overwrite=True)
    if os.path.exists('membercount.csv'):
        os.remove('membercount.csv')
    

if __name__ == '__main__':
    run_sml_pipeline()
