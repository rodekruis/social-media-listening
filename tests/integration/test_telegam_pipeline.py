import os.path
from datetime import datetime, timedelta
from sml.pipeline import Pipeline
from sml.secrets import Secrets
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

label_dict = {"0": "ANOMALY", "1": "ARMY", "2": "CHILDREN", "3": "CONNECTIVITY", "4": "RC CONNECT WITH RED CROSS", "5": "EDUCATION", "6": "FOOD", "7": "GOODS/SERVICES", "8": "HEALTH", "9": "CVA INCLUSION", "10": "LEGAL", "11": "MONEY/BANKING", "12": "NFI", "13": "OTHER PROGRAMS/NGOS", "14": "PARCEL", "15": "CVA PAYMENT", "16": "PETS", "17": "RC PMER/NEW PROGRAMS", "18": "CVA PROGRAM INFO", "19": "RC PROGRAM INFO", "20": "PSS & RFL", "21": "CVA REGISTRATION", "22": "SENTIMENT", "23": "SHELTER", "24": "TRANSLATION/LANGUAGE", "25": "CAR", "26": "TRANSPORT/MOVEMENT", "27": "WASH", "28": "WORK/JOBS"}

start_date = datetime.today()-timedelta(14)
end_date = datetime.today()
country_code = 'USA'

if not os.path.exists("credentials/.env"):
    print('credentials not found, run this test from root directory')
pipe = Pipeline(secrets=Secrets("credentials/.env"))
pipe.load.set_storage("Azure Cosmos DB")

# print(f"scraping messages")
# pipe.extract.set_source("telegram")
# messages = pipe.extract.get_data(
#     start_date=start_date,
#     country=country_code,
#     channels=['t.me/nytimes'],
#     store_temp=False
# )
# print(f"found {len(messages)} messages!")
#
# pipe.transform.set_translator(model="Microsoft",
#                                   from_lang="",  # empty string means auto-detect language
#                                   to_lang="en")
# pipe.transform.set_classifier(type="setfit",
#                               model="rodekruis/sml-ukr-message-classifier",
#                               lang="en")
# messages = pipe.transform.process_messages(messages, translate=True, classify=True)
# logging.info(f"processed {len(messages)} messages!")
#
# print(f"saving messages")
# pipe.load.save_messages(messages)
# # load messages to check that they were correctly saved
# messages_reload = pipe.load.get_messages()
# for message, message_argilla in zip(messages, messages_reload):
#     for key, value in message.to_dict().items():
#         if value != message_argilla.to_dict()[key]:
#             print(f"key {key} is different in messages_reload")
#             print(f"local: {value}")
#             print(f"argilla: {message_argilla.to_dict()[key]}")
messages = pipe.load.get_messages()

# print(f"save to and get from argilla")
# pipe.load.save_to_argilla(messages, "test-pipeline", label_schema=list(label_dict.values()))
messages_argilla = pipe.load.get_from_argilla("test-pipeline", only_submitted=False)
print(f"total: {len(messages_argilla)} messages from argilla")
for message, message_argilla in zip(messages, messages_argilla):
    for key, value in message.to_dict().items():
        if value != message_argilla.to_dict()[key]:
            print(f"key {key} is different in message_argilla")
            print(f"local: {value}")
            print(f"argilla: {message_argilla.to_dict()[key]}")