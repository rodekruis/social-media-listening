import pytest
import logging
from smm.extract import Extract
from smm.secrets import Secrets
from smm.message import Message
from datetime import datetime, timedelta

# Get logger
logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# Initialize transformer
test_secrets = Secrets(path_or_url="credentials/credentials_test.json", source="json")
test_extractor = Extract(source="telegram",
                         secrets=test_secrets)

test_extractor.get_data(channels=["t.me/UAinplovdiv"],
                        end_date=datetime.today()-timedelta(days=1))








# # Initialize test message
# template_message = Message(
#     id_=0,
#     datetime_=datetime.now(),
#     datetime_scraped_=datetime.now(),
#     country="NLD",
#     source="Twitter",
#     text="Hello world!")

