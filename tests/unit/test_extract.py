import pytest
import logging
from smm.extract import Extract
from smm.secrets import Secrets
from datetime import datetime, timedelta

# For opening a new sync loop if needed
# import nest_asyncio
# nest_asyncio.apply()

# Get logger
logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# Initialize transformer
test_secrets = Secrets(path_or_url="credentials/credentials_test.json", source="json")
test_extractor = Extract(source="telegram",
                         secrets=test_secrets)

messages = test_extractor.get_data(channels=["t.me/dopomogaukraini"],
                                   start_date=datetime.today()-timedelta(days=1))



