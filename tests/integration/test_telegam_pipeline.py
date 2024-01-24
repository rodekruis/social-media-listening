import os.path
from datetime import datetime, timedelta
from sml.pipeline import Pipeline
from sml.secrets import Secrets

if not os.path.exists("credentials/.env"):
    print('credentials not found, run this test from root directory')
pipe = Pipeline(secrets=Secrets("credentials/.env"))

print(f"scraping messages")
pipe.extract.set_source("telegram")
messages = pipe.extract.get_data(
    start_date=datetime.today()-timedelta(days=7),
    country='USA',
    channels=['t.me/nytimes'],
    store_temp=False
)
print(f"found {len(messages)} messages!")


pipe.transform.set_translator(model="Microsoft", from_lang="en", to_lang="it")
pipe.transform.set_classifier(type="setfit", model="rodekruis/sml-ukr-message-classifier", lang="it")
messages = pipe.transform.process_messages(messages, translate=True, classify=True)
print(f"processed {len(messages)} messages!")


pipe.load.set_storage("Azure SQL Database")
pipe.load.save_messages(messages)
# load messages to check that they were correctly saved
messages = pipe.load.get_messages(
    start_date=datetime.today()-timedelta(days=7),
    end_date=datetime.today(),
    country='USA',
    source='Telegram'
)
print(f"saved {len(messages)} messages!")