import os.path
from datetime import datetime, timedelta
from sml.pipeline import Pipeline
from sml.secrets import Secrets

pipe = Pipeline()
if not os.path.exists("credentials/.env"):
    print('credentials not found, run this test from root directory')

pipe.load.set_storage("Azure SQL Database", secrets=Secrets("credentials/.env"))
# load messages to check that they were correctly saved
messages = pipe.load.get_messages(
    start_date=datetime.today() - timedelta(days=7),
    end_date=datetime.today(),
    country='USA',
    source='Telegram'
)
print(f"loaded {len(messages)} messages!")

pipe.transform.set_translator(from_lang="en", to_lang="it", secrets=Secrets("credentials/.env"))
pipe.transform.set_classifier(type="setfit", model="rodekruis/sml-ukr-message-classifier", lang="it")
messages = pipe.transform.process_messages(messages, translate=True, classify=True)
print(f"processed {len(messages)} messages!")
