import os.path
from datetime import datetime, timedelta
from sml.pipeline import Pipeline
from sml.secrets import Secrets

start_date = datetime.today()-timedelta(1)
start_date = start_date.strftime('%d-%m-%Y')
end_date = datetime.today().strftime('%d-%m-%Y')
country = 'USA'

if not os.path.exists("credentials/.env"):
    print('credentials not found, run this test from root directory')
pipe = Pipeline(secrets=Secrets("credentials/.env"))

print(f"scraping messages")
pipe.extract.set_source("telegram")
messages = pipe.extract.get_data(
    start_date=datetime.today()-timedelta(days=1),
    country=country,
    channels=['t.me/nytimes'],
    store_temp=False
)
print(f"found {len(messages)} messages!")


pipe.transform.set_translator(model="Microsoft", from_lang="en", to_lang="vi")
pipe.transform.set_classifier(type="setfit", model="rodekruis/sml-ukr-message-classifier", lang="en")
messages = pipe.transform.process_messages(messages, translate=True, classify=True)
print(f"processed {len(messages)} messages!")

pipe.load.set_storage("Azure SQL Database")
pipe.load.save_messages(messages)
pipe.load.push_to_argilla(
    messages=messages,
    dataset_name=f"{country.lower()}-{start_date}-{end_date}",
    tags={"Country": "USA"}
)

# # load messages to check that they were correctly saved
# messages = pipe.load.get_messages(
#     start_date=datetime.today()-timedelta(days=1),
#     end_date=datetime.today(),
#     country='USA',
#     source='Telegram'
# )
# print(f"saved {len(messages)} messages!")