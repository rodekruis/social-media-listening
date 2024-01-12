import os.path
from datetime import datetime, timedelta
from sml.pipeline import Pipeline
from sml.secrets import Secrets

# with pipeline
pipe = Pipeline()
pipe.load.set_storage("local")

print(os.path.exists("credentials/.env"))
tg_secrets = Secrets("credentials/.env")
messages = pipe.extract.set_source("telegram", secrets=tg_secrets).get_data(
    start_date=datetime.today()-timedelta(days=7),
    channels=['t.me/nytimes'],
    store_temp=False
)
print(f"found {len(messages)}!")



# test_date = datetime(2023, 11, 29, 17, 00, 00)
# sql_secrets = Secrets("../credentials/.env")
# messages = pipe.load.set_storage("Azure SQL Database", secrets=sql_secrets).get_messages(
#     start_date=test_date-timedelta(days=1),
#     end_date=test_date,
#     country='BGR',
#     source='telegram'
# )
# print(messages[0].text)

# my_pipeline.transform.set_translator(model="Microsoft", from="uk", to="en")
#
# my_pipeline.run_pipline(start_date="03-11-2022",
#                         end_date="10-11-2022",
#                         queries=["refugees", "shelter"])
#
#
# # without pipeline
# twitter_secrets = smm.SocialMediaSecrets("twitter", "env")
#     #api_key="", api_secrets="")
# my_extractor = smm.Extract("twitter", twitter_secrets)
#
# messages = my_extractor.get_data(start_date="03-11-2022",
#                                  end_date="10-11-2022",
#                                  queries=["refugees", "shelter"])
#
# my_transformer = smm.Transform()
# my_transformer.set_translator(model="Microsoft", from="uk", to="en")
# messages = my_transformer.process_data(messages)
#
# my_loader = smm.Loader()
# my_loader.save_messages(messages)


