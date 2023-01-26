import smm

# with pipeline
my_pipeline = smm.Pipeline()

twitter_secrets = smm.SocialMediaSecrets("twitter", "env")
    #api_key="", api_secrets="")
my_pipeline.extract.set_source("twitter", twitter_secrets)

my_pipeline.transform.set_translator(model="Microsoft", from="uk", to="en")

my_pipeline.run_pipline(start_date="03-11-2022",
                        end_date="10-11-2022",
                        queries=["refugees", "shelter"])


# without pipeline
twitter_secrets = smm.SocialMediaSecrets("twitter", "env")
    #api_key="", api_secrets="")
my_extractor = smm.Extract("twitter", twitter_secrets)

messages = my_extractor.get_data(start_date="03-11-2022",
                                 end_date="10-11-2022",
                                 queries=["refugees", "shelter"])

my_transformer = smm.Transform()
my_transformer.set_translator(model="Microsoft", from="uk", to="en")
messages = my_transformer.process_data(messages)

my_loader = smm.Loader()
my_loader.save_messages(messages)


