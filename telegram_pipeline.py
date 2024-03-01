import os.path
from datetime import datetime, timedelta
from sml.pipeline import Pipeline
from sml.secrets import Secrets
import yaml
import click


@click.command()
@click.option("--country", type=str, required=True, help="Specify one country name : bulgaria, poland, slovakia, ukraine")
def run_sml_pipeline(country):
    if os.path.exists('config/config.yaml'):
        with open('config/config.yaml','r') as f:
            settings = yaml.safe_load(f)
    else:
        settings = yaml.safe_load(os.environ["CONFIG"])
    # todo: add reading az keyvault 
    # if os.path.exists("credentials/.env"):
    #     pipe = Pipeline(secrets=Secrets("credentials/.env"))
    # else:
    #     print('Azure Key Vault not found, try with local env')
    pipe = Pipeline(secrets=Secrets("env"))

    print(f"scraping messages")
    pipe.extract.set_source("telegram")
    messages = pipe.extract.get_data(
        start_date=datetime.today()-timedelta(days=1), # todo 
        country=settings[country]['country-code'],
        channels=settings[country]['channels-to-track'][:1], # todo 
        store_temp=False
    )
    print(f"found {len(messages)} messages!")

    pipe.transform.set_translator(model="Microsoft", 
                                  from_lang=["ru","uk"], 
                                  to_lang="en")
    pipe.transform.set_classifier(type="setfit", 
                                  model="rodekruis/sml-ukr-message-classifier",
                                  lang="en")
    messages = pipe.transform.process_messages(messages, translate=True, classify=True)
    print(f"processed {len(messages)} messages!")

    pipe.load.set_storage("Azure SQL Database")
    pipe.load.save_messages(messages)
    # pipe.load.push_to_argilla(messages)
    print(f"saved {len(messages)} messages!")


if __name__ == '__main__':
    run_sml_pipeline()