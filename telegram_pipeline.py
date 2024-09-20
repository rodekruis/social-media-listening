import os.path
from datetime import datetime, timedelta
from sml.pipeline import Pipeline
from sml.secrets import Secrets
import yaml
import click
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)


@click.command()
@click.option("--country", type=str, required=True, help="Country ISO3")
@click.option("--days", type=int, default=14, help="How many days in the past")
def run_sml_pipeline(country, days):
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml", "r") as f:
            settings = yaml.safe_load(f)
    else:
        settings = yaml.safe_load(os.environ["CONFIG"])

    start_date = datetime.today() - timedelta(days=days)
    end_date = datetime.today()
    country = country.upper()

    # load secrets from .env
    pipe = Pipeline(secrets=Secrets("credentials/.env"))

    logging.info(f"scraping messages")
    pipe.extract.set_source(settings[country]["source"])
    messages = pipe.extract.get_data(
        start_date=start_date,
        country=country,
        channels=settings[country]["channels"],
        store_temp=False,
    )
    logging.info(f"found {len(messages)} messages!")

    pipe.transform.set_translator(
        model="Microsoft",
        from_lang="",  # empty string means auto-detect language
        to_lang="en",
    )
    pipe.transform.set_classifier(
        type="setfit", model="rodekruis/sml-ukr-message-classifier", lang="en"
    )
    messages = pipe.transform.process_messages(messages, translate=True, classify=True)
    logging.info(f"processed {len(messages)} messages!")

    pipe.load.set_storage("Azure Cosmos DB")
    pipe.load.save_messages(messages)
    pipe.load.save_to_argilla(
        messages=messages,
        dataset_name=f"{country}-{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}",
        workspace=country,
    )
    logging.info(f"saved {len(messages)} messages!")


if __name__ == "__main__":
    run_sml_pipeline()
