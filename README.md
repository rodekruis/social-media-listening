# social-media-listening
Data pipeline to collect, analyse and store text messages from social media platforms.

## Description

Synopsis: a [dockerized](https://www.docker.com/get-started) [python](https://www.python.org/) application that collects
text messages from [X](https://x.com/) and [Telegram](https://telegram.org/), translates them using [Azure AI Translator](https://azure.microsoft.com/en-us/products/ai-services/ai-translator), 
classifies them using [custom models](https://huggingface.co/rodekruis) and saves them in [Argilla](https://argilla.io/), which is used to validate the classifications.

## Setup

### Prequisites
- Ensure files `pyproject.toml` and `poetry.lock` available at root. The latter is for caching dependencies
- [ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16) if storing data in Microsoft Azure SQL Server 

### Set up locally (with `poetry`)
1. Install [Python Poetry](https://python-poetry.org/docs/)
2. Edit config file to your need using the template `config-template.yaml` in `config` folder. Save it as `config.yaml` in the same folder.
3. Run command:
    ```
    poetry run python -m telegram_pipeline --country <someCountry>
    ```
    where `<someCountry>` is a country name in the yaml file.

### Set up with Docker 
1. Install [Docker](https://www.docker.com/get-started)
2. Build the docker image from the root directory
    ```
    docker build -t rodekruis/social-media-listening .
    ```
3. Run the dockerised pipeline in 2 ways:
    - With default configurations:
        ```
        docker run -it rodekruis/social-media-listening --country <someCountry>
        ```
    - Or enter the docker image interactively for more run options (such as running for specific countries one by one):
        ```
        docker run -it --entrypoint /bin/bash rodekruis/social-media-listening
        ```
        Then run in the opened container:
        ```
        poetry run python -m telegram_pipeline --country <someCountry>
        ```