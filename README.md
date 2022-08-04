# rumor-tracker

Track specific topic(s) in news and social media.
Featuring: geolocation, translation to English, [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis), and [topic modelling](https://en.wikipedia.org/wiki/Topic_model).

Built to support Philippine, Namibia and Ethiopian Red Cross Society.

Credits: [Wessel de Jong](https://github.com/Wessel93), [Phuoc Phung](https://github.com/p-phung), [Jacopo Margutti](https://github.com/jmargutt)

## Introduction
This repo contains the code to:
1. Download text data on a specific topic (e.g. COVID-19 vaccines)
2. Translate it to English
3. Analyze sentiment (is it positive or negative?)
4. Divide it into topics
5. Assign a topic and a representative example to each group

Topic modelling is built on top of [GSDMM: short text clustering](https://github.com/rwalk/gsdmm), while sentiment and translation use [Hugging Face Models](https://huggingface.co/) and/or [Google Cloud Natural Language](https://cloud.google.com/natural-language).

N.B. the creation of groups (a.k.a. clustering) is automated, but the topic description is not. You need a human to read some representative examples of each group and come up 
with a meaningful, human-readable description.

Data sources supported by the rumor-tracker:
1. Twitter
2. YouTube
3. KoBo
4. Facebook
5. Telegram
6. Azure Table Storage

## Setup
Generic requirements:
-   [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/)
-   [Azure Data Lake Storage](https://docs.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction)
-   OPTIONAL (Twitter): [Twitter developer account](https://developer.twitter.com/en/apply-for-access)
-   OPTIONAL (geolocate): vector files of locations and country boundaries
-   OPTIONAL (YouTube, translate): [Google Cloud account](https://cloud.google.com/)
-   OPTIONAL (Telegram): [Telegram API Development](https://my.telegram.org)

More in detail:
- Follow [these instructions](https://docs.google.com/document/d/1q6h5zYDFLMaWDGBfSEe0EGl8Ymi09WhuqpHPxnQy6DU/edit?usp=sharing) to store credentials in Azure Key Vault and use them with the rumor-tracker. Secrets need to be in json format and contain all necessary fields, templates TBI
- For 510: Google cloud service account credentials are accessible [here](https://console.cloud.google.com/apis/credentials?project=eth-conflict-tracker&folder=&organizationId=&supportedpurview=project), but create a new project if needed!. Login credentials in Bitwarden.

The rumor-tracker can be confgured via one configuration file (yaml), see country-specific examples under [config](https://github.com/rodekruis/rumor-tracker/tree/master/config)

### with Docker
1. Install [Docker](https://www.docker.com/get-started)
2. Build the docker image from the root directory
```
docker build -t rodekruis/rumor-tracker .
```
3. Run the docker image in a new container and access it
```
docker run -it --entrypoint /bin/bash rodekruis/rumor-tracker
```
4. Check that everything is working by running the pipeline (see [Usage](https://github.com/rodekruis/news-tracker-ethiopia#usage) below)
5. Congratulations! You can now use the rumor-tracker as a dockerized app in your favorite cloud provider, e.g. [using Azure Logic App](https://docs.google.com/document/d/182aQPVRZkXifHDNjmE66tj5L1l4IvAt99rxBzpmISPU/edit?usp=sharing)

### Manual Setup
TBI

## Usage
Command:
```
run-pipeline [OPTIONS]
```
Options:
  ```
  --config                    configuration file (json)
  --help                      show this message and exit
  ```


##  Versions

| Pipeline version | Changes |
| --- | --- |
| 0.1.3 | Add function to post non-trigger in off-season |
| 0.1.2 | Corrected generation of link to raw chirps file <br> Fixed misdownloading a processed rainfall from datalake <br> Fixed raw chirps files listing for calculating zonal statistics <br> Minor fixes |
| 0.1.1 | ENSO+rainfall model added <br> Minor fixes | 
| 0.1.0 | Initial version, ENSO-only model |