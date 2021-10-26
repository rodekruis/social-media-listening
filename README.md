# rumor-tracker

Track specific topic(s) in news and social media, translate them to English and do [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis).

Built to support Philippine, Namibia and Ethiopian Red Cross Society.

## Introduction
This repo contains the code to:
1. Download text data on a specific topic (e.g. COVID-19 vaccines)
2. Translate it to English
3. Analyze sentiment (is it positive or negative?)
4. Divide it into topics
5. Assign a topic and a representative example to each group

Built on top of [GSDMM: short text clustering](https://github.com/rwalk/gsdmm) and [Google Cloud Natural Language](https://cloud.google.com/natural-language).

N.B. the creation of groups (a.k.a. clustering) is automated, but the topic description is not. You need a human to read some representative examples of each group and come up 
with a meaningful, human-readable description.

Data sources supported by the rumor-tracker:
1. Twitter
2. YouTube
3. Kobo [TBI]
4. Facebook [TBI]

## Setup
Generic requirements:
-   [Twitter developer account](https://developer.twitter.com/en/apply-for-access)
-   [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/)
-   [Azure Data Lake Storage](https://docs.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction)
-   OPTIONAL (geolocate): vector files of locations and country boundaries
-   OPTIONAL (translate): [Google Cloud account](https://cloud.google.com/)

More in detail:
- Follow [these instructions](https://docs.google.com/document/d/1q6h5zYDFLMaWDGBfSEe0EGl8Ymi09WhuqpHPxnQy6DU/edit?usp=sharing) to store credentials in Azure Key Vault and use them with the rumor-tracker. Secrets need to be in json format and contain all necessary fields, see secrets_template.txt
- For 510: Google cloud service account credentials are accessible [here](https://console.cloud.google.com/apis/credentials?project=eth-conflict-tracker&folder=&organizationId=&supportedpurview=project), but create a new project if needed!. Login credentials in Bitwarden

The rumor-tracker in confgured via one configuration file, which is structured structured as follows

### with Docker
1. Install [Docker](https://www.docker.com/get-started)
3. Build the docker image from the root directory
```
docker build -t rodekruis/rumor-tracker .
```
4. Run and access the docker container
```
docker run -it --entrypoint /bin/bash rodekruis/rumor-tracker
```
5. Check that everything is working by running the pipeline (see [Usage](https://github.com/rodekruis/news-tracker-ethiopia#usage) below)
6. You can now use the rumor-tracker as a dockerized app in your cloud provider, e.g. [Azure Logic App](https://docs.google.com/document/d/182aQPVRZkXifHDNjmE66tj5L1l4IvAt99rxBzpmISPU/edit?usp=sharing)

### Manual Setup
TBI

## Usage
```
Usage: run-pipeline [OPTIONS]

Options:
  --config                    configuration file (json)
  --help                      show this message and exit
  ```
