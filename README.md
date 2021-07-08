# news-tracker-ethiopia

Track specific topic(s) in news and social media.

Built to support Ethiopian Red Cross Society (ERCS).

## Setup
Generic requirements:
-   [Twitter developer account](https://developer.twitter.com/en/apply-for-access)
-   OPTIONAL (translate): [Google Cloud account](https://cloud.google.com/)
-   OPTIONAL (upload to Azure datalake): [Azure account](https://azure.microsoft.com/en-us/get-started/) and 

For 510: Google cloud service account credentials are accessible [here](https://console.cloud.google.com/apis/credentials?project=eth-conflict-tracker&folder=&organizationId=&supportedpurview=project), login credentials in Bitwarden

### with Docker
1. Install [Docker](https://www.docker.com/get-started)
3. Download vector input data from [here](https://rodekruis.sharepoint.com/sites/510-CRAVK-510/_layouts/15/guestaccess.aspx?docid=09ee1386e97b54b7cbd9399c730181efa&authkey=AelH_jSEguHCrGEp5gh2oyI&expiration=2022-07-04T22%3A00%3A00.000Z&e=OBsIge), unzip and move it to
```
vector/
```
5. Copy your Google, Twitter and Azure credentials in
```
credentials/
```
3. Build the docker image from the root directory
```
docker build -t rodekruis/news-tracker-ethiopia .
```
4. Run and access the docker container
```
docker run -it --entrypoint /bin/bash rodekruis/news-tracker-ethiopia
```
5. Check that everything is working by running the pipeline (see [Usage](https://github.com/rodekruis/news-tracker-ethiopia#usage) below)


### Manual Setup
TBI

## Usage
```
Usage: run-pipeline [OPTIONS]

Options:
  --translate                    translate text with Google API
  --datalake                     upload to Azure datalake
  --help                         show this message and exit
  ```
