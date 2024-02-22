# social-media-listening
TBA

## Setup

### Prequisites
- Ensure files `poetry.lock` and `pyproject.toml` available at root for caching dependencies
- [ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16) if storing data in Microsoft Azure SQL Server 

### Set up locally (with `poetry`)
1. Install [Python Poetry](https://python-poetry.org/docs/)
2. Run:
```
poetry run python -m xxxx.py
```

### Set up with Docker 
1. Install [Docker](https://www.docker.com/get-started)
2. Build the docker image from the root directory
```
docker build -t rodekruis/social- .
```
3. Run the dockerised pipeline
```
docker run -it rodekruis/sml --entrypoint
```
Or enter the docker image interactively
```
docker run -it --entrypoint /bin/bash rodekruis/sml
```