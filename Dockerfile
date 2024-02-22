FROM python:3.11-slim

# install ODBC Driver for SQL Server
RUN deps='curl gnupg gnupg2' && \
	apt-get update && \
	apt-get install -y $deps
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
	curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
	apt-get update && \
	ACCEPT_EULA=Y apt-get install -y msodbcsql18
RUN pip install poetry

# clean up
RUN set -ex apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# add credentials and install SML pipeline
WORKDIR .
COPY credentials /credentials
COPY pyproject.toml poetry.lock /
RUN poetry install --no-root --no-interaction
COPY sml /sml
COPY "test_telegam_pipeline_copy.py" .

ENTRYPOINT ["poetry", "run", "python", "-m", "test_telegam_pipeline_copy.py"]