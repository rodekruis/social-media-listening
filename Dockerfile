FROM python:3.7.10-buster

RUN apt-get update && \
	apt-get install -y python3-pip && \
	ln -sfn /usr/bin/python3.7 /usr/bin/python && \
	ln -sfn /usr/bin/pip3 /usr/bin/pip

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN deps='build-essential gdal-bin python-gdal libgdal-dev kmod wget apache2 libenchant1c2a libspatialindex-dev' && \
	apt-get update && \
	apt-get install -y $deps

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
	curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
	apt-get update && \
	ACCEPT_EULA=Y apt-get install -y msodbcsql18

RUN pip install --upgrade pip && \
	pip install GDAL==$(gdal-config --version)

# install spaCy modules for NLP
RUN python -m spacy download uk_core_news_sm
RUN python -m spacy download ru_core_news_sm
RUN python -m spacy download en_core_web_sm

ADD config /config

WORKDIR /pipeline
ADD pipeline .
RUN pip install .
