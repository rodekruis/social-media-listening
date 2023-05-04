FROM python:3.9-slim-bullseye
RUN deps='enchant-2' && \
	apt-get update && \
	apt-get install -y $deps

# install spaCy modules for NLP
RUN pip install -U pip setuptools wheel
RUN pip install spacy
RUN python -m spacy download uk_core_news_sm
RUN python -m spacy download ru_core_news_sm
RUN python -m spacy download en_core_web_sm

# install ODBC Driver for SQL Server
RUN deps='curl gnupg gnupg2' && \
	apt-get update && \
	apt-get install -y $deps
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
	curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
	apt-get update && \
	ACCEPT_EULA=Y apt-get install -y msodbcsql18

# install GDAL and fiona for geopandas
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN apt-get install -y build-essential gdal-bin libgdal-dev binutils libproj-dev
RUN pip install setuptools==58.0.0
RUN pip install GDAL==$(gdal-config --version) fiona

# add config files and credentials
ADD config /config
ADD credentials /credentials

# install SML pipeline
WORKDIR /pipeline
ADD pipeline .
RUN pip install .


#FROM python:3.7.10-buster
#
#RUN apt-get update && \
#	apt-get install -y python3-pip && \
#	ln -sfn /usr/bin/python3.7 /usr/bin/python && \
#	ln -sfn /usr/bin/pip3 /usr/bin/pip
#
#ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
#ENV C_INCLUDE_PATH=/usr/include/gdal
#
#RUN deps='build-essential gdal-bin python-gdal libgdal-dev kmod wget apache2 libenchant1c2a libspatialindex-dev' && \
#	apt-get update && \
#	apt-get install -y $deps
#
#RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
#	curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
#	apt-get update && \
#	ACCEPT_EULA=Y apt-get install -y msodbcsql18
#
#RUN pip install --upgrade pip && \
#	pip install GDAL==$(gdal-config --version)
#
## install spaCy modules for NLP
#RUN pip install -U spacy
#RUN python -m spacy download en_core_web_trf
#RUN python -m spacy download ru_core_news_lg
#RUN python -m spacy download uk_core_news_trf
#
#ADD config /config
#ADD credentials /credentials
#
#WORKDIR /pipeline
#ADD pipeline .
#RUN pip install .

##########################################################################################

#FROM python:3.8-slim-bullseye
#
#RUN apt-get update
#
#ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
#ENV C_INCLUDE_PATH=/usr/include/gdal
#
#RUN deps='build-essential gdal-bin libgdal-dev kmod wget apache2 libspatialindex-dev ca-certificates wget curl' && \
#	apt-get update && \
#	apt-get install -y $deps
#
#RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
#	curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
#	apt-get update && \
#	ACCEPT_EULA=Y apt-get install -y msodbcsql18
#
## install spaCy modules for NLP
#RUN pip install -U pip setuptools wheel
#RUN pip install -U spacy==3.4.1
#RUN python -m spacy download uk_core_news_sm
#RUN python -m spacy download ru_core_news_sm
#RUN python -m spacy download en_core_web_sm
#
#ADD config /config
#ADD credentials /credentials
#
#WORKDIR /pipeline
#ADD pipeline .
#RUN pip install .
