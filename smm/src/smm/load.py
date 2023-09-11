from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json
from azure.storage.blob import BlobServiceClient
import logging
import pyodbc
from secrets import Secrets
from message import Message


supported_storages = ["local", "Azure SQL Database", "Azure Blob Storage"]


class Storage:
    """
    input/output data storage
    """
    def __init__(self, name=None, secrets: Secrets = None):
        if name is None:
            self.name = "local"
        if name not in supported_storages:
            raise ValueError(f"Storage {name} is not supported."
                             f"Supported storages are {supported_storages}")
        else:
            self.name = name
        self.secrets = self.set_secrets(secrets)

    def set_secrets(self, secrets):
        if not isinstance(secrets, Secrets):
            raise TypeError(f"invalid format of secrets, use secrets.Secrets")

        missing_secrets = []
        if self.name == "Azure SQL Database":
            missing_secrets =self.secrets.check_secrets(["SQL_DB_SERVER", "SQL_DB", "SQL_USER", "SQL_PASSWORD"])
        if self.name == "Azure Blob Storage":
            missing_secrets = self.secrets.check_secrets(["container", "connection_string"])

        if missing_secrets:
            raise Exception(f"Missing secret(s) {missing_secrets} for storage of type {self.name}")
        else:
            self.secrets = secrets


class Load:
    """
    load data from/into a data storage
    """
    def __init__(self, storage_name=None, secrets=None):
        self.set_storage(storage_name, secrets)

    def set_storage(self, storage_name=None, secrets=None):
        if storage_name is not None:
            if self.storage.name == storage_name:
                logging.info(f"Storage already set to {storage_name}")
            else:
                self.storage = Storage(storage_name, secrets)
        else:
            raise ValueError(f"Storage not specified; provide one of {supported_storages}")

    def save_messages(self, messages, directory, filename):

        # Read messages to dataframe
        df_messages = pd.DataFrame.from_records([msg.to_dict() for msg in messages])

        if self.storage.name == "local":
            # save locally
            os.makedirs(f"./{directory}", exist_ok=True)
            messages_path = f"./{directory}/{filename}.csv"
            df_messages.to_csv(messages_path, index=False, encoding="utf-8")
            logging.info(f"Succesfully saved messages at {messages_path}")

        elif self.storage.name == "Azure Blob Storage":
            # save locally
            os.makedirs(f"./{directory}", exist_ok=True)
            messages_path = f"./{directory}/{filename}.csv"
            df_messages.to_csv(messages_path, index=False, encoding="utf-8")

            # upload local file to AZ
            directory_blob = f"{directory}/{filename}.csv"
            try
                self.upload_blob(messages_path, directory_blob)
            except Exception as e:
                logging.error(f"Failed uploading to Azure Blob Service: {e}")


        elif self.storage.name == "Azure SQL Database":
            # save to Azure SQL Database
           try:
                self.save_to_db(df_messages)
           except Exception as e:
               logging.error(f"Failed storing in Azure SQL Database: {e}")

        else:
            raise ValueError(f"storage {self.storage.name} is not supported."
                             f"Supported storages are {supported_storages}")

    def save_wordfrequencies(self, frequencies):
        # TODO: Can only be saved locally for now
        # TODO: Check that this is indeed not necessary anymore
        if not self.storage.check_secrets():
            raise ValueError("no storage secrets found")

        if self.storage.name == "local":
            # save locally
        else:
            raise ValueError(f"storage {self.storage.name} is not supported."
                             f"Supported storages are {supported_storages}")

    def get_messages(self, directory, filename):
        if not self.storage.check_secrets():
            raise ValueError("no storage secrets found")

        if self.storage.name == "local":
            # load locally
            messages_path = f"./{directory}/{filename}.csv"
            df_messages = pd.read_csv(messages_path)
        elif self.storage.name == "Azure SQL Database":
            # TODO: Finalize
            # load from Azure SQL Database
        elif self.storage.name == "Azure Blob Storage":
            # TODO: Finalize
            # load from Azure Blob Storage
        else:
            raise ValueError(f"storage {self.storage.name} is not supported."
                             f"Supported storages are {supported_storages}")

        # Convert dataframe of messages to list of message objects
        messages = []
        for idx, row in df_messages.iterrows():
            #TODO: Convert translations, info and classifications from string to list of strings
            message = Message(
                row['_id'],
                row['datetime_'],
                row['datetime_scraped_'],
                row['country'],
                row['source'],
                row['text'],
                row['group'],
                row['reply'],
                row['reply_to'],
                row['translations'],
                row['info'],
                row['classifications']
            )
            messages.append(message)

        return messages


    def save_to_db(self, data):
        data_final = self._prepare_messages_for_db(data)
        current_datetime = datetime.datetime.now()
        if not data_final.empty:
            # Make connection to Azure database
            db_table_name = "smm.messages_new"  # TODO: Replace this to config variable?
            connection, cursor = self.connect_to_db()

            try:
                mySql_insert_query = f"""INSERT INTO {db_table_name} (id_, datetime_, datetime_scraped_, \
                    country, source, text, group, reply, reply_to, translations, info, classifications) \
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

                for idx, row in data_final.iterrows():
                    cursor.execute(
                        mySql_insert_query,
                        row['id_'],
                        row['datetime_'],
                        current_datetime,
                        row['country'], # TODO: convert country to country code?
                        row['source'],
                        row['text'],
                        row['group'],
                        row['reply'],
                        row['reply_to'],
                        row['translations'],
                        row['info'],
                        row['classifications']
                    )
                    connection.commit()
                logging.info(f"Succesfully inserted {len(data_final)} entries into table {db_table_name}, "
                            f"{len(data) - len(data_final)} duplicates already in database")

            except pyodbc.Error as error:
                logging.warning("Failed to insert into SQL table {}".format(error))

            finally:
                cursor.close()
                connection.close()
                logging.info("Pyodbc connection is closed")
        else:
            logging.info(f"All scraped messages already existing in table {db_table_name}")


    def read_db(self, start_date, end_date, country, source):
        # Connect to db
        db_table_name = "smm.messages_new"  #TODO: Replace this to config variable?
        connection, cursor = self.connect_to_db()

        # prepare query
        countries = "(" + ", ".join(country) + ")"

        query = f"""SELECT * \
            FROM {db_table_name} \
            WHERE source = '{source}' \
            AND country IN {countries} \
            AND datetime_ \
            BETWEEN '{start_date}' AND '{end_date}' \
        """
        try:
            df_messages = pd.read_sql(query, connection)
            logging.info(f"Succesfully retrieved {len(df_messages)} {source} messages \n from {start_date} to {end_date} from table {db_table_name}")
        except Exception:
            df_messages = None
            logging.error(f"Failed to retrieve SQL table {db_table_name}: {e}")
            logging.info(f"query: {query}")
        finally:
            cursor.close()
            connection.close()
            logging.info("AZ Database connection is closed")
        return df_messages


    def _prepare_messages_for_db(self, data):
        # Prepare for storing in Azure db
        data['datetime_'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
        data['datetime_'] = data['datetime'].dt.tz_localize(None)
        data["id_"] = data["id_"].astype("int64")
        data = data.astype(object).where(pd.notnull(data), None)

        # Read data from database to check for duplicates
        logging.info("Check for duplicates in database")
        data['in_db'] = False

        start_date = min(data['datetime_'])
        end_date = max(data['datetime_'])
        country = data['country'].unique()
        source = data['source'].unique()

        data_in_db = self.read_db(start_date, end_date, country, source)

        if not data_in_db.empty:
            # Make sure format of new data and data in database are the same
            data_in_db.drop(columns=['ID'], inplace=True)

            data_in_db["datetime_"] = pd.to_datetime(data_in_db["datetime_"], format='%Y-%m-%d %H:%M:%S')
            data_in_db["id_"] = data_in_db["id_"].astype("int64")

            data_in_db = data_in_db.astype(object).where(pd.notnull(data_in_db), None)

            data_in_db['in_db'] = True

            # Concatenate new data and data already in database and drop duplicates
            df_concat = pd.concat([data, data_in_db], ignore_index=True)
            df_concat.drop_duplicates(
                subset=['id_', 'source', 'datetime', 'text_post', 'text_reply'],
                keep=False,
                inplace=True
            )

            # Remove all rows there were read from database
            df_concat = df_concat[~df_concat['in_db']]
            data_final = df_concat.drop(columns=['in_db'])
        else:
            data_final = data.drop(columns=['in_db'])

        return data_final

    def get_blob_service_client(self, blob_path):
        blob_service_client = BlobServiceClient.from_connection_string(self.secrets.get_secret("connection_string"))
        container = self.secrets.get_secret("container")
        return blob_service_client.get_blob_client(container=container, blob=blob_path)

    def upload_blob(self, file_dir_local, file_dir_blob):
        blob_client = self.get_blob_service_client(file_dir_blob)
        with open(file_dir_local, "rb") as upload_file:
            blob_client.upload_blob(upload_file, overwrite=True)
        logging.info("Succesfully uploaded to Azure Blob Storage")


    def download_blob(self, blob_path, local_path=None):
        blob_client = self.get_blob_service_client(blob_path)
        if local_path is None:
            local_path = blob_path
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())


    def connect_to_db(self):
        # Connect to db
        try:
            driver = '{ODBC Driver 18 for SQL Server}'
            connection = pyodbc.connect(
                f'DRIVER={driver};'
                f'SERVER=tcp:{self.secrets.get_secret("SQL_DB_SERVER")};'
                f'PORT=1433;DATABASE={self.secrets.get_secret("SQL_DB")};'
                f'UID={self.secrets.get_secret("SQL_USER")};'
                f'PWD={self.secrets.get_secret("SQL_PASSWORD")}'
            )
            cursor = connection.cursor()
            logging.info("Successfully connected to database")
        except pyodbc.Error as error:
            logging.info("Failed to connect to database {}".format(error))
        return connection, cursor

