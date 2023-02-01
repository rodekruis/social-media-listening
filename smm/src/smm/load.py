from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient
import logging
import pyodbc


supported_storages = ["local", "Azure SQL Database", "Azure Blob Storage"]


class StorageSecrets:
    """
    Secrets (API keys, tokens, etc.) for input/output data storage
    """
    def __init__(self,
                 keyvault_url=None,
                 database_secret=None):

        self.keyvault_url = keyvault_url
        self.database_secret = database_secret
        #TODO: automatically get secrets from environment variables (?)

        def get_secret(self):
            if self.secret_location == "env":
                sm_secrets = self.get_secret_env()
            if self.secret_location == "file":
                sm_secrets = self.get_secret_file()
            if self.secret_location == "azure":
                sm_secrets = self.get_secret_azure()
            return sm_secrets

        def get_secret_env(self):
            secret_file = "../credentials/env"  # TODO: where to input the file directory
            sm_secrets = dotenv_values(secret_file)
            return sm_secrets

        def get_secret_file(self):
            secret_file = "../credentials/secrets.json"
            with open(secret_file, "r") as secret_file:
                if secret_file.endswith("json"):
                    sm_secrets = json.load(secret_file)  # TODO: set constraints of json file format?
                elif secret_file.endswith("yaml"):
                    sm_secrets = yaml.load(secret_file, Loader=yaml.FullLoader)
                return sm_secrets

        def get_secret_azure(self):
            secret_name_az = ""
            sm_secrets = self.load.get_secret_keyvault(secret_name_az)
            sm_secrets = json.loads(sm_secrets)
            return sm_secrets


class Storage:
    """
    input/output data storage
    """
    def __init__(self, name=None, secrets=None):
        if name is None:
            self.name = "local"
        if name not in supported_storages:
            raise ValueError(f"Storage {name} is not supported."
                             f"Supported storages are {supported_storages}")
        else:
            self.name = name
        self.secrets = self.set_secrets(secrets)

    def set_secrets(self, secrets):
        if not isinstance(secrets, StorageSecrets):
            raise TypeError(f"invalid format of secrets, use load.StorageSecrets")

        if self.check_secrets(secrets):
            self.secrets = secrets

    def check_secrets(self):
        #TODO check that right secrets are filled for data source
        return True


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

        # Reformat dataframe
        df_messages = self._reformat_messages(df_messages)

        if self.storage.name == "local":
            # save locally
            os.makedirs(f"./{directory}", exist_ok=True)
            messages_path = f"./{directory}/{filename}.csv"
            df_messages.to_csv(messages_path, index=False, encoding="utf-8")

        elif self.storage.name == "Azure SQL Database":
            # save to Azure SQL Database
            if not self.storage.check_secrets():
                raise ValueError("no storage secrets found")
            else:
                self.save_to_db(df_messages)

        elif self.storage.name == "Azure Blob Storage":
            # save to Azure Blob Storage
            if not self.storage.check_secrets():
                raise ValueError("no storage secrets found")
            else:
                # save locally
                os.makedirs(f"./{directory}", exist_ok=True)
                messages_path = f"./{directory}/{filename}.csv"
                df_messages.to_csv(messages_path, index=False, encoding="utf-8")
                # upload local file to AZ
                directory_blob = f"{directory}/{filename}.csv"
                self.upload_blob(messages_path, directory_blob)

        else:
            raise ValueError(f"storage {self.storage.name} is not supported."
                             f"Supported storages are {supported_storages}")

    def save_wordfrequencies(self, frequencies):

        if not self.storage.check_secrets():
            raise ValueError("no storage secrets found")

        if self.storage.name == "local":
            # save locally
        elif self.storage.name == "Azure SQL Database":
            # save to Azure SQL Database
        elif self.storage.name == "Azure Blob Storage":
            # save to Azure Blob Storage
        else:
            raise ValueError(f"storage {self.storage.name} is not supported."
                             f"Supported storages are {supported_storages}")

    def get_messages(self):
        if not self.storage.check_secrets():
            raise ValueError("no storage secrets found")

        if self.storage.name == "local":
            # load locally
        elif self.storage.name == "Azure SQL Database":
            # load from Azure SQL Database
        elif self.storage.name == "Azure Blob Storage":
            # load from Azure Blob Storage
        else:
            raise ValueError(f"storage {self.storage.name} is not supported."
                             f"Supported storages are {supported_storages}")
        return messages

    def _reformat_messages(self, df):

        df.drop(columns=['Info'], inplace=True)
        static_columns = [col for col in df.columns if col != 'classification']

        # Explode classification to multiple rows
        df = df.explode['classification']
        df['classification'] = df['classification'].apply(lambda x: x['class'])
        #TODO:Multiple translations
        df['translation'] = df['translation'].apply(lambda x: x['text'])

        # Classes to columns
        pd.concat(
            [
                df[static_columns],
                pd.get_dummies(df['classification'])
            ],
            1
        )

        # Merge duplicate rows created by explode of classification
        class_columns = [col for col in df.columns if col not in static_columns]

        agg_dict = {}
        for col in class_columns:
            agg_dict[col]='sum'

        df = df.groupby(static_columns).agg(agg_dict).reset_index()

        return df


    def get_secret_keyvault(self, secret_name):
        az_credential = DefaultAzureCredential()
        kv_secretClient = SecretClient(vault_url=self.key_vault, 
                                        credential=az_credential)
        secret_value = kv_secretClient.get_secret(secret_name).value
        return secret_value

    def get_blob_service_client(self, blob_path):
        blob_secret_name = 'blobstorage-secret' # TODO: where to input secret name
        blobstorage_secrets = self.get_secret_keyvault(blob_secret_name)
        blobstorage_secrets = json.loads(blobstorage_secrets)
        blob_service_client = BlobServiceClient.from_connection_string(blobstorage_secrets['connection_string'])
        container = blobstorage_secrets['container']
        return blob_service_client.get_blob_client(container=container, blob=blob_path)


    def get_table_service_client(self, table):
        table_secret_name = 'table-secret'
        table_secret = self.get_secret_keyvault(table_secret_name)
        table_service_client = TableServiceClient.from_connection_string(table_secret)
        return table_service_client.get_table_client(table_name=table)

    def remove_pii(df, text_columns, config):


    def connect_to_db(self):
        # Get credentials
        database_secret_name = "azure-database-secret" # TODO: 
        database_secret = self.get_secret_keyvault(database_secret_name)
        database_secret = json.loads(database_secret)

        try:
            # Connect to db
            driver = '{ODBC Driver 18 for SQL Server}'
            connection = pyodbc.connect(
                f'DRIVER={driver};'
                f'SERVER=tcp:{database_secret["SQL_DB_SERVER"]};'
                f'PORT=1433;DATABASE={database_secret["SQL_DB"]};'
                f'UID={database_secret["SQL_USER"]};'
                f'PWD={database_secret["SQL_PASSWORD"]}'
            )
            cursor = connection.cursor()
            logging.info("Successfully connected to database")
        except pyodbc.Error as error:
            logging.info("Failed to connect to database {}".format(error))
        return connection, cursor


    def read_db(self, sm_source, start_date, end_date):
        db_table_name = ""
        connection, cursor = self.connect_to_db()
        query = f"""SELECT * \
            FROM {db_table_name} \
            WHERE sm_code = '{sm_source}' \
            AND country = '{config['country-code']}' \ 
            AND date \
            BETWEEN '{start_date}' AND '{end_date}' \
            """ # TODO: get country code from message
        try:
            df_messages = pd.read_sql(query, connection)
            logging.info(f"Succesfully retrieved {len(df_messages)} {sm_source} messages \n from {start_date} to {end_date} from table {db_table_name}")
        except Exception:
            df_messages = None
            logging.error(f"Failed to retrieve SQL table {db_table_name}: {e}")
            logging.info(f"query: {query}")
        finally:
            cursor.close()
            connection.close()
            logging.info("AZ Database connection is closed")
        # TODO: map df to msg object here?
        return df_messages


    def save_to_db(self, data):
        data_final = self._prepare_messages_for_db(data)
        sm_source = "" # TODO: extract sm_source from the data_final
        current_datetime = datetime.datetime.now()
        if not data_final.empty:
            # Make connection to Azure database
            db_table_name = ""
            connection, cursor = self.connect_to_db()

            try:
                mySql_insert_query = f"""INSERT INTO {db_table_name} (id_post, country, sm_code, source, datetime_scraped,\
                datetime, date, post, text_post, text_reply) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""" # TODO: rename sm_code, source columns for consistency

                for idx, row in data_final.iterrows():
                    cursor.execute(
                        mySql_insert_query,
                        row['id_post'],
                        config['country-code'], # TODO: get country code from message
                        sm_source,
                        row['source'],
                        current_datetime,
                        row['datetime'],
                        row['date'],
                        row['post'],
                        row['text_post'],
                        row['text_reply']
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


    def _prepare_messages_for_db(self, data):
        # Prepare for storing in Azure db
        # TODO: remap msg instant to pd dataframe?, identify db collumns from msg instant attribute
        data['post'] = np.where(
            data['post'],
            1,
            0
        )

        data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
        data['datetime'] = data['datetime'].dt.tz_localize(None)
        data["id_post"] = data["id_post"].astype("int64")
        data = data.astype(object).where(pd.notnull(data), None)

        # Read data from database to check for duplicates
        logging.info("Check for duplicates in database")
        data['in_db'] = False

        start_date = min(data['date'])
        end_date = max(data['date'])

        data_in_db = self.read_db(sm_source, start_date, end_date, config)

        if not data_in_db.empty:
            # Make sure format of new data and data in database are the same
            data_in_db.drop(columns=['ID'], inplace=True)

            data_in_db["datetime"] = pd.to_datetime(data_in_db["datetime"], format='%Y-%m-%d %H:%M:%S')
            data_in_db["id_post"] = data_in_db["id_post"].astype("int64")

            data_in_db = data_in_db.astype(object).where(pd.notnull(data_in_db), None)

            data_in_db['in_db'] = True

            # Concatenate new data and data already in database and drop duplicates
            df_concat = pd.concat([data, data_in_db], ignore_index=True)
            df_concat.drop_duplicates(
                subset=['id_post', 'source', 'datetime', 'text_post', 'text_reply'],
                keep=False,
                inplace=True
            )

            # Remove all rows there were read from database
            df_concat = df_concat[~df_concat['in_db']]
            data_final = df_concat.drop(columns=['in_db'])
        else:
            data_final = data.drop(columns=['in_db'])

        return data_final


    def upload_blob(self, file_dir_local, file_dir_blob):
        blob_client = self.get_blob_service_client(file_dir_blob)
        with open(file_dir_local, "rb") as upload_file:
            blob_client.upload_blob(upload_file, overwrite=True)


    def download_blob(self, blob_path, local_path=None):
        blob_client = self.get_blob_service_client(blob_path)
        if local_path is None:
            local_path = blob_path
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

