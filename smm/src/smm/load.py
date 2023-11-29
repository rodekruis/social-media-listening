from datetime import datetime
import pandas as pd
import os
from azure.storage.blob import BlobServiceClient
import logging
import pyodbc
import sqlalchemy as db
import urllib
import ast
from smm.secrets import Secrets
from smm.message import Message

supported_storages = ["local", "Azure SQL Database", "Azure Blob Storage"]


class Storage:
    """
    input/output data storage
    """
    # TODO: Do we need this?
    def __init__(self, name=None):
        if name is None:
            self.name = "local"
        else:
            self.name = name


class Load:
    """
    load data from/into a data storage
    """

    def __init__(self, storage_name=None, secrets: Secrets = None):
        self.storage = self.set_storage(storage_name)
        self.secrets = self.set_secrets(secrets)

    def set_storage(self, storage_name=None):
        if storage_name is not None:
            if hasattr(self, "storage"):
                if self.storage.name == storage_name:
                    logging.info(f"Storage already set to {storage_name}")
            else:
                if storage_name not in supported_storages:
                    raise ValueError(f"Storage {storage_name} is not supported."
                                     f"Supported storages are {supported_storages}")
                else:
                    return Storage(storage_name)
        else:
            raise ValueError(f"Storage not specified; provide one of {supported_storages}")

    def set_secrets(self, secrets):
        if not isinstance(secrets, Secrets):
            raise TypeError(f"invalid format of secrets, use secrets.Secrets")

        missing_secrets = []
        if self.storage.name == "Azure SQL Database":
            missing_secrets = secrets.check_secrets(
                [
                    "sql_db_server",
                    "sql_db",
                    "sql_user",
                    "sql_password",
                    "table_name"
                ]
            )
        if self.storage.name == "Azure Blob Storage":
            missing_secrets = secrets.check_secrets(
                [
                    "container",
                    "connection_string"
                ]
            )

        if missing_secrets:
            raise Exception(f"Missing secret(s) {missing_secrets} for storage of type {self.storage.name}")
        else:
            return secrets

    def save_messages(self, messages, directory=None, filename=None):

        # Read messages to dataframe
        df_messages = pd.DataFrame.from_records([msg.to_dict() for msg in messages])
        df_messages["info"] = df_messages["info"].apply(lambda x: str(x))
        df_messages["translations"] = df_messages["translations"].apply(lambda x: str(x))
        df_messages["classifications"] = df_messages["classifications"].apply(lambda x: str(x))
        df_messages["datetime_"] = df_messages["datetime_"].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))


        if self.storage.name == "local":
            # save locally
            os.makedirs(f"./{directory}", exist_ok=True)
            messages_path = f"./{directory}/{filename}"
            df_messages.to_csv(messages_path, index=False, encoding="utf-8")
            logging.info(f"Successfully saved messages at {messages_path}")

        elif self.storage.name == "Azure Blob Storage":
            # save locally
            os.makedirs(f"./{directory}", exist_ok=True)
            messages_path = f"./{directory}/{filename}"
            df_messages.to_csv(messages_path, index=False, encoding="utf-8")

            # upload local file to AZ
            directory_blob = f"{directory}/{filename}"
            try:
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

    def save_wordfrequencies(self, frequencies, directory, filename):

        if self.storage.name == "local":
            # save locally
            os.makedirs(f"./{directory}", exist_ok=True)
            frequencies_path = f"./{directory}/{filename}.csv"
            frequencies.to_csv(frequencies_path, index=False, encoding="utf-8")
        else:
            raise ValueError(f"storage {self.storage.name} is not supported."
                             f"Supported storages are {supported_storages}")

    def get_messages(self,
                     directory=None,
                     filename=None,
                     start_date=None,
                     end_date=None,
                     country=None,
                     source=None
                     ):

        if self.storage.name == "local":
            # load locally
            messages_path = f"./{directory}/{filename}"
            df_messages = pd.read_csv(messages_path)
            
        elif self.storage.name == "Azure SQL Database":
            # load from Azure SQL Database
            if start_date is None or end_date is None:
                raise Exception(f"Please provide an start and end date for reading from Azure SQL Database")
            if country is None:
                raise Exception(f"Please provide a country for reading from Azure SQL Database")
            if source is None:
                raise Exception(f"Please provide a source for reading from Azure SQL Database")
            df_messages = self.read_db(start_date, end_date, country, source)
            
        elif self.storage.name == "Azure Blob Storage":
            messages_path = f"./{directory}/{filename}"

            # load from Azure Blob Storage
            blob_client = self.get_blob_service_client(messages_path)
            with open(messages_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            df_messages = pd.read_csv(directory)
            
        else:
            raise ValueError(f"storage {self.storage.name} is not supported."
                             f"Supported storages are {supported_storages}")

        # Convert dataframe of messages to list of message objects
        messages = []
        for idx, row in df_messages.iterrows():
            # Process
            classifications = None
            if row['classifications']:
                classifications = ast.literal_eval(row['classifications'])

            translations = None
            if row['translations']:
                translations = ast.literal_eval(row['translations'])

            info = None
            if row['info']:
                info = ast.literal_eval(row['info'])

            # Initiate Message objects
            message = Message(
                row['id_'],
                row['datetime_'],
                row['datetime_scraped_'],
                row['country'],
                row['source'],
                row['text'],
                row['group'],
                row['reply'],
                row['reply_to'],
                translations,
                info,
                classifications
            )
            messages.append(message)

        return messages

    def save_to_db(self, data):
        data_final = self._prepare_messages_for_db(data)
        current_datetime = datetime.now()
        db_table_name = self.secrets.get_secret("table_name")
        if not data_final.empty:
            # Make connection to Azure database
            engine, connection = self.connect_to_db()

            messages_table = db.Table('messages_new', db.MetaData(schema="smm"), autoload=True, autoload_with=engine)

            try:
                for idx, row in data_final.iterrows():
                    connection.execute(
                        db.insert(messages_table),
                        [{
                            "id_": row['id_'],
                            "datetime_": row['datetime_'],
                            "datetime_scraped_": current_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                            "country": row['country'],
                            "source": row['source'],
                            "text": row['text'],
                            "group": row['group'],
                            "reply": row['reply'],
                            "reply_to": row['reply_to'],
                            "translations": row['translations'],
                            "info": row['info'],
                            "classifications": row['classifications']
                        }]
                    )

                logging.info(f"Successfully inserted {len(data_final)} entries into table {db_table_name}, "
                             f"{len(data) - len(data_final)} duplicates already in database")

            except pyodbc.Error as error:
                logging.warning("Failed to insert into SQL table {}".format(error))

            finally:
                connection.close()
                engine.dispose()
                logging.info("Pyodbc connection is closed")
        else:
            logging.info(f"All scraped messages already existing in table {db_table_name}")

    def read_db(self, start_date, end_date, country, source):
        # Connect to db
        db_table_name = self.secrets.get_secret("table_name")  
        engine, connection = self.connect_to_db()

        messages_table = db.Table('messages_new', db.MetaData(schema="smm"), autoload=True,  autoload_with=engine)

        # prepare query
        query = messages_table.select().where(
            db.and_(
                messages_table.columns.source == source,
                messages_table.columns.country == country,
                messages_table.columns.datetime_.between(start_date, end_date)
            )
        )
        try:
            output = connection.execute(query)
            results = output.fetchall()
            df_messages = pd.DataFrame(results)
            df_messages.columns = results[0].keys()
            logging.info(
                f"Successfully retrieved {len(df_messages)} {source} messages \n from {start_date} to {end_date} from table {db_table_name}")
        except Exception as e:
            df_messages = None
            logging.error(f"Failed to retrieve SQL table {db_table_name}: {e}")
            logging.info(f"query: {query}")
        finally:
            connection.close()
            engine.dispose()
            logging.info("AZ Database connection is closed")
        return df_messages

    def _prepare_messages_for_db(self, data):
        # Prepare for storing in Azure db
        data['datetime_'] = pd.to_datetime(data['datetime_'], format='%Y-%m-%d %H:%M:%S')
        data['datetime_'] = data['datetime_'].dt.tz_localize(None)
        data["id_"] = data["id_"].astype("int64")
        data = data.astype(object).where(pd.notnull(data), None)

        # Read data from database to check for duplicates
        logging.info("Check for duplicates in database")
        data['in_db'] = False

        start_date = min(data['datetime_'])
        end_date = max(data['datetime_'])
        country = data['country'].unique()[0]
        source = data['source'].unique()[0]

        data_in_db = self.read_db(start_date, end_date, country, source)

        if data_in_db is not None and not data_in_db.empty:
            # Make sure format of new data and data in database are the same
            data_in_db.drop(columns=['ID'], inplace=True)

            data_in_db["datetime_"] = pd.to_datetime(data_in_db["datetime_"], format='%Y-%m-%d %H:%M:%S')
            data_in_db["id_"] = data_in_db["id_"].astype("int64")

            data_in_db = data_in_db.astype(object).where(pd.notnull(data_in_db), None)

            data_in_db['in_db'] = True

            # Concatenate new data and data already in database and drop duplicates
            df_concat = pd.concat([data, data_in_db], ignore_index=True)
            df_concat.drop_duplicates(
                subset=['id_', 'source', 'datetime_', 'text'],
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
        logging.info("Successfully uploaded to Azure Blob Storage")

    def download_blob(self, blob_path):
        blob_client = self.get_blob_service_client(blob_path)
        with open(blob_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

    def connect_to_db(self):
        # Connect to db
        try:
            driver = '{ODBC Driver 18 for SQL Server}'
            params = urllib.parse.quote_plus(
                f'Driver={driver};'
                f'Server=tcp:{self.secrets.get_secret("sql_db_server")},1433;'
                f'Database={self.secrets.get_secret("sql_db")};'
                f'Uid={self.secrets.get_secret("sql_user")};'
                f'Pwd={self.secrets.get_secret("sql_password")};'
                f'Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
            )
            conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
            engine = db.create_engine(conn_str)
            connection = engine.connect()
            logging.info("Successfully connected to database")
            return engine, connection
        except pyodbc.Error as error:
            logging.info("Failed to connect to database {}".format(error))
