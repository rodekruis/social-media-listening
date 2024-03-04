from datetime import datetime
import pandas as pd
import os
from azure.storage.blob import BlobServiceClient
import logging
import pyodbc
import sqlalchemy as db
import urllib
import ast
import argilla as rg
import numpy as np
from sml.secrets import Secrets
from sml.message import Message

supported_storages = ["local", "Azure SQL Database", "Azure Blob Storage"]


def _messages_to_df(messages):
    df_messages = pd.DataFrame.from_records([msg.to_dict() for msg in messages])
    df_messages["info"] = df_messages["info"].apply(lambda x: str(x))
    df_messages["translations"] = df_messages["translations"].apply(lambda x: str(x))
    df_messages["classifications"] = df_messages["classifications"].apply(lambda x: str(x))
    df_messages["datetime_"] = df_messages["datetime_"].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    return df_messages


class Load:
    """
    load data from/into a data storage
    """

    def __init__(self, secrets: Secrets = None):
        self.storage = "local"
        self.secrets = None
        if secrets is not None:
            self.set_secrets(secrets)
            
    def set_secrets(self, secrets):
        if not isinstance(secrets, Secrets):
            raise TypeError(f"invalid format of secrets, use secrets.Secrets")
        missing_secrets = []
        if self.storage == "Azure SQL Database":
            missing_secrets = secrets.check_secrets(
                [
                    "SQL_DB_SERVER",
                    "SQL_DB",
                    "SQL_USER",
                    "SQL_PASSWORD",
                    "TABLE_NAME",
                    "TABLE_SCHEMA"
                ]
            )
        if self.storage == "Azure Blob Storage":
            missing_secrets = secrets.check_secrets(
                [
                    "container",
                    "connection_string"
                ]
            )
        if missing_secrets:
            raise Exception(f"Missing secret(s) {', '.join(missing_secrets)} for storage {self.storage}")
        else:
            self.secrets = secrets
            return self

    def set_storage(self, storage_name, secrets: Secrets = None):
        if storage_name is not None:
            if storage_name not in supported_storages:
                raise ValueError(f"Storage {storage_name} is not supported."
                                 f"Supported storages are {', '.join(supported_storages)}")

            if hasattr(self, "storage") and self.storage == storage_name:
                logging.info(f"Storage already set to {storage_name}")
                return
            self.storage = storage_name
        else:
            raise ValueError(f"Storage not specified; provide one of {', '.join(supported_storages)}")
        if secrets is not None:
            self.set_secrets(secrets)
        elif self.secrets is not None:
            self.set_secrets(self.secrets)
        return self

    def save_wordfrequencies(self, frequencies, directory, filename):

        if self.storage == "local":
            # save locally
            os.makedirs(f"{directory}", exist_ok=True)
            frequencies_path = f"./{directory}/{filename}.csv"
            frequencies.to_csv(frequencies_path, index=False, encoding="utf-8")
        else:
            raise ValueError(f"storage {self.storage} is not supported."
                             f"Supported storages are {supported_storages}")

    def get_messages(self,
                     local_path=None,
                     blob_path=None,
                     start_date=None,
                     end_date=None,
                     country=None,
                     source=None
                     ):
        if self.storage is None:
            raise RuntimeError("Storage not specified, use set_storage()")
        df_messages = pd.DataFrame()
        if self.storage == "local":
            # load locally
            if not local_path.endswith('.csv'):
                local_path = os.path.join(local_path, 'messages.csv')
            df_messages = pd.read_csv(local_path)
        elif self.storage == "Azure SQL Database":
            # load from Azure SQL Database
            if start_date is None or end_date is None:
                raise Exception(f"Please specify start and end date to query Azure SQL Database")
            if country is None:
                raise Exception(f"Please specify country to query Azure SQL Database")
            if source is None:
                raise Exception(f"Please specify source to query Azure SQL Database")
            df_messages = self._read_db(start_date, end_date, country, source)
        elif self.storage == "Azure Blob Storage":
            local_directory = local_path[:local_path.rfind("/")]
            os.makedirs(local_directory, exist_ok=True)
            try:
                self._download_blob(local_path, blob_path)
            except Exception as e:
                logging.error(f"Failed downloading from Azure Blob Service: {e}")
            df_messages = pd.read_csv(local_path)

        # Convert dataframe of messages to list of message objects
        messages = []
        for idx, row in df_messages.iterrows():
            # Initiate Message objects with mandatory fields
            try:
                message = Message(
                    row['id_'],
                    row['datetime_'],
                    row['datetime_scraped_'],
                    row['country'],
                    row['source'],
                    row['text'],
                )
            except Exception:
                raise ValueError("Mandatory fields of message missing")
            # Add optional fields
            if row['group']:
                message.group = row['group']
            if row['reply']:
                message.reply = row['reply']
            if row['reply_to']:
                message.reply_to = row['reply_to']
            if row['classifications']:
                message.classifications = ast.literal_eval(row['classifications'])
            if row['translations']:
                message.translations = ast.literal_eval(row['translations'])
            if row['info']:
                message.info = ast.literal_eval(row['info'])
            messages.append(message)
            
        return messages

    def save_messages(self, messages, local_path=None, blob_path=None):
        if self.storage is None:
            raise RuntimeError("Storage not specified, use set_storage()")
        # Read messages to dataframe
        df_messages = _messages_to_df(messages)

        if self.storage == "local":
            # save locally
            if not local_path.endswith('.csv'):
                os.makedirs(local_path, exist_ok=True)
                local_path = os.path.join(local_path, 'messages.csv')
            df_messages.to_csv(local_path, index=False, encoding="utf-8")
            logging.info(f"Successfully saved messages at {local_path}")

        elif self.storage == "Azure Blob Storage":
            # save locally
            local_directory = local_path[:local_path.rfind("/")]
            os.makedirs(local_directory, exist_ok=True)
            df_messages.to_csv(local_path, index=False, encoding="utf-8")
            try:
                self._upload_blob(local_path, blob_path)
            except Exception as e:
                logging.error(f"Failed uploading to Azure Blob Service: {e}")

        elif self.storage == "Azure SQL Database":
            if df_messages['datetime_'].isnull().values.any():
                raise ValueError(f"Please specify datetime_ before saving to Azure SQL Database")
            if df_messages['country'].isnull().values.any():
                raise ValueError("Please specify country before saving to Azure SQL Database")
            if df_messages['source'].isnull().values.any():
                raise ValueError("Please specify source before saving to Azure SQL Database")
            # save to Azure SQL Database
            try:
                self._save_to_db(df_messages)
            except Exception as e:
                logging.error(f"Failed storing in Azure SQL Database: {e}")

    def push_to_argilla(self, messages, tags=None):
        # init argilla
        rg.init(
            api_url=self.secrets.get_secret("ARGILLA_API_URL"),
            api_key=self.secrets.get_secret("ARGILLA_API_KEY"),
            workspace=self.secrets.get_secret("ARGILLA_WORKSPACE")
        )

        topics = [next(iter(classification)) for message in messages for classification in message.classifications]
        topics = set(topics)

        records = []
        for ix, message in enumerate(messages):
            # Set predictions
            if not message.classifications:
                prediction = [(topic, 0.) for topic in topics]
            else:
                predicted_topics = [next(iter(classification)) for classification in message.classifications]
                prediction = [(next(iter(topic)), next(iter(topic.values()))) for topic in message.classifications]
                prediction += [(topic, 0.) for topic in topics if topic not in predicted_topics]
                prediction.sort()

            # Set translations
            inputs = {'Original message': message.text}
            if message.translations:
                for translation in message.translations:
                    inputs[f"Translation ({next(iter(translation))})"] = next(iter(translation.values()))
            inputs['Message number'] = ix+1

            records.append(
                rg.TextClassificationRecord(
                    inputs=inputs,
                    prediction=prediction,
                    prediction_agent='sml-model-0.0.1',
                    multi_label=True,
                    metadata={
                        'channel': message.group,
                        'source': message.source,
                        'to read': np.random.choice(['yes', 'no'], size=1, p=[0.25, 0.75])[0],
                        'number': ix+1
                    },
                    event_timestamp=message.datetime_
                ),
            )

        dataset = rg.DatasetForTextClassification(records)

        settings = rg.TextClassificationSettings(label_schema=topics)
        rg.configure_dataset_settings(name=f"{tags['country']}-{tags['scrape']}", settings=settings)

        # log the dataset
        rg.log(
            dataset,
            name=f"{tags['country']}-{tags['scrape']}",
            workspace=self.secrets.get_secret("argilla_workspace"),
            tags=tags
        )

    def _save_to_db(self, data):
        data_final = self._prepare_messages_for_db(data)
        current_datetime = datetime.now()
        db_table_name = self.secrets.get_secret("TABLE_NAME")
        db_schema = self.secrets.get_secret("TABLE_SCHEMA")
        if not data_final.empty:
            try:
                # Make connection to Azure database
                engine, connection = self._connect_to_db()

                messages_table = db.Table(db_table_name, db.MetaData(schema=db_schema), autoload_with=engine)

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
                    connection.commit()

                logging.info(f"Successfully inserted {len(data_final)} entries into table {db_table_name}, "
                             f"{len(data) - len(data_final)} duplicates already in database")

            except pyodbc.Error as error:
                logging.error("Failed to insert into SQL table {}".format(error))

            finally:
                connection.close()
                engine.dispose()
                logging.info("Pyodbc connection is closed")
        else:
            logging.info(f"All scraped messages already existing in table {db_table_name}")

    def _read_db(self, start_date, end_date, country, source):
        # Connect to db
        db_table_name = self.secrets.get_secret("TABLE_NAME")
        db_schema = self.secrets.get_secret("TABLE_SCHEMA")

        try:
            engine, connection = self._connect_to_db()
            messages_table = db.Table(db_table_name, db.MetaData(schema=db_schema), autoload_with=engine)

            # prepare query
            query = messages_table.select().where(
                db.and_(
                    messages_table.columns.source == source,
                    messages_table.columns.country == country,
                    messages_table.columns.datetime_.between(start_date, end_date)
                )
            )

            output = connection.execute(query)
            results = output.fetchall()
            df_messages = pd.DataFrame(results)
            logging.info(
                f"Successfully retrieved {len(df_messages)} {source} messages"
                f"from {start_date} to {end_date} from table {db_table_name}"
            )
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

        data_in_db = self._read_db(start_date, end_date, country, source)

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

    def _get_blob_service_client(self, blob_path):
        blob_service_client = BlobServiceClient.from_connection_string(self.secrets.get_secret("connection_string"))
        container = self.secrets.get_secret("container")
        return blob_service_client.get_blob_client(container=container, blob=blob_path)

    def _upload_blob(self, local_path, file_dir_blob):
        blob_client = self._get_blob_service_client(file_dir_blob)
        with open(local_path, "rb") as upload_file:
            blob_client.upload_blob(upload_file, overwrite=True)
        logging.info("Successfully uploaded to Azure Blob Storage")

    def _download_blob(self, local_path, blob_path):
        blob_client = self._get_blob_service_client(blob_path)

        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        logging.info("Successfully downloaded from Azure Blob Storage")

    def _connect_to_db(self):
        # Connect to db
        try:
            driver = '{ODBC Driver 18 for SQL Server}'
            params = urllib.parse.quote_plus(
                f'Driver={driver};'
                f'Server=tcp:{self.secrets.get_secret("SQL_DB_SERVER")},1433;'
                f'Database={self.secrets.get_secret("SQL_DB")};'
                f'Uid={self.secrets.get_secret("SQL_USER")};'
                f'Pwd={self.secrets.get_secret("SQL_PASSWORD")};'
                f'Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
            )
            conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
            engine = db.create_engine(conn_str)
            connection = engine.connect()
            logging.info("Successfully connected to database")
            return engine, connection
        except pyodbc.Error as error:
            logging.info("Failed to connect to database {}".format(error))
