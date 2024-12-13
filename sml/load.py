from datetime import datetime, timedelta
import pandas as pd
import os
from typing import List
from azure.storage.blob import BlobServiceClient
import logging
import pyodbc
import sqlalchemy as db
import urllib
import ast
import json
import argilla as rg
import azure.cosmos.cosmos_client as cosmos_client
from azure.cosmos.exceptions import CosmosResourceExistsError
from datasets import Dataset
import re
from sml.secrets import Secrets
from sml.message import Message
from typing import List
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

supported_storages = [
    "local",
    "Azure SQL Database",
    "Azure Blob Storage",
    "Azure Cosmos DB",
]


class Load:
    """Download/upload data from/to a data storage"""

    def __init__(self, secrets: Secrets = None):
        self.storage = "local"
        self.secrets = None
        if secrets is not None:
            self.set_secrets(secrets)

    def set_secrets(self, secrets):
        """Set secrets for storage"""
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
                    "TABLE_SCHEMA",
                ]
            )
        if self.storage == "Azure Blob Storage":
            missing_secrets = secrets.check_secrets(["container", "connection_string"])
        if self.storage == "Azure Cosmos DB":
            missing_secrets = secrets.check_secrets(
                ["COSMOS_URL", "COSMOS_KEY", "COSMOS_DATABASE", "COSMOS_CONTAINER"]
            )
        if missing_secrets:
            raise Exception(
                f"Missing secret(s) {', '.join(missing_secrets)} for storage {self.storage}"
            )
        else:
            self.secrets = secrets
            return self

    def set_storage(self, storage_name, secrets: Secrets = None):
        """Set storage to save/load data"""
        if storage_name is not None:
            if storage_name not in supported_storages:
                raise ValueError(
                    f"Storage {storage_name} is not supported."
                    f"Supported storages are {', '.join(supported_storages)}"
                )

            if hasattr(self, "storage") and self.storage == storage_name:
                logging.info(f"Storage already set to {storage_name}")
                return
            self.storage = storage_name
        else:
            raise ValueError(
                f"Storage not specified; provide one of {', '.join(supported_storages)}"
            )
        if secrets is not None:
            self.set_secrets(secrets)
        elif self.secrets is not None:
            self.set_secrets(self.secrets)
        return self

    def get_messages(
        self,
        local_path=None,
        blob_path=None,
        start_date=None,
        end_date=None,
        country=None,
        source=None,
    ):
        """Download messages from storage"""
        if self.storage is None:
            raise RuntimeError("Storage not specified, use set_storage()")
        messages = []

        if self.storage == "local":
            # load locally
            if local_path is None:
                raise RuntimeError(
                    "Local path not specified, provide path to local .csv file"
                )
            if not local_path.endswith(".csv"):
                local_path = os.path.join(local_path, "messages.csv")
            df_messages = pd.read_csv(local_path)
            messages = self._df_to_messages(df_messages)

        elif self.storage == "Azure SQL Database":
            # load from Azure SQL Database
            if start_date is None or end_date is None:
                raise Exception(
                    f"Please specify start and end date to query Azure SQL Database"
                )
            if country is None:
                raise Exception(f"Please specify country to query Azure SQL Database")
            if source is None:
                raise Exception(f"Please specify source to query Azure SQL Database")
            df_messages = self._get_from_sql(start_date, end_date, country, source)
            messages = self._df_to_messages(df_messages)

        elif self.storage == "Azure Blob Storage":
            local_directory = local_path[: local_path.rfind("/")]
            os.makedirs(local_directory, exist_ok=True)
            try:
                self._get_from_blob(local_path, blob_path)
            except Exception as e:
                logging.error(f"Failed downloading from Azure Blob Service: {e}")
            df_messages = pd.read_csv(local_path)
            messages = self._df_to_messages(df_messages)

        elif self.storage == "Azure Cosmos DB":
            try:
                messages = self._get_from_cosmos(start_date, end_date, country, source)
            except Exception as e:
                logging.error(f"Failed downloading from Azure Cosmos DB: {e}")
        return messages

    def save_messages(self, messages, local_path=None, blob_path=None):
        """Upload messages to storage"""
        if self.storage is None:
            raise RuntimeError("Storage not specified, use set_storage()")

        if self.storage == "local":
            df_messages = self._messages_to_df(messages)
            if not local_path.endswith(".csv"):
                os.makedirs(local_path, exist_ok=True)
                local_path = os.path.join(local_path, "messages.csv")
            df_messages.to_csv(local_path, index=False, encoding="utf-8")
            logging.info(f"Successfully saved messages at {local_path}")

        elif self.storage == "Azure Blob Storage":
            try:
                self._save_to_blob(messages, local_path, blob_path)
            except Exception as e:
                logging.error(f"Failed uploading to Azure Blob Service: {e}")

        elif self.storage == "Azure Cosmos DB":
            try:
                self._save_to_cosmos(messages)
            except Exception as e:
                logging.error(f"Failed uploading to Azure Cosmos DB: {e}")

        elif self.storage == "Azure SQL Database":
            df_messages = self._messages_to_df(messages)
            if df_messages["datetime_"].isnull().values.any():
                raise ValueError(
                    f"Please specify datetime_ before saving to Azure SQL Database"
                )
            if df_messages["country"].isnull().values.any():
                raise ValueError(
                    "Please specify country before saving to Azure SQL Database"
                )
            if df_messages["source"].isnull().values.any():
                raise ValueError(
                    "Please specify source before saving to Azure SQL Database"
                )
            # save to Azure SQL Database
            try:
                self._save_to_sql(df_messages)
            except Exception as e:
                logging.error(f"Failed uploading to Azure SQL Database: {e}")

    def save_to_argilla(
        self,
        messages: List[Message],
        dataset_name: str,
        workspace: str,
        label_schema: List[str] = None,
        replace: bool = True,
    ):
        """Save messages to Argilla"""

        workspace = workspace.lower()
        if workspace not in [ws.name for ws in rg.list_workspaces()]:
            rg.Workspace.create(workspace)

        # init argilla
        rg.init(
            api_url=self.secrets.get_secret("ARGILLA_API_URL"),
            api_key=self.secrets.get_secret("ARGILLA_API_KEY"),
            workspace=workspace,
        )

        rc_keywords = [
            "червоний хрест",
            "червоного хреста",
            "червоному хресту",
            "червоним хрестом",
            "червоному хресті",
            "бчх",
            "красный крест",
            "красного крестa",
            "красному кресту",
            "красного креста",
            "красным крестом",
            "красном кресте",
            "крест",
            "крестa",
            "кресту",
            "крестом",
            "кресте",
            "Хреста",
            "Хрест",
            "хрестом",
            "хресту",
            "хресті",
            "КК",
            "ЧХ",
            "МКЧХ",
            "МФОКК",
            "МККК",
            "МФОККиКП",
            "МФЧХ",
            "IFRC",
        ]

        # Remove duplicate messages
        unique_messages = []
        for message in messages:
            is_duplicate = False
            for unique_message in unique_messages:
                if message.text == unique_message.text:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_messages.append(message)
        messages = unique_messages.copy()

        # get topics
        topics = [
            classification["topic"]
            for message in messages
            for classification in message.classifications
        ]
        topics = set(topics)
        if not topics:
            if not label_schema:
                raise ValueError(
                    "No classifications found in messages, label_schema must be specified"
                )
            else:
                topics = label_schema

        dataset = rg.FeedbackDataset(
            fields=[
                rg.TextField(
                    name="text_original",
                    title="Original message",
                    required=True,
                    use_markdown=True,
                ),
                rg.TextField(
                    name="text_translated",
                    title="Translated message",
                    required=True,
                    use_markdown=True,
                ),
            ],
            questions=[
                rg.MultiLabelQuestion(
                    name="topic",
                    title="What is the topic of the message?",
                    labels=topics,
                    required=True,
                    visible_labels=20,
                ),
                rg.TextQuestion(
                    name="anonymization",
                    title="Re-write the message if it contains personally identifiable information",
                    description="The original message will be replaced with the new one; example: 'My name is John' -> 'My name is [name]'",
                    use_markdown=True,
                    required=False,
                ),
                rg.TextQuestion(
                    name="translation",
                    title="Re-translate the message if incorrect",
                    use_markdown=True,
                    required=False,
                ),
            ],
            guidelines=None,
        )

        records = []
        for ix, message in enumerate(messages):
            # Check if there is a message text
            if not message.text:
                continue

            # set suggestions
            classifications = []
            # suggest only highest-scoring model classification
            message_classifications = [
                x for x in message.classifications if x["agent"] == "model"
            ]
            if len(message_classifications) > 0:
                message_classification = sorted(
                    message_classifications, key=lambda x: x["score"], reverse=True
                )[0]
                classifications.append(
                    {
                        "question_name": "topic",
                        "value": [message_classification["topic"]],
                        "score": [message_classification["score"]],
                        "agent": "model",
                    }
                )

            # set translations
            text_translated, from_lang, to_lang = "", "", ""
            if message.translations:
                text_translated = message.translations[0]["text"]
                from_lang = message.translations[0]["from_lang"]
                to_lang = message.translations[0]["to_lang"]

            # check if message is about the red cross
            red_cross = "No"
            for word in rc_keywords:
                if len(word.split()) > 1:
                    if word.lower() in message.text.lower():
                        red_cross = "Yes"
                        break
                else:
                    if re.search(
                        r"\b" + re.escape(word.lower()) + r"\b", message.text.lower()
                    ):
                        red_cross = "Yes"
                        break

            record = rg.FeedbackRecord(
                fields={
                    "text_translated": text_translated,
                    "text_original": message.text,
                },
                responses=[],
                suggestions=classifications,
                external_id=message.id_,
                metadata={
                    "group": message.group,
                    "group_members": (
                        message.info["group_members"]
                        if "group_members" in message.info
                        else None
                    ),
                    "source": message.source,
                    "number": ix + 1,
                    "Red Cross": red_cross,
                    "translation_from_lang": from_lang,
                    "translation_to_lang": to_lang,
                    "datetime": message.datetime_.strftime("%Y-%m-%dT%H:%M:%S"),
                    "datetime_scraped": message.datetime_scraped_.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                    "country": message.country,
                },
            )
            records.append(record)

        dataset.add_records(records)
        if replace:
            try:
                old_dataset = rg.FeedbackDataset.from_argilla(
                    name=dataset_name,
                    workspace=workspace,
                )
                old_dataset.delete()
            except ValueError:
                pass
        dataset.push_to_argilla(name=dataset_name, workspace=workspace)

    def get_from_argilla(
        self, dataset_name: str, workspace: str, only_submitted: bool = False
    ):
        """Get messages from Argilla"""
        # init argilla
        rg.init(
            api_url=self.secrets.get_secret("ARGILLA_API_URL"),
            api_key=self.secrets.get_secret("ARGILLA_API_KEY"),
            workspace=workspace,
        )
        dataset = (
            rg.FeedbackDataset.from_argilla(dataset_name)
            .pull(max_records=1000000)
            .format_as("datasets")
        )
        messages = []
        for record in dataset:
            if only_submitted:
                if not record["topic"] or record["topic"][0]["status"] == "discarded":
                    continue
            else:
                metadata = json.loads(record["metadata"])

                # get translation if available
                translations = []
                if record["text_translated"] != "":
                    translation = {
                        "text": record["text_translated"],
                        "from_lang": metadata["translation_from_lang"],
                        "to_lang": metadata["translation_to_lang"],
                    }
                    if record["translation"]:
                        translation["text"] = record["translation"][0]["value"]
                    translations.append(translation)

                # get topic if available
                classifications = []
                if record["topic-suggestion"]:
                    classifications = [
                        {
                            "class": record["topic-suggestion"][0],
                            "score": record["topic-suggestion-metadata"]["score"][0],
                            "agent": "model",
                        }
                    ]
                if record["topic"] and record["topic"][0]["status"] == "submitted":
                    classifications.append(
                        {
                            "class": record["topic"][0]["value"],
                            "score": 1.0,
                            "agent": "human",
                        }
                    )

                # map record to message object
                message = Message(
                    id_=record["external_id"],
                    datetime_=metadata["datetime"],
                    datetime_scraped_=metadata["datetime_scraped"],
                    country=metadata["country"],
                    source=metadata["source"],
                    text=record["text_original"],
                    group=metadata["group"],
                    reply=False,
                    reply_to=None,
                    translations=translations,
                    info={"group_members": metadata["group_members"]},
                    classifications=classifications,
                )
                messages.append(message)
        return messages

    def _save_to_cosmos(self, messages: List[Message]):
        """Save messages to Cosmos DB"""
        client_ = cosmos_client.CosmosClient(
            self.secrets.get_secret("COSMOS_URL"),
            {"masterKey": self.secrets.get_secret("COSMOS_KEY")},
            user_agent="sml-api",
            user_agent_overwrite=True,
        )
        cosmos_db = client_.get_database_client(
            self.secrets.get_secret("COSMOS_DATABASE")
        )
        cosmos_container_client = cosmos_db.get_container_client(
            self.secrets.get_secret("COSMOS_CONTAINER")
        )
        for message in messages:
            record = message.to_dict()
            record["id"] = str(record.pop("id_"))
            cosmos_container_client.upsert_item(body=record)

    def _get_blob_service_client(self, blob_path: str):
        blob_service_client = BlobServiceClient.from_connection_string(
            self.secrets.get_secret("connection_string")
        )
        container = self.secrets.get_secret("container")
        return blob_service_client.get_blob_client(container=container, blob=blob_path)

    def _save_to_blob(
        self, messages: List[Message], local_path: str, file_dir_blob: str
    ):
        """Save messages to Azure Blob Storage"""
        # save locally
        df_messages = self._messages_to_df(messages)
        local_directory = local_path[: local_path.rfind("/")]
        os.makedirs(local_directory, exist_ok=True)
        df_messages.to_csv(local_path, index=False, encoding="utf-8")
        # upload to Azure Blob Storage
        blob_client = self._get_blob_service_client(file_dir_blob)
        with open(local_path, "rb") as upload_file:
            blob_client.upload_blob(upload_file, overwrite=True)
        if os.path.exists(local_path):
            os.remove(local_path)
        logging.info("Successfully uploaded to Azure Blob Storage")

    def _get_from_blob(self, local_path: str, blob_path: str):
        """Get messages from Azure Blob Storage"""
        blob_client = self._get_blob_service_client(blob_path)

        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        logging.info("Successfully downloaded from Azure Blob Storage")

    def _save_to_sql(self, data: pd.DataFrame):
        """Save messages to Azure SQL Database"""
        data_final = self._prepare_messages_for_db(data)
        current_datetime = datetime.now()
        db_table_name = self.secrets.get_secret("TABLE_NAME")
        db_schema = self.secrets.get_secret("TABLE_SCHEMA")
        if not data_final.empty:
            try:
                # Make connection to Azure database
                engine, connection = self._connect_to_db()

                messages_table = db.Table(
                    db_table_name, db.MetaData(schema=db_schema), autoload_with=engine
                )

                for idx, row in data_final.iterrows():
                    connection.execute(
                        db.insert(messages_table),
                        [
                            {
                                "id_": row["id_"],
                                "datetime_": row["datetime_"],
                                "datetime_scraped_": current_datetime.strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "country": row["country"],
                                "source": row["source"],
                                "text": row["text"],
                                "group": row["group"],
                                "reply": row["reply"],
                                "reply_to": row["reply_to"],
                                "translations": row["translations"],
                                "info": row["info"],
                                "classifications": row["classifications"],
                            }
                        ],
                    )
                    connection.commit()

                logging.info(
                    f"Successfully inserted {len(data_final)} entries into table {db_table_name}, "
                    f"{len(data) - len(data_final)} duplicates already in database"
                )

            except pyodbc.Error as error:
                logging.error("Failed to insert into SQL table {}".format(error))

            finally:
                connection.close()
                engine.dispose()
                logging.info("Pyodbc connection is closed")
        else:
            logging.info(
                f"All scraped messages already existing in table {db_table_name}"
            )

    def _get_from_sql(self, start_date, end_date, country, source):
        """Get messages from Azure SQL Database"""
        # Connect to db
        db_table_name = self.secrets.get_secret("TABLE_NAME")
        db_schema = self.secrets.get_secret("TABLE_SCHEMA")

        try:
            engine, connection = self._connect_to_db()
            messages_table = db.Table(
                db_table_name, db.MetaData(schema=db_schema), autoload_with=engine
            )

            # prepare query
            query = messages_table.select().where(
                db.and_(
                    messages_table.columns.source == source,
                    messages_table.columns.country == country,
                    messages_table.columns.datetime_.between(start_date, end_date),
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

    def _get_from_cosmos(self, start_date, end_date, country, source):
        """Get messages from Cosmos DB"""
        client_ = cosmos_client.CosmosClient(
            self.secrets.get_secret("COSMOS_URL"),
            {"masterKey": self.secrets.get_secret("COSMOS_KEY")},
            user_agent="sml-api",
            user_agent_overwrite=True,
        )
        cosmos_db = client_.get_database_client(
            self.secrets.get_secret("COSMOS_DATABASE")
        )
        cosmos_container_client = cosmos_db.get_container_client(
            self.secrets.get_secret("COSMOS_CONTAINER")
        )
        query = "SELECT * FROM c WHERE "
        if start_date is not None:
            query += f'c.datetime_ >= "{start_date.strftime("%Y-%m-%dT%H:%M:%S")}" '
        if end_date is not None:
            query += f'AND c.datetime_ <= "{end_date.strftime("%Y-%m-%dT%H:%M:%S")}" '
        if country is not None:
            query += f'AND c.country = "{country}" '
        if source is not None:
            query += f'AND c.source = "{source}" '
        if query.endswith("WHERE "):
            query = query.replace("WHERE ", "")
        query = query.replace("WHERE AND", "WHERE")
        print(f"QUERY: {query}")
        records = cosmos_container_client.query_items(
            query=query,
            enable_cross_partition_query=(
                True if country is None else None
            ),  # country must be the partition key
        )
        messages = []
        for record in records:
            message = Message(
                record["id"],
                record["datetime_"],
                record["datetime_scraped_"],
                record["country"],
                record["source"],
                record["text"],
                record["group"] if "group" in record else None,
                record["reply"] if "reply" in record else None,
                record["reply_to"] if "reply_to" in record else None,
                record["translations"] if "translations" in record else None,
                record["info"] if "info" in record else None,
                record["classifications"] if "classifications" in record else None,
            )
            messages.append(message)
        return messages

    def _prepare_messages_for_db(self, data):
        # Prepare for storing in Azure db
        data["datetime_"] = pd.to_datetime(
            data["datetime_"], format="%Y-%m-%d %H:%M:%S"
        )
        data["datetime_"] = data["datetime_"].dt.tz_localize(None)
        data["id_"] = data["id_"].astype("int64")
        data = data.astype(object).where(pd.notnull(data), None)

        # Read data from database to check for duplicates
        logging.info("Check for duplicates in database")
        data["in_db"] = False

        start_date = min(data["datetime_"])
        end_date = max(data["datetime_"])
        country = data["country"].unique()[0]
        source = data["source"].unique()[0]

        data_in_db = self._get_from_sql(start_date, end_date, country, source)

        if data_in_db is not None and not data_in_db.empty:
            # Make sure format of new data and data in database are the same
            data_in_db.drop(columns=["ID"], inplace=True)

            data_in_db["datetime_"] = pd.to_datetime(
                data_in_db["datetime_"], format="%Y-%m-%d %H:%M:%S"
            )
            data_in_db["id_"] = data_in_db["id_"].astype("int64")

            data_in_db = data_in_db.astype(object).where(pd.notnull(data_in_db), None)

            data_in_db["in_db"] = True

            # Concatenate new data and data already in database and drop duplicates
            df_concat = pd.concat([data, data_in_db], ignore_index=True)
            df_concat.drop_duplicates(
                subset=["id_", "source", "datetime_", "text"], keep=False, inplace=True
            )

            # Remove all rows there were read from database
            df_concat = df_concat[~df_concat["in_db"]]
            data_final = df_concat.drop(columns=["in_db"])
        else:
            data_final = data.drop(columns=["in_db"])

        return data_final

    def _connect_to_db(self):
        # Connect to db
        try:
            driver = "{ODBC Driver 18 for SQL Server}"
            params = urllib.parse.quote_plus(
                f"Driver={driver};"
                f'Server=tcp:{self.secrets.get_secret("SQL_DB_SERVER")},1433;'
                f'Database={self.secrets.get_secret("SQL_DB")};'
                f'Uid={self.secrets.get_secret("SQL_USER")};'
                f'Pwd={self.secrets.get_secret("SQL_PASSWORD")};'
                f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
            )
            conn_str = "mssql+pyodbc:///?odbc_connect={}".format(params)
            engine = db.create_engine(conn_str)
            connection = engine.connect()
            logging.info("Successfully connected to database")
            return engine, connection
        except pyodbc.Error as error:
            logging.info("Failed to connect to database {}".format(error))

    def _messages_to_df(self, messages: List[Message]):
        """Convert a list of Message objects to a pandas DataFrame"""
        df_messages = pd.DataFrame.from_records([msg.to_dict() for msg in messages])
        df_messages["info"] = df_messages["info"].apply(lambda x: str(x))
        df_messages["translations"] = df_messages["translations"].apply(
            lambda x: str(x)
        )
        df_messages["classifications"] = df_messages["classifications"].apply(
            lambda x: str(x)
        )
        df_messages["datetime_"] = df_messages["datetime_"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )
        return df_messages

    def _df_to_messages(self, df_messages: pd.DataFrame):
        """Convert dataframe to list of Message objects"""
        messages = []
        for idx, row in df_messages.iterrows():
            # Initiate Message objects with mandatory fields
            try:
                message = Message(
                    row["id_"],
                    row["datetime_"],
                    row["datetime_scraped_"],
                    row["country"],
                    row["source"],
                    row["text"],
                )
            except Exception:
                raise ValueError("Mandatory fields of message missing")
            # Add optional fields
            if row["group"]:
                message.group = row["group"]
            if row["reply"]:
                message.reply = row["reply"]
            if row["reply_to"]:
                message.reply_to = row["reply_to"]
            if row["classifications"]:
                message.classifications = ast.literal_eval(row["classifications"])
            if row["translations"]:
                message.translations = ast.literal_eval(row["translations"])
            if row["info"]:
                message.info = ast.literal_eval(row["info"])
            messages.append(message)
        return messages
