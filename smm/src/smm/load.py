from datetime import datetime, timedelta
import pandas as pd
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

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
        # TBI automatically get secrets from environment variables (?)


class Storage:
    """
    input/output data storage
    """
    def __init__(self, name=None, secrets=None):
        if name is None:
            self.name = "local"
        elif name not in supported_storages:
            raise ValueError(f"storage {name} is not supported."
                             f"Supported storages are {supported_storages}")
        else:
            self.name = name
        self.secrets = secrets

    def set_secrets(self, secrets):
        if not isinstance(secrets, StorageSecrets):
            raise TypeError(f"invalid format of secrets, use extract.SMSecrets")
        self.secrets = secrets

    def check_secrets(self):
        # TBI check that right secrets are filled for data source
        return True


class Load:
    """
    load data from/into a data storage
    """
    def __init__(self, storage=None, secrets=None):
        if storage is not None:
            self.storage = Storage(storage, secrets)
        else:
            self.storage = None

    def save_messages(self, messages, directory, filename):

        # Read messages to dataframe
        df_messages = pd.DataFrame.from_records([msg.to_dict() for msg in messages])

        # Reformat dataframe
        df_messages = self._reformat_messages(df_messages)

        if self.storage.name == "local":
            # save locally
            os.makedirs(f"./{directory}", exist_ok=True)
            messages_path = f"./{directory}/{filename}_latest.csv"
            df_messages.to_csv(messages_path, index=False, encoding="utf-8")

        elif self.storage.name == "Azure SQL Database":
            # save to Azure SQL Database
            if not self.storage.check_secrets():
                raise ValueError("no storage secrets found")

        elif self.storage.name == "Azure Blob Storage":
            # save to Azure Blob Storage
            if not self.storage.check_secrets():
                raise ValueError("no storage secrets found")

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

    def get_blob_service_client(blob_path, config):

    def get_table_service_client(table, config):

    def remove_pii(df, text_columns, config):

    def read_db(sm_code, start_date, end_date, config):

    def save_to_db(sm_code, data, config):

    def connect_to_db(config):

    def download_blob(blob_path, config, local_path=None):

