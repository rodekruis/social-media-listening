from datetime import datetime, timedelta
supported_storages = ["local", "Azure SQL Database", "Azure Blob Storage"]


class StorageSecrets:
    """
    Secrets (API keys, tokens, etc.) for input/output data storage
    """
    def __init__(self,
                 keyvault_url="",
                 database_secret=""):
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

    def save_messages(self,
                  messages):

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
