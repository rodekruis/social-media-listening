import logging
from smm.load import Load
from smm.secrets import Secrets
from smm.message import Message
from datetime import datetime
import copy


# Get logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize transformer
test_secrets = Secrets(path_or_url="../credentials/tests.env", source="env")

# Initialize test messages for storing
template_message = Message(
    id_="0",
    datetime_=datetime(2023, 11, 29, 17, 00, 00),
    datetime_scraped_=datetime.now(),
    country="NLD",
    source="Telegram",
    text="Привіт Світ!",
    group="test",
    reply=False,
    reply_to=None,
    translations=[{"en": "Hello world!"}],
    info={'test': 'random'},
    classifications=[{"label1": 0.80}, {"label2": 0.75}]
)

def test_store_local():
    test_load_local = Load(storage_name="local", secrets=test_secrets)
    test_message = copy.deepcopy(template_message)
    test_load_local.save_messages([test_message], local_path="../data/test.csv")


def test_store_azureblob():
    test_load_azureblob = Load(storage_name="Azure Blob Storage", secrets=test_secrets)
    test_message = copy.deepcopy(template_message)
    test_load_azureblob.save_messages([test_message], local_path="../data/test.csv", blob_path="test/test.csv")


def test_store_azuresql():
    test_load_azuresql = Load(storage_name="Azure SQL Database", secrets=test_secrets)
    test_message = copy.deepcopy(template_message)
    test_load_azuresql.save_messages([test_message])


def test_read_local():
    test_load_local = Load(storage_name="local", secrets=test_secrets)
    messages = test_load_local.get_messages(local_path="../data/test.csv")
    assert type(messages) == list
    assert isinstance(messages[0], Message)
    assert type(messages[0].classifications) == list
    assert type(messages[0].classifications[0]) == dict
    assert messages[0].text == "Привіт Світ!"


def test_read_azureblob():
    test_load_azureblob = Load(storage_name="Azure Blob Storage", secrets=test_secrets)
    messages = test_load_azureblob.get_messages(local_path="../data/test.csv", blob_path="test/test.csv")
    assert type(messages) == list
    assert isinstance(messages[0], Message)
    assert type(messages[0].classifications) == list
    assert type(messages[0].classifications[0]) == dict
    assert messages[0].text == "Привіт Світ!"


def test_read_azuresql():
    test_load_azuresql = Load(storage_name="Azure SQL Database", secrets=test_secrets)
    messages = test_load_azuresql.get_messages(
        start_date=datetime(2023, 11, 29, 15, 00, 00),
        end_date=datetime(2023, 11, 29, 17, 00, 00),
        country='NLD',
        source='Telegram'
    )
    assert type(messages) == list
    assert isinstance(messages[-1], Message)
    assert type(messages[-1].classifications) == list
    assert type(messages[-1].classifications[0]) == dict
    assert messages[-1].text == "Привіт Світ!"


def test_full():
    # initiate local loader
    loader = Load(storage_name="local", secrets=test_secrets)

    # read local messages
    messages = loader.get_messages(local_path="../data/test.csv")

    # set storage to azure blob
    loader.set_storage("Azure Blob Storage")

    # push to azure blob
    loader.save_messages(messages, local_path="../data/prepareBlob.csv", blob_path="test/test.csv")

    # push to argilla
    loader.push_to_argilla(messages, tags={"country": "test", "scrape": "scrape-1"})

    # set storage to azure sql databse
    loader.set_storage("Azure SQL Database")

    # push to azure sql db
    loader.save_messages(messages)

    # set storage to azure blob
    loader.set_storage("Azure Blob Storage")

    # read from azure blob
    messagesBlob = loader.get_messages(local_path="../data/messagesBlob.csv", blob_path="test/test.csv")

    # set storage to local
    loader.set_storage("local")

    # store locally
    loader.save_messages(messagesBlob, local_path="../data/messagesBlob.csv")

    # set storage to azure sql databse
    loader.set_storage("Azure SQL Database")

    # read from azure sql db
    messagesSql = loader.get_messages(
        start_date=datetime(2023, 9, 4, 6, 00, 00),
        end_date=datetime(2023, 9, 18, 23, 59, 00),
        country='Test',
        source='Telegram'
    )

    # set storage to local
    loader.set_storage("local")

    # store locally
    loader.save_messages(messagesSql, local_path="../data/messagesSql.csv")