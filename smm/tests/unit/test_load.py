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
test_load_local = Load(storage_name="local", secrets=test_secrets)
test_load_azureblob = Load(storage_name="Azure Blob Storage", secrets=test_secrets)
test_load_azuresql = Load(storage_name="Azure SQL Database", secrets=test_secrets)

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
    test_message = copy.deepcopy(template_message)
    test_load_local.save_messages([test_message], "../../../test", "test.csv")
    # assert


def test_store_azureblob():
    test_message = copy.deepcopy(template_message)
    test_load_azureblob.save_messages([test_message], "test", "test.csv")
    # assert


def test_store_azuresql():
    test_message = copy.deepcopy(template_message)
    test_load_azuresql.save_messages([test_message])
    # assert


def test_read_local():
    messages = test_load_local.get_messages(directory="../../../test", filename="test.csv")
    assert type(messages) == list
    assert isinstance(messages[0], Message)
    assert type(messages[0].classifications) == list
    assert type(messages[0].classifications[0]) == dict
    assert messages[0].text == "Привіт Світ!"


def test_read_azureblob():
    messages = test_load_azureblob.get_messages(directory="test", filename="test.csv")
    assert type(messages) == list
    assert isinstance(messages[0], Message)
    assert type(messages[0].classifications) == list
    assert type(messages[0].classifications[0]) == dict
    assert messages[0].text == "Привіт Світ!"


def test_read_azuresql():
    messages = test_load_azuresql.get_messages(
        start_date=datetime(2023,11,29,15,00,00),
        end_date=datetime(2023,11,29,17,00,00),
        country='NLD',
        source='Telegram'
    )
    assert type(messages) == list
    assert isinstance(messages[-1], Message)
    assert type(messages[-1].classifications) == list
    assert type(messages[-1].classifications[0]) == dict
    assert messages[-1].text == "Привіт Світ!"