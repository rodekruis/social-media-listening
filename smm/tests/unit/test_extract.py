import pytest
import logging
from smm.extract import Extract
from smm.secrets import Secrets
from smm.message import Message
from datetime import datetime
import numbers
import re
import copy
from shapely.geometry import Point

# Get logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize transformer
test_secrets = Secrets(path_or_url="../../../credentials/tests.json", source="json")
test_extractor = Extract(source="telegram",
                         secrets=test_secrets)

test_extractor.get_data(channels=["t.me/UAinplovdiv"])


# Initialize test message
template_message = Message(
    id_=0,
    datetime_=datetime.now(),
    datetime_scraped_=datetime.now(),
    country="NLD",
    source="Twitter",
    text="Hello world!")


def test_translate():
    test_message = copy.deepcopy(template_message)
    test_transformer.set_translator(name="HuggingFace", from_lang="en", to_lang="fr")
    output = test_transformer.translate_message(test_message)
    assert type(output.translations) == list and len(output.translations) == 1
    assert 'text' in output.translations[0].keys()
    assert 'from_lang' in output.translations[0].keys() and output.translations[0]['from_lang'] == 'en'
    assert 'to_lang' in output.translations[0].keys() and output.translations[0]['to_lang'] == 'fr'


def test_classify_sentiment():
    test_message = copy.deepcopy(template_message)
    test_transformer.set_classifier(name="HuggingFace", task="sentiment-analysis")
    output = test_transformer.classify_message(test_message)
    assert type(output.classifications) == list and len(output.classifications) == 2
    assert 'class' in output.classifications[0].keys() and output.classifications[0]['class'] == 'POSITIVE'
    assert 'score' in output.classifications[0].keys() and isinstance(output.classifications[0]['score'],
                                                                      numbers.Number)
    assert 'class' in output.classifications[1].keys() and output.classifications[1]['class'] == 'NEGATIVE'
    assert 'score' in output.classifications[1].keys() and isinstance(output.classifications[0]['score'],
                                                                      numbers.Number)


def test_classify_zeroshot():
    test_message = copy.deepcopy(template_message)
    test_transformer.set_classifier(name="HuggingFace", task="zero-shot-classification",
                                    class_labels=['greetings', 'travel'])
    output = test_transformer.classify_message(test_message)
    assert type(output.classifications) == list and len(output.classifications) == 2
    assert 'class' in output.classifications[0].keys() and output.classifications[0]['class'] == 'greetings'
    assert 'score' in output.classifications[0].keys() and isinstance(output.classifications[0]['score'],
                                                                      numbers.Number)
    assert 'class' in output.classifications[1].keys() and output.classifications[1]['class'] == 'travel'
    assert 'score' in output.classifications[1].keys() and isinstance(output.classifications[0]['score'],
                                                                      numbers.Number)


def test_translate_and_classify():
    test_message = copy.deepcopy(template_message)
    test_transformer.set_translator(name="HuggingFace", from_lang="en", to_lang="fr")
    test_transformer.set_classifier(name="HuggingFace", task="sentiment-analysis", lang="fr")
    output = test_transformer.translate_message(test_message)
    output = test_transformer.classify_message(output)
    assert type(output.classifications) == list and len(output.classifications) == 2
    assert 'class' in output.classifications[0].keys() and output.classifications[0]['class'] == 'POSITIVE'
    assert 'score' in output.classifications[0].keys() and isinstance(output.classifications[0]['score'],
                                                                      numbers.Number)
    assert 'class' in output.classifications[1].keys() and output.classifications[1]['class'] == 'NEGATIVE'
    assert 'score' in output.classifications[1].keys() and isinstance(output.classifications[0]['score'],
                                                                      numbers.Number)


def test_anonymize():
    test_message = copy.deepcopy(template_message)
    test_message.text = "Hello my name is Jean."
    test_transformer.set_anonymizer(name="anonymization-app")
    output = test_transformer.anonymize_message(test_message)
    assert "<PERSON>" in output.text


def test_translate_and_anonymize():
    test_message = copy.deepcopy(template_message)
    test_message.text = "Hello my name is Jean."
    test_transformer.set_translator(name="HuggingFace", from_lang="en", to_lang="fr")
    test_transformer.set_anonymizer(name="anonymization-app", lang="fr")
    output = test_transformer.translate_message(test_message)
    output = test_transformer.anonymize_message(output)
    assert "<PERSON>" in output.translations[0]['text']


def test_geolocate():
    test_message = copy.deepcopy(template_message)
    test_message.text = "Gigi is from Piemonte and Anna is from Veneto"
    test_transformer.set_geolocator(locations_file="tests/data/locations.geojson", locations_fields=["reg_name"])
    output = test_transformer.geolocate_message(test_message)
    assert type(output.info['locations']) == list and len(output.info['locations']) == 2
    assert 'name' in output.info['locations'][0].keys() and output.info['locations'][0]['name'] == 'piemonte'
    assert 'name' in output.info['locations'][1].keys() and output.info['locations'][1]['name'] == 'veneto'

    test_message.info['coordinates'] = Point(14.305573, 40.853294)
    output = test_transformer.geolocate_message(test_message)
    assert type(output.info['locations']) == list and len(output.info['locations']) == 3
    assert 'name' in output.info['locations'][2].keys() and output.info['locations'][2]['name'] == 'campania'