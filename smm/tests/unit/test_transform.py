import pytest
import logging
from smm.transform import Transform
from smm.secrets import Secrets
from smm.message import Message
from datetime import datetime
import numbers
import re
import copy

# Get logger
logger = logging.getLogger('__transform__')
logger.setLevel(logging.INFO)

# Initialize transformer
test_secrets = Secrets(path_or_url="../credentials/tests.env", source="env")
test_transformer = Transform(secrets=test_secrets)

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
    assert re.sub('[^A-Za-z0-9]+', '', output.translations[0]['text']) == re.sub('[^A-Za-z0-9]+', '', 'Bonjour le monde!')
    assert 'from_lang' in output.translations[0].keys() and output.translations[0]['from_lang'] == 'en'
    assert 'to_lang' in output.translations[0].keys() and output.translations[0]['to_lang'] == 'fr'


def test_classify_sentiment():
    test_message = copy.deepcopy(template_message)
    test_transformer.set_classifier(name="HuggingFace", task="sentiment-analysis", lang="en")
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
    test_transformer.set_classifier(name="HuggingFace", task="zero-shot-classification", lang="en",
                                    class_labels=['greetings', 'travel'])
    print('test_classify_zeroshot:', test_transformer.class_labels)
    output = test_transformer.classify_message(test_message)
    assert type(output.classifications) == list and len(output.classifications) == 2
    assert 'class' in output.classifications[0].keys() and output.classifications[0]['class'] == 'greetings'
    assert 'score' in output.classifications[0].keys() and isinstance(output.classifications[0]['score'],
                                                                      numbers.Number)
    assert 'class' in output.classifications[1].keys() and output.classifications[1]['class'] == 'travel'
    assert 'score' in output.classifications[1].keys() and isinstance(output.classifications[0]['score'],
                                                                      numbers.Number)