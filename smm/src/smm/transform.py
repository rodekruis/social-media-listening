from smm.message import Message
from transformers import pipeline
from google.cloud import translate_v2 as google_translate
from google.oauth2 import service_account as google_service_account
from time import sleep
import logging
import json
import requests
import uuid
import os
supported_translators = ["HuggingFace", "Google", "Microsoft"]
supported_classifiers = ["HuggingFace", "URL"]
supported_anonymizers = ["anonymization-app"]
supported_classifier_tasks = ["sentiment-analysis", "zero-shot-classification"]


class Transform:
    """
    transform data
    """

    def __init__(self):
        self.translator = self.set_translator()
        self.classifier = self.set_classifier()
        self.wordfreqer = self.set_wordfreqer()
        self.anonymizer = self.set_anonymizer()
        self.geolocator = self.set_geolocator()
        # translator fields
        self.translator_name = None
        self.from_lang = None
        self.to_lang = None
        self.translator = None
        # classifier fields
        self.classifier_name = None
        self.classifier_task = None
        self.class_labels = None
        self.classifier = None
        # anonymizer fields
        self.anonymizer_name = None
        self.anonymizer = None


    def set_translator(self, from_lang, to_lang, name=None, secrets=None):
        self.from_lang = from_lang
        self.to_lang = to_lang
        if self.from_lang is None or self.to_lang is None:
            raise ValueError(f"original and target language must be specified for translator.")

        if name is None:
            self.translator_name = "HuggingFace"
        else:
            self.translator_name = name

        if self.translator_name == "HuggingFace":
            self.translator = pipeline(f"translation_{from_lang}_to_{to_lang}", model="t5-base")

        elif self.translator_name == "Microsoft":
            subcription_info = {
                "subscription_key": secrets["MSCOGNITIVE_KEY"],
                "location": secrets["MSCOGNITIVE_LOCATION"]
            }
            constructed_url = secrets['MSCOGNITIVE_URL']
            params = {
                'api-version': '3.0',
                'from': [from_lang],
                'to': [to_lang],
            }
            headers = {
                'Ocp-Apim-Subscription-Key': subcription_info["subscription_key"],
                'Ocp-Apim-Subscription-Region': subcription_info["location"],
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
            }
            self.translator = [constructed_url, params, headers]

        elif self.translator_name == "Google":
            service_account_info = secrets["GOOGLE_SERVICEACCOUNT"]
            credentials = google_service_account.Credentials.from_service_account_info(json.loads(service_account_info))
            self.translator = google_translate.Client(credentials=credentials)

        else:
            raise ValueError(f"translator {name} is not supported."
                             f"Supported translators are {supported_translators}")

    def translate(self, text):
        translation_data = {'text': ""}

        if self.translator_name == "HuggingFace":
            translation = self.translator(text)
            translation_data = {
                "text": translation[0]['translation_text'],
                "from_lang": self.from_lang,
                "to_lang": self.to_lang,
            }

        elif self.translator_name == "Microsoft":
            constructed_url = self.translator[0]
            params = self.translator[1]
            headers = self.translator[2]
            for retry in range(10):
                try:
                    request = requests.post(constructed_url, params=params, headers=headers, json=[{'text': text}])
                    response = request.json()
                    translation_data = {
                        "text": response[0]['translations'][0]['text'],
                        "from_lang": self.from_lang,
                        "to_lang": self.to_lang,
                    }
                except Exception as e:
                    logging.error(e)
                    sleep(10)

        elif self.translator_name == "Google":
            for retry in range(10):
                try:
                    response = self.translator.translate(text, target_language="en")
                    translation_data = {
                        "text": response["translatedText"],
                        "from_lang": self.from_lang,
                        "to_lang": self.to_lang,
                    }
                except Exception as e:
                    logging.error(e)
                    sleep(10)

        if translation_data['text'] == text:
            logging.warning("translator returned identical message, check configuration.")
        return translation_data

    def set_classifier(self, name=None, task=None, class_labels=None, secrets=None):
        self.class_labels = class_labels
        if name is None:
            self.classifier_name = "HuggingFace"
        else:
            self.classifier_name = name
        if task is None:
            self.classifier_task = "sentiment-analysis"
        else:
            self.classifier_task = task

        if task == "sentiment-analysis":
            self.class_labels = ["POSITIVE", "NEGATIVE"]
            if class_labels is not None:
                logging.warning(f"classifier task sentiment-analysis uses fixed class labels,"
                                f" overwriting {class_labels} with {self.class_labels}.")
        else:
            if class_labels is None:
                raise ValueError(f"class labels must be specified for classifier task {task}")
            self.class_labels = class_labels

        if self.classifier_name == "HuggingFace":
            if self.classifier_task == "sentiment-analysis":
                self.classifier = pipeline(f"sentiment-analysis",
                                           model="distilbert-base-uncased-finetuned-sst-2-english")
            elif self.classifier_task == "zero-shot-classification":
                self.classifier = pipeline(f"zero-shot-classification",
                                           model="facebook/bart-large-mnli")
            else:
                raise ValueError(f"classifier task {task} is not supported."
                                 f"Supported classifier tasks are {supported_classifier_tasks}")

        elif self.translator_name == "URL":
            if self.classifier_task in supported_classifier_tasks:
                self.classifier = secrets["CLASSIFIER_URL"]
            else:
                raise ValueError(f"classifier task {task} is not supported."
                                 f"Supported classifier tasks are {supported_classifier_tasks}")

        else:
            raise ValueError(f"classifier {name} is not supported."
                             f"Supported classifiers are {supported_translators}")

    def classify(self, text):
        classification_data = []

        if self.classifier_name == "HuggingFace":
            if self.classifier_task == "sentiment-analysis":
                result = self.classifier(text)[0]
                classification_data = [
                    {'class': result['label'], 'score': result['score']},
                    {'class': 'POSITIVE' if result['label']=='NEGATIVE' else 'NEGATIVE', 'score': 1.-result['score']}
                ]
            if self.classifier_task == "zero-shot-classification":
                result = self.classifier(sequences=text, candidate_labels=self.class_labels, multi_label=True)
                for label, score in zip(result['labels'], result['scores']):
                    classification_data.append({'class': label, 'score': score})

        elif self.translator_name == "URL":
            payload = {'text': text}
            if self.classifier_task != "sentiment-analysis":
                payload['labels'] = self.class_labels
                payload['multi_label'] = True
            for retry in range(10):
                try:
                    result = requests.post(self.classifier, json=payload).json()
                    for label, score in zip(result['labels'], result['scores']):
                        classification_data.append({'class': label, 'score': score})
                except Exception as e:
                    logging.error(e)
                    sleep(10)

        if not classification_data:
            logging.warning("classifier returned no results, check configuration.")
        return classification_data

    def set_wordfreqer(self):
        # TBI

    def set_anonymizer(self, name=None):
        if name is None:
            self.anonymizer_name = "anonymization-app"
            self.anonymizer = "https://anonymization-app.azurewebsites.net/anonymize/"
        else:
            self.anonymizer_name = name

        if self.anonymizer_name != "anonymization-app":
            raise ValueError(f"anonymizer {name} is not supported."
                             f"Supported anonymizers are {supported_anonymizers}")

    def anonymize(self, text):
        anonymized_text = text
        for retry in range(10):
            try:
                request = requests.post(self.anonymizer, json={"text": text, "model": "ensemble"})
                response = request.json()
                if 'anonymized_text' in response.keys():
                    anonymized_text = response['anonymized_text']
            except Exception as e:
                logging.error(e)
                sleep(10)

        return anonymized_text

    def set_geolocator(self):
        # TBI

    def process_data(self,
                     messages,
                     translate=False,
                     classify=False,
                     wordfreq=False,
                     anonymize=False,
                     geolocate=False):
        if translate:
            logging.info('translating messages')
            for idx, message in enumerate(messages):
                translation = self.translate(message.text)
                messages[idx].set_translation(translation)
        if classify:
            logging.info('classifying messages')
            for idx, message in enumerate(messages):
                classification = self.classify(message.text)
                messages[idx].set_classification(classification)
        if wordfreq:
            # TBI wordfreq messages
            messages = []
        if geolocate:
            # TBI geolocate messages
            messages = []
        if anonymize:
            logging.info('anonymizing messages')
            for idx, message in enumerate(messages):
                messages[idx].text = self.anonymize(message.text)

        return messages