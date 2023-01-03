from smm.message import Message
import re
import stopwordsiso
import pandas as pd
from transformers import pipeline
from google.cloud import translate_v2 as google_translate
from google.oauth2 import service_account as google_service_account
from time import sleep
import logging
import json
import requests
import uuid
import spacy
supported_translators = ["HuggingFace", "Google", "Microsoft"]
supported_classifiers = ["HuggingFace", "URL"]
supported_anonymizers = ["anonymization-app"]
supported_classifier_tasks = ["sentiment-analysis", "zero-shot-classification"]


class Transform:
    """
    transform data
    """

    def __init__(self):
        # translator fields
        self.translator_name = None
        self.from_lang = None
        self.to_lang = None
        self.translator_secrets = None
        self.translator = None
        # classifier fields
        self.classifier_name = None
        self.classifier_lang = None
        self.classifier_task = None
        self.class_labels = None
        self.classifier = None
        # anonymizer fields
        self.anonymizer_name = None
        self.anonymizer = None
        # wordfreq fields
        self.wordfreq_lang = None
        self.lemmatizer = None

    def set_translator(self, from_lang, to_lang, name=None, secrets=None):
        self.from_lang = from_lang
        self.to_lang = to_lang
        if self.from_lang is None or self.to_lang is None:
            raise ValueError(f"Original and target language must be specified for translator")

        if name is None:
            self.translator_name = "HuggingFace"
        else:
            self.translator_name = name

        if self.translator_name == "HuggingFace":
            try:
                self.translator = pipeline(f"translation_{from_lang}_to_{to_lang}")
            except ValueError:
                try:
                    self.translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{from_lang}-{to_lang}")
                except:
                    raise KeyError("Translation model not found on HuggingFace, please use Microsoft or Google")

        elif self.translator_name == "Microsoft":
            constructed_url = secrets['MSCOGNITIVE_URL']
            params = {
                'api-version': '3.0',
                'from': [from_lang],
                'to': [to_lang],
            }
            headers = {
                'Ocp-Apim-Subscription-Key': secrets["MSCOGNITIVE_KEY"],
                'Ocp-Apim-Subscription-Region': secrets["MSCOGNITIVE_LOCATION"],
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
            }
            self.translator = [constructed_url, params, headers]
            self.translator_secrets = secrets

        elif self.translator_name == "Google":
            service_account_info = secrets["GOOGLE_SERVICEACCOUNT"]
            credentials = google_service_account.Credentials.from_service_account_info(json.loads(service_account_info))
            self.translator = google_translate.Client(credentials=credentials)
            self.translator_secrets = secrets

        else:
            raise ValueError(f"Translator {name} is not supported. "
                             f"Supported translators are {supported_translators}")

    def translate_text(self, text):
        if self.translator_name is None:
            raise RuntimeError("Translator not initialized, use set_translator()")
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
                    response = self.translator.translate(text, target_language=self.to_lang)
                    translation_data = {
                        "text": response["translatedText"],
                        "from_lang": self.from_lang,
                        "to_lang": self.to_lang,
                    }
                except Exception as e:
                    logging.error(e)
                    sleep(10)

        if translation_data['text'] == text:
            logging.warning("Translator returned identical message, check configuration.")
        return translation_data

    def translate_message(self, message):
        translation = self.translate_text(message.text)
        message.set_translation(translation)
        return message

    def set_classifier(self, name=None, lang=None, task=None, class_labels=None, secrets=None):
        self.class_labels = class_labels
        if name is None:
            self.classifier_name = "HuggingFace"
        else:
            self.classifier_name = name
        if task is None:
            self.classifier_task = "sentiment-analysis"
        else:
            self.classifier_task = task

        if lang is None:
            self.classifier_lang = "en"
        else:
            self.classifier_lang = lang

        if self.classifier_task == "sentiment-analysis":
            self.class_labels = ["POSITIVE", "NEGATIVE"]
            if class_labels is not None:
                logging.warning(f"Classifier task sentiment-analysis uses fixed class labels, "
                                f"overwriting {class_labels} with {self.class_labels}.")
        else:
            if class_labels is None:
                raise ValueError(f"Class labels must be specified for classifier task {self.classifier_task}")
            self.class_labels = class_labels

        if self.classifier_name == "HuggingFace":
            if self.classifier_task == "sentiment-analysis":
                self.classifier = pipeline(f"sentiment-analysis",
                                           model="distilbert-base-uncased-finetuned-sst-2-english")
            elif self.classifier_task == "zero-shot-classification":
                self.classifier = pipeline(f"zero-shot-classification",
                                           model="facebook/bart-large-mnli")
            else:
                raise ValueError(f"Classifier task {task} is not supported. "
                                 f"Supported classifier tasks are {supported_classifier_tasks}")

        elif self.translator_name == "URL":
            if self.classifier_task in supported_classifier_tasks:
                self.classifier = secrets["CLASSIFIER_URL"]
            else:
                raise ValueError(f"Classifier task {task} is not supported. "
                                 f"Supported classifier tasks are {supported_classifier_tasks}")

        else:
            raise ValueError(f"Classifier {name} is not supported. "
                             f"Supported classifiers are {supported_translators}")

    def classify_text(self, text):
        if self.classifier_name is None:
            raise RuntimeError("Classifier not initialized, use set_classifier()")
        classification_data = []

        if self.classifier_name == "HuggingFace":
            if self.classifier_task == "sentiment-analysis":
                result = self.classifier(text)[0]
                classification_data = [
                    {'class': result['label'], 'score': result['score']},
                    {'class': 'POSITIVE' if result['label'] == 'NEGATIVE' else 'NEGATIVE', 'score': 1.-result['score']}
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
            logging.warning("Classifier returned no results, check configuration")
        return classification_data

    def classify_message(self, message):
        try:
            text = next(x['text'] for x in message.translations if x['to_lang'] == self.classifier_lang)
        except StopIteration:
            logging.warning(f"Classifier language is {self.classifier_lang}, no specific translations found. "
                            "Classifying on original language.")
            text = message.text
        classification = self.classify_text(text)
        message.set_classification(classification)
        return message

    def set_anonymizer(self, name=None):
        if name is None:
            self.anonymizer_name = "anonymization-app"
            self.anonymizer = "https://anonymization-app.azurewebsites.net/anonymize/"
        else:
            self.anonymizer_name = name

        if self.anonymizer_name != "anonymization-app":
            raise ValueError(f"Anonymizer {name} is not supported. "
                             f"Supported anonymizers are {supported_anonymizers}")

    def anonymize(self, text):
        if self.classifier_name is None:
            raise RuntimeError("Anonymizer not initialized, use set_anonymizer()")
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
        return 0

    def process_messages(self,
                         messages,
                         translate=False,
                         classify=False,
                         anonymize=False,
                         geolocate=False):
        if translate:
            logging.info('Translating messages')
            for idx, message in enumerate(messages):
                messages[idx] = self.translate_message(message)
        if classify:
            logging.info('Classifying messages')
            for idx, message in enumerate(messages):
                messages[idx] = self.classify_message(message)
        if geolocate:
            # TBI geolocate messages
            pass
        if anonymize:
            logging.info('Anonymizing messages')
            for idx, message in enumerate(messages):
                messages[idx].text = self.anonymize(message.text)

        return messages

    def set_wordfreq(self, wordfreq_lang, lemmatizer_model=None):

        self.wordfreq_lang = wordfreq_lang
        if self.wordfreq_lang is None:
            raise ValueError(f"Original language must be specified for word frequencies")
        if lemmatizer_model is None:
            # try to get the standard spaCy model "{lang}_core_news_sm"
            spacy_model_name = f"{self.wordfreq_lang}_core_news_sm"
            if not spacy.util.is_package(spacy_model_name):
                try:
                    spacy.cli.download(spacy_model_name)
                except:
                    # model not found
                    raise ValueError(f"spaCy model {spacy_model_name} not found, "
                                     "please specify it as lemmatizer_model=...")
        else:
            # if model name is specified, use that one
            spacy_model_name = lemmatizer_model
            if not spacy.util.is_package(spacy_model_name):
                try:
                    spacy.cli.download(spacy_model_name)
                except:
                    # model not found
                    raise ValueError(f"spaCy model {spacy_model_name} not found")
        self.lemmatizer = spacy.load(spacy_model_name, disable=["tokenizer", "parser", "ner"])

    def get_wordfreq(self, messages):
        if self.lemmatizer is None:
            raise RuntimeError("Lemmatizer not initialized, use set_wordfreq()")

        text = ''
        for idx, message in enumerate(messages):
            text = text + " " + message.text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        text_list = text.split()  # create list of space-separated words

        # lemmatize every word
        text_list_lemmatized = []
        for text in text_list:
            lemmatized_text = " ".join([token.lemma_ for token in self.lemmatizer(text)])
            text_list_lemmatized.append(lemmatized_text)
        text_list = text_list_lemmatized.copy()

        # Remove stopwords and numbers
        stop_words = stopwordsiso.stopwords([self.wordfreq_lang])
        text_list = [word for word in text_list if word not in stop_words and not word.isnumeric()]

        dict_word_freq = {}
        # Take each word from text_list and count occurrence
        for element in text_list:
            # check if each word has '.' at its last. If so then ignore '.'
            if element[-1] == '.':
                element = element[0:len(element) - 1]
            # if there exists a key as "elements" then simply increase its value.
            if element in dict_word_freq:
                dict_word_freq[element] += 1
            else:
                dict_word_freq.update({element: 1})
        # sort dict
        dict_word_freq = {
            key: value for key, value in sorted(
                dict_word_freq.items(),
                key=lambda item: item[1],
                reverse=True
            )
        }
        # map to dataframe
        df_word_freq = pd.DataFrame.from_dict(dict_word_freq, orient='index')
        df_word_freq.reset_index(inplace=True)
        df_word_freq.columns = ['Word', 'Frequency']

        # if translator is specified, use it to translate wordfreq
        if self.translator_name is not None:
            if self.from_lang != self.wordfreq_lang:
                self.set_translator(from_lang=self.wordfreq_lang, to_lang=self.to_lang, name=self.translator_name,
                                    secrets=self.translator_secrets)
            df_word_freq['Translation'] = df_word_freq['Word'].apply(lambda x: self.translate(x)['text'], axis=1)

        return df_word_freq
