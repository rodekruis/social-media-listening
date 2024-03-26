from datetime import datetime
import pandas as pd


class Message:
    """
    Message class
    """
    def __init__(self,
                 id_,
                 datetime_,
                 datetime_scraped_,
                 country,  # TBI move to info
                 source,  # social media source (twitter, telegram..)
                 text,
                 group=None,  # group, channel, page, account, etc.
                 reply=False,
                 reply_to=None,
                 translations=None,
                 info=None,  # country, locations, etc.
                 classifications=None):
        self.id_ = id_
        if isinstance(datetime_, datetime):
            self.datetime_ = datetime_
        else:
            self.datetime_ = pd.to_datetime(datetime_)
        if isinstance(datetime_scraped_, datetime):
            self.datetime_scraped_ = datetime_scraped_
        else:
            self.datetime_scraped_ = pd.to_datetime(datetime_scraped_)
        self.country = country
        self.source = source
        self.text = text
        self.group = group
        self.text = text
        self.reply = reply
        self.reply_to = reply_to
        if translations is None:
            self.translations = []
        else:
            self.translations = translations
        if info is None:
            self.info = {}
        else:
            self.info = info
        if classifications is None:
            self.classifications = []
        else:
            self.classifications = classifications

    def add_translation(self, dict_):
        if type(dict_) is not dict:
            raise TypeError("Translation must be a dictionary")
        if list(dict_.keys()) != ['text', 'from_lang', 'to_lang']:
            raise KeyError("Translation must contain the keys 'text', 'from_lang', 'to_lang'")
        self.translations.append(dict_)

    def set_translation(self, dict_):
        if type(dict_) is not dict:
            raise TypeError("Translation must be a dictionary")
        if list(dict_.keys()) != ['text', 'from_lang', 'to_lang']:
            raise KeyError("Translation must contain the keys 'text', 'from_lang', 'to_lang'")
        translation_found = False
        for ix, trans in enumerate(self.translations):
            if dict_['to_lang'] == trans['to_lang'] and dict_['from_lang'] == trans['from_lang']:
                self.translations[ix]['text'] = dict_['text']
                translation_found = True
        if not translation_found:
            self.translations.append(dict_)

    def add_classification(self, dict_list):
        if type(dict_list) is not list:
            raise TypeError("Classification must be a list of dictionaries")
        for dict_ in dict_list:
            if list(dict_.keys()) != ['class', 'score']:
                raise KeyError("Classification must contain the keys 'class', 'score'")
        self.classifications.extend(dict_list)

    def add_location(self, name, lon, lat):
        if 'locations' in self.info.keys():
            self.info['locations'].append({'name': name, 'longitude': lon, 'latitude': lat})
        else:
            self.info['locations'] = [{'name': name, 'longitude': lon, 'latitude': lat}]

    def to_dict(self):
        return {
            'id_': self.id_,
            'datetime_': self.datetime_,
            'datetime_scraped_': self.datetime_scraped_,
            'country': self.country,
            'source': self.source,
            'group': self.group,
            'text': self.text,
            'reply': self.reply,
            'reply_to': self.reply_to,
            'translations': self.translations,
            'info': self.info,
            'classifications': self.classifications
        }