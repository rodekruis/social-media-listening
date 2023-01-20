from datetime import datetime
import pandas as pd


class Message:
    """
    Message class
    """
    def __init__(self,
                 id_,
                 datetime_,
                 source,  # social media source (twitter, telegram..)
                 text,
                 group=None,  # group, channel, page, account, etc.
                 reply=False,
                 reply_text=None,
                 reply_to=None,
                 translations=None,
                 info=None,  # country, etc.
                 classifications=None):
        self.id_ = id_
        if isinstance(datetime_, datetime):
            self.datetime_ = datetime_
        else:
            self.datetime_ = pd.to_datetime(datetime_)
        self.source = source
        self.text = text
        self.group = group
        self.text = text
        self.reply = reply
        self.reply_text = reply_text
        self.reply_to = reply_to
        if translations is None:
            self.translations = []
        if info is None:
            self.info = {}
        if classifications is None:
            self.classifications = []

    def from_twitter(self, dict_):
        # TBI automatically map from tweet
        pass

    def from_telegram(self, dict_):
        # TBI automatically map from telegram
        pass

    def from_kobo(self, dict_):
        # TBI automatically map from telegram
        pass

    def set_translation(self, dict_):
        if type(dict_) is not dict:
            raise TypeError("Translation must be a dictionary")
        if list(dict_.keys()) != ['text', 'from_lang', 'to_lang']:
            raise KeyError("Translation must contain the keys 'text', 'from_lang', 'to_lang'")
        self.translations.append(dict_)

    def set_classification(self, dict_list):
        if type(dict_list) is not list:
            raise TypeError("Classification must be a list of dictionaries")
        for dict_ in dict_list:
            if list(dict_.keys()) != ['class', 'score']:
                raise KeyError("Classification must contain the keys 'class', 'score'")
        self.classifications.extend(dict_list)

    def to_dict(self):
        return {
            'id': self.id_,
            'datetime': self.datetime_,
            'source': self.source,
            'group': self.group,
            'text': self.text,
            'reply': self.reply,
            'reply_to': self.reply_to,
            'translations': self.translations,
            'info': self.info,
            'classifications': self.classifications
        }

