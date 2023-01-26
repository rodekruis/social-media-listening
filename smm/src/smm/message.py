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
        self.reply_to = reply_to
        if translations is None:
            self.translations = []
        if info is None:
            self.info = {}
        if classifications is None:
            self.classifications = []

    def from_twitter(self, dict_):
        self.id = dict_.id
        self.datetime_ = dict_.created_at
        self.source = "twitter"
        self.group = None
        self.text = dict_.full_text
        self.reply_to = dict_.in_reply_to_status_id
        if self.reply_to:
            self.reply = True
        else:
            self.reply = False
        self.translations = None
        self.info = {}
        self.classifications = []

    def from_telegram(self, dict_):
        self.id = dict_.id
        self.datetime_ = dict_.date
        self.source = "telegram"
        self.group = dict_.PeerChannel.channel_id # TODO: verify
        self.text = dict_.message
        if dict_.post:
            self.reply = False
            self.reply_to = None
        else:
            self.reply = True
            self.reply_to = dict_.MessageReplyHeader.reply_to_msg_id # TODO: verify 
        self.translations = []
        self.info = {}
        self.classifications = []

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

