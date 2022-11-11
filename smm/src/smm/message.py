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
                 translation=None,
                 info=None,
                 classification=None):
        self.id_ = id_
        if isinstance(datetime_, datetime):
            self.datetime_ = datetime
        else:
            self.datetime_ = pd.to_datetime(datetime_)
        self.source = source
        self.text = text
        self.group = group
        self.reply = reply
        self.reply_to = reply_to
        if translation is None:
            self.translation = {}
        if info is None:
            self.info = {}
        if classification is None:
            self.classification = {}

    def from_twitter(self, dict_):
        # TBI automatically map from tweet

    def from_telegram(self, dict_):
        # TBI automatically map from telegram

    def from_kobo(self, dict_):
        # TBI automatically map from telegram

    def set_translation(self, dict_):
        # TBI check that dict_ follow structure
        # [{"text": "...", "from": "...", "to": "..."}, ...]

    def set_classification(self, dict_):
        # TBI check that dict_ follow structure
        # [{"class": "...", "score": "..."}, ...]

