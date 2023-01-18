import os
from dotenv import dotenv_values
import yaml
from datetime import datetime, timedelta
import time
import pandas as pd
import json
import tweepy
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.functions.channels import GetFullChannelRequest
import requests
import logging


supported_sources = ["twitter", "kobo", "telegram"]
supported_secret_locations = ["env", "file", "azure"]

class SocialMediaSecrets:
    """
    Secrets (API keys, tokens, etc.) for social media source.
    Secrets should be in json format.
    """
    def __init__(self,
                 source=None,
                 secret_location=None): # EX: SocialMediaSecrets("twitter", "env")
        if secret_location is None:
            self.secret_location = "env"
        elif secret_location not in supported_secret_locations:
            raise ValueError(f"storage {secret_location} is not supported."
                             f"Supported storages are {supported_secret_locations}")
        else:
            self.secret_location = secret_location.lower()
        self.source = source.lower()


    def get_secret(self):
        if self.secret_location == "env":
            sm_secrets = self.get_secret_env()
        if self.secret_location == "file":
            sm_secrets = self.get_secret_file()
        if self.secret_location == "azure":
            sm_secrets = self.get_secret_azure()
        return sm_secrets


    def get_secret_env(self):
        sm_secrets = dotenv_values(f"../credentials/env")
        return sm_secrets


    def get_secret_file(self):
        secret_file_dir = f"../credentials/secrets.json"
        with open(secret_file_dir, "r") as secret_file:
            if secret_file_dir.endswith("json"):
                sm_secrets = json.load(secret_file) #TODO: set constraints of json file format?
            elif secret_file_dir.endswith("yaml"):
                sm_secrets = yaml.load(secret_file, Loader=yaml.FullLoader)
            return sm_secrets


    def get_secret_azure(self):
        sm_secrets = self.load.get_secret_keyvault(self.source)
        sm_secrets = json.loads(sm_secrets)
        return sm_secrets


    def _get_secret_names(self):
        if self.source == "twitter":
            secret_names = ["API_ACCESS_TOKEN", 
                            "API_ACCESS_SECRET", 
                            "API_CONSUMER_KEY", 
                            "API_CONSUMER_SECRET"]
        elif self.source == "kobo":
            secret_names = ["TOKEN", 
                            "ASSET"]
        elif self.source == "telegram":
            secret_names = ["API_ID", 
                            "API_HASH", 
                            "SESSION_STRING"]
            return secret_names


class SocialMediaSource:
    """
    social media source
    """
    def __init__(self, name, secrets=None): #EX: SocialMediaSource("twitter", twitter_secrets)
        if name not in supported_sources:
            raise ValueError(f"source {name} is not supported."
                             f"Supported sources are {', '.join(supported_sources)}")
        else:
            self.name = name
        self.secrets = secrets

    def set_secrets(self, secrets):
        if not isinstance(secrets, SocialMediaSecrets):
            raise TypeError(f"invalid format of secrets, use extract.SMSecrets")
        self.secrets = secrets

    def check_secrets(self):
        secret_names = self._get_secret_names(self.source)
        # TBI check that right secrets are filled for data source
        if all(x in self.secrets.keys() for x in secret_names) and \
            any("" in val for val in self.secrets.values()):
            return True
        else:
            # TODO: raise error
            pass


class Extract:
    """
    extract data from social media
    """

    def __init__(self, source=None, secrets=None): #EX: smm.Extract("twitter", twitter_secrets)
        if source is not None:
            self.source = SocialMediaSource(source, secrets)
        else:
            self.source = None

    def set_source(self, source, secrets):
        self.source = SocialMediaSource(source, secrets)

    def get_data(self,
                 start_date=datetime.today(),
                 end_date=datetime.today()-timedelta(days=7),
                 queries=None,
                 users=None,
                 channels=None,
                 pages=None):
        
        self.start_date = start_date
        self.end_date = end_date
        self.queries = queries
        self.users = users
        self.channels = channels
        self.pages = pages

        if not self.source.check_secrets():
            raise ValueError("no social media secrets found")

        if self.source.name == "twitter":
            # TBI get data from Twitter
            logging.info('Getting Twitter data')
            messages = self.get_data_twitter()
        elif self.source.name == "kobo":
            # TBI get data from Twitter
            logging.info('Getting Kobo data')
            messages = self.get_data_kobo()
        elif self.source.name == "telegram":
            # TBI get data from Telegram
            logging.info('Getting Telegram data')
            messages = self.get_data_telegram()
        else:
            raise ValueError(f"source {self.source.name} is not supported."
                             f"Supported sources are {', '.join(supported_sources)}")

        return messages


    def get_data_twitter(self):
        # initialize twitter API
        twitter_secrets = self.source.secrets.copy()
        auth = tweepy.OAuthHandler(
            twitter_secrets['API_CONSUMER_KEY'], 
            twitter_secrets['API_CONSUMER_SECRET']
            )
        auth.set_access_token(
            twitter_secrets['API_ACCESS_TOKEN'], 
            twitter_secrets['API_ACCESS_SECRET']
            )
        api = tweepy.API(auth, wait_on_rate_limit=True)

        # track individual twitter users
        if not self.users:
            tw_users = self.users

            for userID in tw_users:
                logging.info(f'Getting Twitter user: {userID}')

                tweets = api.user_timeline(screen_name=userID,
                                           count=200,
                                           include_rts=False,
                                           tweet_mode='extended'
                                           )
                all_tweets = []
                all_tweets.extend(tweets)
                oldest_id = tweets[-1].id
                while True:
                    tweets = api.user_timeline(screen_name=userID,
                                               count=200,
                                               include_rts=False,
                                               max_id=oldest_id - 1,
                                               tweet_mode='extended'
                                               )
                    if len(tweets) == 0:
                        break
                    oldest_id = tweets[-1].id
                    all_tweets.extend(tweets)
                df_tweets = pd.json_normalize(all_tweets)

        # track specific queries
        if not self.queries:
            all_tweets = []
            # loop over queries and search
            for query in self.queries:
                try:
                    for page in tweepy.Cursor(api.search_tweets,
                                            q=query,
                                            # geocode="39.822197,34.808097,800km",
                                            # lang="tr",
                                            include_entities=True,
                                            count=100).pages():
                        try:
                            for tweet in page:
                                all_tweets.append(tweet)
                        except Exception as e:
                            logging.warning(f"Error {e}, skipping page {n}")
                            pass
                except Exception as e:
                    logging.warning(f"Error {e}, skipping page {n}")
                    pass
            
            if 'df_tweets' in locals():
                df_tweets_queries = pd.json_normalize(all_tweets)
                df_tweets = df_tweets.append(df_tweets_queries, ignore_index=True)
            else:
                df_tweets = pd.json_normalize(all_tweets)

        # drop duplicates
        df_tweets = df_tweets.drop_duplicates(subset=['id'])
        return df_tweets


    def get_data_kobo(self):
        kobo_secrets = self.source.secrets.copy()
        url = f'https://kobonew.ifrc.org/api/v2/assets/{kobo_secrets["asset"]}/data.json'
        headers = {'Authorization': f'Token {kobo_secrets["token"]}'}
        data_request = requests.get(url,
                                    headers=headers)
        data = data_request.json()['results']
        df_form = pd.DataFrame(data)
        # TB updated
        return df_form


    def get_data_telegram(self):
        telegram_secrets = self.source.secrets.copy()

        # Set timmer to avoid scraping for too long
        time_limit = 60*45 # 45 min
        time_start = time.time()
        telegram_client = TelegramClient(
            StringSession(telegram_secrets["SESSION_STRING"]),
            telegram_secrets["API_ID"],
            telegram_secrets["API_HASH"]
        )
        telegram_client.connect()

        df_messages = pd.DataFrame()
        df_member_counts = pd.DataFrame()
        for channel in self.channels:
            logging.info(f"Getting in Telegram channel {channel}")
            try:
                channel_entity = telegram_client.get_entity(channel)
                channel_full_info = telegram_client(
                    GetFullChannelRequest(channel=channel_entity)
                    )

                for message in telegram_client.iter_messages(
                    channel_entity,
                    offset_date = self.end_date,
                    reverse = True,
                    wait_time = 5
                ):
                    reply = None
                    df_messages = self._arrange_telegram_messages(df_messages, message, reply, channel)
                    if channel_entity.broadcast and message.post and message.replies:
                        df_replies = pd.DataFrame()
                        try:
                            for reply in telegram_client.iter_messages(
                                channel_entity,
                                reply_to=message.id,
                                wait_time = 5
                            ):
                                df_replies = self._arrange_telegram_messages(df_replies, message, reply, channel)
                                time.sleep(5)
                            df_messages = df_messages.append(df_replies, ignore_index=True)
                        except Exception as e:
                            logging.warning(f"Skipping replies for {message.id}: {e}")
                        time_duration = time.time() - time_start
                        if time_duration >= time_limit:
                            logging.warning(f"Getting replies for {channel} stopped: timeout {time_duration} seconds")
                            break
                else:
                    df_member_counts = self._count_messages(df_member_counts, df_messages, channel_full_info)
                    time.sleep(10)
                    continue
            except Exception as e:
                logging.warning(f"Failed getting in Telegram channel {channel}: {e}")
                break

        telegram_client.disconnect()

        # Add index column
        df_member_counts.reset_index(inplace=True)
        df_member_counts['id'] = df_member_counts.index
        df_messages.reset_index(inplace=True)
        df_messages['id'] = df_messages.index
        df_messages['datetime'] = pd.to_datetime(df_messages['datetime']).dt.strftime('%Y-%m-%d')

        return df_messages


    def _arrange_telegram_messages(self, df_messages, message, reply, channel):
        '''
        Arrange posts and their replies from Telegram channel
        '''
        ix = len(df_messages)
        df_messages.at[ix, "source"] = channel
        df_messages.at[ix, "text"] = str(message.text)
        if reply:
            df_messages.at[ix, "id"] = f'{message.id}-{reply.id}'
            df_messages.at[ix, "reply_text"] = str(reply.text)
            df_messages.at[ix, "datetime"] = reply.date
            df_messages.at[ix, "reply"] = True
        else:
            df_messages.at[ix, "id"] = f'{message.id}-0'
            df_messages.at[ix, "reply_text"] = reply
            df_messages.at[ix, "datetime"] = message.date
            df_messages.at[ix, "reply"] = False
        return df_messages


    def _count_messages(self, df_count, df_messages, channel_full_info):
        
        idx = len(df_count)
        df_count.at[idx, 'source'] = self.source
        member_count = channel_full_info.full_chat.participants_count
        df_count.at[idx, 'member_count'] = member_count
        df_count.at[idx, 'date'] = self.start_date
        df_count.at[idx, 'message_count'] = len(df_messages[df_messages['source']==self.source])
        
        return df_count