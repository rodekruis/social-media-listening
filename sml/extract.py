from sml.message import Message
from sml.secrets import Secrets
import os
from datetime import datetime, timedelta
import time
import pandas as pd
import tweepy
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.functions.channels import GetFullChannelRequest
import requests
import logging
import asyncio

supported_sources = ["telegram"]  # "twitter", "kobo",


class Extract:
    """
    extract data from social media
    """
    
    def __init__(self, secrets: Secrets = None):
        self.source = None
        self.start_date = None
        self.end_date = None
        self.country = None
        self.queries = None
        self.users = None
        self.channels = None
        self.pages = None
        self.store_temp = None
        self.secrets = None
        if secrets is not None:
            self.set_secrets(secrets)
        
    def set_secrets(self, secrets):
        if not isinstance(secrets, Secrets):
            raise TypeError(f"invalid format of secrets, use secrets.Secrets")
        missing_secrets = []
        if self.source == "telegram":
            missing_secrets = secrets.check_secrets(
                [
                    "STRING_SESSION",
                    "API_ID",
                    "API_HASH"
                ]
            )
        if self.source == "twitter":
            missing_secrets = secrets.check_secrets(
                [
                    "API_CONSUMER_KEY",
                    "API_CONSUMER_SECRET",
                    "API_ACCESS_TOKEN",
                    "API_ACCESS_SECRET"
                ]
            )
        if self.source == "kobo":
            missing_secrets = secrets.check_secrets(
                [
                    "KOBO_URL",
                    "KOBO_ASSET",
                    "KOBO_TOKEN"
                ]
            )
        if missing_secrets:
            raise Exception(f"Missing secret(s) {', '.join(missing_secrets)} for source {self.source}")
        else:
            self.secrets = secrets
            return self
    
    def set_source(self, source_name, secrets: Secrets = None):
        if source_name is not None:
            if source_name not in supported_sources:
                raise ValueError(f"Source {source_name} is not supported."
                                 f"Supported sources are {', '.join(supported_sources)}")
            else:
                self.source = source_name
        else:
            raise ValueError(f"Source not specified; provide one of {', '.join(supported_sources)}")
        if secrets is not None:
            self.set_secrets(secrets)
        elif self.secrets is not None:
            self.set_secrets(self.secrets)
        else:
            raise ValueError(f"Set secrets before setting source")
        return self
    
    def get_data(self,
                 start_date=datetime.today(),
                 end_date=datetime.today() - timedelta(days=7),
                 country=None,
                 queries=None,
                 users=None,
                 channels=None,
                 pages=None,
                 store_temp=True):
        if self.source is None:
            raise RuntimeError("Source not specified, use set_source()")
        self.start_date = start_date.date()
        self.end_date = end_date.date()
        self.country = country
        self.queries = queries
        self.users = users
        self.channels = channels
        self.pages = pages
        self.store_temp = store_temp
        messages = []
        if self.source == "twitter":
            logging.info('Getting Twitter data')
            messages = self.get_data_twitter()
        elif self.source == "kobo":
            logging.info('Getting Kobo data')
            messages = self.get_data_kobo()
        elif self.source == "telegram":
            logging.info('Getting Telegram data')
            messages = self.get_data_telegram()
        return messages
    
    def get_data_twitter(self):
        auth = tweepy.OAuthHandler(
            self.secrets['API_CONSUMER_KEY'],
            self.secrets['API_CONSUMER_SECRET']
        )
        auth.set_access_token(
            self.secrets['API_ACCESS_TOKEN'],
            self.secrets['API_ACCESS_SECRET']
        )
        api = tweepy.API(auth, wait_on_rate_limit=True)
        all_messages = []
        
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
                # map all raw tweets to messages
                for tweet in all_tweets:
                    message = Message.from_twitter(tweet)
                    all_messages.append(message)
        
        # track specific queries
        if not self.queries:
            all_tweets = []
            # loop over queries and search
            for query in self.queries:
                try:
                    for page in tweepy.Cursor(api.search_tweets,
                                              q=query,
                                              include_entities=True,
                                              count=100).pages():
                        try:
                            for tweet in page:
                                all_tweets.append(tweet)
                                message = Message.from_twitter(tweet)
                                all_messages.append(message)
                        except Exception as e:
                            logging.warning(f"Error {e}, skipping page {n}")
                            pass
                except Exception as e:
                    logging.warning(f"Error {e}, skipping page {n}")
                    pass
        
        # drop duplicates
        df_messages = pd.DataFrame(all_messages)
        df_messages = df_messages.drop_duplicates(subset=['id'])
        all_messages = df_messages.to_dict("records")
        
        # save temp
        if self.store_temp:
            self._save_temp(df_messages, 'messages')
        
        return all_messages
    
    def get_data_kobo(self):
        url = f'{self.secrets["KOBO_URL"]}/api/v2/assets/{self.secrets["KOBO_ASSET"]}/data.json'
        headers = {'Authorization': f'Token {self.secrets["KOBO_TOKEN"]}'}
        data_request = requests.get(url, headers=headers)
        records = data_request.json()['results']
        messages = []
        for record in records:
            # TBI: update mapping df to list of dict
            message = Message(
                id_=record['_id'],
                datetime_=record['_submission_time'],
                datetime_scraped_=datetime.today(),
                country=self.country,
                source="Kobo",
                text=record['text'],
                group=record['group'],
                info={"kobo_asset": self.secrets["KOBO_ASSET"]}
            )
            messages.append(message)
        
        # save temp
        if self.store_temp:
            df_messages = pd.DataFrame([x.to_dict() for x in messages])
            self._save_temp(df_messages, 'messages')
        
        return messages
    
    def get_data_telegram(self):
        telegram_client = TelegramClient(
            StringSession(self.secrets.get_secret('STRING_SESSION')),
            self.secrets.get_secret('API_ID'),
            self.secrets.get_secret('API_HASH')
        )
        
        # loop = asyncio.get_event_loop()
        messages = telegram_client.loop.run_until_complete(
            self._scrape_telegram(telegram_client,
                                  self.channels,
                                  self.start_date)
        )
        
        # drop duplicates
        messages = self._deduplicate(messages)
        
        # save temp
        if self.store_temp:
            df_messages = pd.DataFrame([x.to_dict() for x in messages])
            self._save_temp(df_messages, 'messages')
        
        return messages
    
    async def _scrape_telegram(self, telegram_client, telegram_channels, start_date):
        
        await telegram_client.connect()
        all_messages = []
        
        for channel in telegram_channels:
            # Set timer per channel to avoid scraping for too long
            time_out = time.time() + 60 * 10  # = 10 min
            channel_count = 0
            
            logging.info(f"starting telegram channel {channel} at {time.time()}")
            
            try:
                channel_entity = await telegram_client.get_entity(channel)
                
                # get number of channel members
                channel_full_info = await telegram_client(GetFullChannelRequest(channel=channel_entity))
                channel_members = channel_full_info.full_chat.participants_count
                
                # scrape messages
                replied_post_id = []
                async for raw_message in telegram_client.iter_messages(
                        channel_entity,
                        offset_date=start_date,
                        reverse=True,
                        wait_time=5
                ):
                    message = self._from_telegram(raw_message, channel, channel_members)
                    if message.text != "":
                        all_messages.append(message)
                        channel_count += 1
                    if time.time() >= time_out:
                        logging.info(f"time_out channel {channel} at {time.time()}")
                        break
                    
                    if channel_entity.broadcast and raw_message.post and raw_message.replies:
                        replied_post_id.append(raw_message.id)
                        # scrape replies
                        for post_id in replied_post_id:
                            try:
                                async for raw_reply in telegram_client.iter_messages(
                                        channel_entity,
                                        offset_date=start_date,
                                        reverse=True,
                                        reply_to=post_id,
                                        wait_time=5
                                ):
                                    reply = self._from_telegram(raw_reply, channel, channel_members)
                                    if reply.text != "":
                                        all_messages.append(reply)
                                        channel_count += 1
                            except Exception as e:
                                logging.info(f"getting replies for {message.id} failed: {e}")
            except Exception as e:
                logging.info(f"Unable to get in telegram channel {channel}: {e}")
            logging.info(f"found {channel_count} messages in channel {channel}")
            
        telegram_client.disconnect()
        return all_messages
    
    def _from_telegram(self, message_entity, channel_name, channel_members=0):
        if not message_entity.reply_to:
            reply_ = False
            reply_to_ = None
        else:
            reply_ = True
            reply_to_ = message_entity.reply_to.reply_to_msg_id
        return Message(
            id_=message_entity.id,
            datetime_=message_entity.date,
            datetime_scraped_=datetime.today(),
            country=self.country,
            source="Telegram",
            group=channel_name,
            text=message_entity.message,
            reply=reply_,
            reply_to=reply_to_,
            info={"group_members": channel_members}
        )
    
    def _deduplicate(self, object_list):
        seen = set()
        uniqueidlist = []
        for obj in object_list:
            if obj.id_ not in seen:
                seen.add(obj.id_)
                uniqueidlist.append(obj)
        return uniqueidlist
    
    def _save_temp(self, df, dataname):
        # save temp
        if not os.path.exists('./temp'):
            os.mkdir('./temp')
        filename = f'./temp/{self.source}_{dataname}_{self.start_date}_{self.end_date}.csv'
        df.to_csv(filename, index=False)
