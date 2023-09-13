from smm.message import Message
from smm.secrets import Secrets
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


supported_sources = ["twitter", "kobo", "telegram"]


class SocialMediaSource:
    """
    social media source
    """
    def __init__(self, name=None, secrets: Secrets=None):
        if name not in supported_sources:
            raise ValueError(f"Source {name} is not supported."
                             f"Supported sources are {', '.join(supported_sources)}")
        else:
            self.name = name
        self.secrets = self.set_secrets(secrets)

    def set_secrets(self, secrets):
        if not isinstance(secrets, Secrets):
            raise TypeError(f"invalid format of secrets, use secrets.Secrets")
        # if self.check_secrets(secrets):
        else:
            self.secrets = secrets
            return self.secrets

    def check_secrets(self):
        #TODO check that right secrets are filled for data source
        # here you check if all storage-specific secrets are present
        try:
            db_secret = self.secrets('azure-database-secret') # TODO change name of secrets to one needed in this script
        except ValueError:
            raise ValueError('Secrets for storage not found!')


class Extract:
    """
    extract data from social media
    """

    def __init__(self, source=None, secrets: Secrets=None):
        self.set_source(source, secrets)

    def set_source(self, source, secrets):
        if source is not None:
            self.source = SocialMediaSource(source, secrets)
        else:
            raise ValueError(f"Source not specified; provide one of {supported_sources}")

    def get_data(self,
                 start_date=datetime.today(),
                 end_date=datetime.today()-timedelta(days=7),
                 queries=None,
                 users=None,
                 channels=None,
                 pages=None,
                 store_temp=True):
        
        self.start_date = start_date
        self.end_date = end_date
        self.queries = queries
        self.users = users
        self.channels = channels
        self.pages = pages
        self.store_temp = store_temp

        if self.source.name == "twitter":
            logging.info('Getting Twitter data')
            messages = self.get_data_twitter()
        elif self.source.name == "kobo":
            logging.info('Getting Kobo data')
            messages = self.get_data_kobo()
        elif self.source.name == "telegram":
            logging.info('Getting Telegram data')
            messages = self.get_data_telegram()
        else:
            raise ValueError(f"source {self.source.name} is not supported."
                             f"Supported sources are {', '.join(supported_sources)}")

        return messages


    def get_data_twitter(self):
        auth = tweepy.OAuthHandler(
            self.source.secrets['API_CONSUMER_KEY'], 
            self.source.secrets['API_CONSUMER_SECRET']
            )
        auth.set_access_token(
            self.source.secrets['API_ACCESS_TOKEN'], 
            self.source.secrets['API_ACCESS_SECRET']
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
            self._save_temp(df_messages, 'tweets')

        return all_messages


    def get_data_kobo(self):
        url = f'{self.kobo_api_url}{self.source.secrets["ASSET"]}/data.json' # kobo api url: https://kobonew.ifrc.org/api/v2/assets/
        headers = {'Authorization': f'Token {self.source.secrets["TOKEN"]}'}
        data_request = requests.get(url,
                                    headers=headers)
        data = data_request.json()['results']
        df_form = pd.DataFrame(data)

        all_messages = []
        for idx, row in df_form.iterrows():
            # TBI: update mapping df to list of dict
            message = Message.from_kobo(row)
            all_messages.append(message)

        # save temp
        if self.store_temp:
            self._save_temp(df_form, 'form')

        return all_messages


    def get_data_telegram(self):
        # Set timmer to avoid scraping for too long
        telegram_client = TelegramClient(
            StringSession(self.source.secrets.SESSION_STRING),
            self.source.secrets.API_ID,
            self.source.secrets.API_HASH
        )
        # telegram_client.connect()
        # logging.info("Telegram client connected")

        # all_messages = []
        # df_messages = pd.DataFrame()

        loop = asyncio.get_event_loop()
        all_messages = loop.run_until_complete(
            self._scrape_messages(telegram_client,
                                self.channels, 
                                self.start_date, 
                                self.end_date)
                                )

        telegram_client.disconnect()

        # drop duplicates
        df_messages = pd.DataFrame(all_messages).reset_index(inplace=True)
        print('df_messages: ', df_messages)
        df_messages['datetime'] = pd.to_datetime(df_messages['datetime']).dt.strftime('%Y-%m-%d')
        df_messages = df_messages.drop_duplicates(subset=['id'])
        all_messages = df_messages.to_dict("records")

        # save temp
        if self.store_temp:
            self._save_temp(df_messages, 'messages')

        return all_messages


    async def _scrape_messages(self, telegram_client, telegram_channels, start_date, end_date):

        await telegram_client.connect()
        logging.info("Telegram client connected")

        time_limit = 60*45 # 45 min
        time_start = time.time()
        all_messages = []

        async with telegram_client:

            for channel in telegram_channels:
                logging.info(f"getting in telegram channel {channel}")
                # try:
                channel_entity = await telegram_client.get_entity(channel)
                # channel_full_info = await telegram_client(
                #     GetFullChannelRequest(channel=channel_entity))
                # scrape posts
                replied_post_id = []
                async for raw_message in telegram_client.iter_messages(
                    channel_entity,
                    offset_date = end_date,
                    reverse = True,
                    wait_time = 5
                ):
                    print('raw_message: ', raw_message)
                    reply = None
                    message = Message.from_telegram(raw_message)
                    all_messages.append(message)
                    if channel_entity.broadcast and raw_message.post and raw_message.replies:
                        replied_post_id.append(raw_message.id)
                    
                    # # scrape replies
                    # for post_id in replied_post_id:
                    #     try:
                    #         async for raw_reply in telegram_client.iter_messages(
                    #             channel_entity,
                    #             offset_date = end_date,
                    #             reverse = True,
                    #             reply_to = post_id,
                    #             wait_time = 5
                    #         ):
                    #             reply = Message.from_telegram(raw_reply)
                    #             all_messages.append(reply)
                    #             time.sleep(5)
                    #     except Exception as e:
                    #         logging.info(f"getting replies for {message.id} failed: {e}")
                        
                    #     time_duration = time.time() - time_start
                    #     if time_duration >= time_limit:
                    #         logging.info(f"getting replies for {channel} stopped: timeout {time_duration} seconds")
                    #         break
                    else:
                        time.sleep(10)
                        continue
                # except Exception as e:
                #     logging.info(f"Unable to get in telegram channel {channel}: {e}")
                #     break
        
        await telegram_client.disconnect()

        return all_messages


    def _save_temp(self, df, dataname):
        # save temp
        if not os.path.exists('./temp'):
            os.mkdir('./temp')
        filename = f'./temp/{self.source}_{dataname}_{self.end_date}_{self.start_date}.csv'
        df.to_csv(filename, index=False)