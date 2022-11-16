from datetime import datetime, timedelta
import time
import pandas as pd
import json
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.functions.channels import GetFullChannelRequest
import requests

supported_sources = ["twitter", "kobo", "telegram"]
supported_secret_locations = ["env", "file", "azure"]

class SocialMediaSecrets:
    """
    Secrets (API keys, tokens, etc.) for social media source
    """
    def __init__(self,
                 source=None,
                 secret_location=None):
        if secret_location is None:
            self.secret_location = "env"
        elif secret_location not in supported_secret_locations:
            raise ValueError(f"storage {secret_location} is not supported."
                             f"Supported storages are {supported_secret_locations}")
        else:
            self.secret_location = secret_location
        self.source = source

    def get_secret(self):
        if self.secret_location == "env":
            sm_secrets = self.get_secret_env()
        if self.secret_location == "file":
            sm_secrets = self.get_secret_file()
        if self.secret_location == "azure":
            sm_secrets = self.get_secret_azure()
        return sm_secrets
    
    def get_secret_env(self):
        # TBI
        return []

    def get_secret_file(self):
        # TBI
        return []
    
    def get_secret_azure(self):
        # TBI
        return []


class SocialMediaSource:
    """
    social media source
    """
    def __init__(self, name, secrets=None):
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
        # TBI check that right secrets are filled for data source
        return True


class Extract:
    """
    extract data from social media
    """

    def __init__(self, source=None, secrets=None):
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
            messages = self.get_data_twitter()
        elif self.source.name == "kobo":
            # TBI get data from Twitter
            messages = self.get_data_kobo()
        elif self.source.name == "telegram":
            # TBI get data from Telegram
            messages = self.get_data_telegram()
        else:
            raise ValueError(f"source {self.source.name} is not supported."
                             f"Supported sources are {', '.join(supported_sources)}")

        return messages

    def get_data_twitter(self):
        # TBI
        return []

    def get_data_kobo(self):
        # TBI
        # get data from kobo
        kobo_secrets = json.loads(self.source.secrets)
        headers = {'Authorization': f'Token {kobo_secrets["token"]}'}
        data_request = requests.get(f'https://kobonew.ifrc.org/api/v2/assets/{kobo_secrets["asset"]}/data.json',
                                    headers=headers)
        data = data_request.json()['results']
        df_form = pd.DataFrame(data)


        return []

    def get_data_telegram(self):
        # TBI
        print("Getting telegram data")

        # get data from telegram
        telegram_secrets = json.loads(self.source.secrets)

        # Set timmer to avoid scraping for too long
        time_limit = 60*45 # 45 min
        time_start = time.time()
        telegram_client = TelegramClient(
            StringSession(telegram_secrets["session-string"]),
            telegram_secrets["api-id"],
            telegram_secrets["api-hash"]
        )
        telegram_client.connect()
        print("Telegram client connected")

        df_messages = pd.DataFrame()
        df_member_counts = pd.DataFrame()
        for channel in self.channels:
            print(f"getting in telegram channel {channel}")
            try:
                channel_entity = telegram_client.get_entity(channel)
                channel_full_info = telegram_client(GetFullChannelRequest(channel=channel_entity))

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
                            print(f"getting replies for {message.id} failed: {e}")
                        time_duration = time.time() - time_start
                        if time_duration >= time_limit:
                            print(f"getting replies for {channel} stopped: timeout {time_duration} seconds")
                            break
                else:
                    df_member_counts = self._count_messages(df_member_counts, df_messages, channel_full_info)
                    print(f"finish telegram channel {channel}, sleep 10 seconds")
                    time.sleep(10)
                    continue
            except Exception as e:
                print(f"in getting in telegram channel {channel}: {e}")
                break

        telegram_client.disconnect()
        print("Telegram client disconnected")

        # Add index column
        # df_member_counts.reset_index(inplace=True)
        # df_member_counts['id'] = df_member_counts.index
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