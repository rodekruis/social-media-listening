import tweepy
import facebook
import pandas as pd
import requests
import os
import time
from google.oauth2 import service_account
import googleapiclient.discovery
from telethon import TelegramClient, events, sync
from telethon.sessions import StringSession
from telethon.tl.functions.channels import GetFullChannelRequest
from pipeline.utils import get_blob_service_client, get_secret_keyvault, arrange_facebook_replies, arrange_telegram_messages, save_data
import logging
import datetime
import yaml

# -*- coding: utf-8 -*-
try:
    import json
except ImportError:
    import simplejson as json


def get_twitter(config):
    logging.info('getting twitter data')

    # initialize twitter API
    twitter_secrets = get_secret_keyvault("twitter-secret", config)
    twitter_secrets = json.loads(twitter_secrets)
    auth = tweepy.OAuthHandler(twitter_secrets['CONSUMER_KEY'], twitter_secrets['CONSUMER_SECRET'])
    auth.set_access_token(twitter_secrets['ACCESS_TOKEN'], twitter_secrets['ACCESS_SECRET'])
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

    twitter_data_path = "./twitter"
    os.makedirs(twitter_data_path, exist_ok=True)

    # track individual twitter users
    if config["track-twitter-users"]:
        df_twitter_users_to_track = pd.read_csv('../config/tweets_to_track.csv')
        tw_users = df_twitter_users_to_track.dropna()['user_id'].tolist()
        if len(tw_users) == 0:
            raise ValueError("No twitter user specified")

        for userID in tw_users:
            # save output as
            save_file = twitter_data_path + '/tweets_' + userID + '.json'

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

            with open(save_file, 'a') as tf:
                for tweet in all_tweets:
                    try:
                        tf.write('\n')
                        json.dump(tweet._json, tf)
                    except Exception as e:
                        logging.warning("Some error occurred, skipping tweet:")
                        logging.warning(e)
                        pass

    # track specific queries
    if config["track-twitter-queries"]:
        save_file = twitter_data_path + '/tweets_queries.json'
        queries = config["twitter-queries"]
        if len(queries) == 0:
            raise ValueError("No twitter query specified")
        all_tweets = []
        # loop over queries and search
        for query in queries:
            n = 0
            try:
                for page in tweepy.Cursor(api.search,
                                          q=query,
                                          tweet_mode='extended',
                                          include_entities=True,
                                          max_results=100).pages():
                    # logging.info('processing page {0}'.format(n))
                    try:
                        for tweet in page:
                            all_tweets.append(tweet)
                    except Exception as e:
                        logging.warning("Some error occurred, skipping page {0}:".format(n))
                        logging.warning(e)
                        pass
                    n += 1
            except Exception as e:
                logging.warning("Some error occurred, skipping query {0}:".format(query))
                logging.warning(e)
                pass

        with open(save_file, 'a') as tf:
            for tweet in all_tweets:
                try:
                    tf.write('\n')
                    json.dump(tweet._json, tf)
                except Exception as e:
                    logging.warning("Some error occurred, skipping tweet:")
                    logging.warning(e)
                    pass


    # parse tweets and store in dataframe
    df_tweets = pd.DataFrame()
    for file in os.listdir(twitter_data_path):
        if file.endswith('.json'):
            df_tweets_ = pd.read_json(os.path.join(twitter_data_path, file), lines=True)
            df_tweets = df_tweets.append(df_tweets_, ignore_index=True)
    # drop duplicates
    df_tweets = df_tweets.drop_duplicates(subset=['id'])

    save_data("tweets", "twitter", df_tweets, "id", "TW", config)


def get_youtube(config):

    df_youtube_channels_to_track = pd.read_csv('../config/youtube_to_track.csv')
    channel_ids = df_youtube_channels_to_track.dropna()['channel_id'].tolist()
    if len(channel_ids) == 0:
        raise ValueError("No youtube channel specified")

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1" # Disable OAuthlib's HTTPS verification
    api_service_name = "youtube"
    api_version = "v3"
    service_account_info = get_secret_keyvault('google-secret', config)
    credentials = service_account.Credentials.from_service_account_info(json.loads(service_account_info))
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)

    df_videos = pd.DataFrame()

    for channel_id in channel_ids:
        request = youtube.search().list(
            part="snippet,id",
            maxResults=50,
            order='date',
            channelId=channel_id,
            type='video'
        )
        response = request.execute()
        for item in response['items']:
            title = item['snippet']['title']
            description = item['snippet']['description']
            videoId = item['id']['videoId']
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=videoId
            )
            response = request.execute()['items'][0]
            if 'viewCount' in response['statistics'].keys():
                viewCount = response['statistics']['viewCount']
            else:
                viewCount = None
            if 'likeCount' in response['statistics'].keys():
                likeCount = response['statistics']['likeCount']
            else:
                likeCount = None
            if 'dislikeCount' in response['statistics'].keys():
                dislikeCount = response['statistics']['dislikeCount']
            else:
                dislikeCount = None
            if 'commentCount' in response['statistics'].keys():
                commentCount = response['statistics']['commentCount']
            else:
                commentCount = None
            publishedAt = response['snippet']['publishedAt']
            source = response['snippet']['channelTitle']
            url = f"https://www.youtube.com/watch?v={videoId}"
            df_videos = df_videos.append(pd.Series({
                'full_text': title,
                'description': description,
                'id': videoId,
                'source': source,
                'viewCount': viewCount,
                'likeCount': likeCount,
                'dislikeCount': dislikeCount,
                'commentCount': commentCount,
                'created_at': publishedAt,
                'url': url,
                'lang': 'unknown'
            }), ignore_index=True)

    save_data("videos", "youtube", df_videos, "id", "YT", config)


def get_kobo(config):
    # get data from kobo
    kobo_secrets = get_secret_keyvault('kobo-secret', config)
    kobo_secrets = json.loads(kobo_secrets)
    headers = {'Authorization': f'Token {kobo_secrets["token"]}'}
    data_request = requests.get(f'https://kobonew.ifrc.org/api/v2/assets/{kobo_secrets["asset"]}/data.json',
                                headers=headers)
    data = data_request.json()['results']
    df_form = pd.DataFrame(data)

    save_data("form_data", "kobo", df_form, "_id", "Kobo", config)


def get_facebook(config):

    end_date = datetime.datetime.today().date()

    # get data from facebook
    facebook_secrets = get_secret_keyvault('facebook-secret', config)
    facebook_secrets = json.loads(facebook_secrets)
    
    graph = facebook.GraphAPI(
        access_token=facebook_secrets["token"])#,
        #version="3.1")
    pages = [{"id": 612726123570520, "name": "fb/Hello"}, \
        {"id": 415050737211518, "name": "fb/Test API"}]
    reaction_types = ['LIKE', 'LOVE', 'WOW', 'HAHA', 'SORRY', 'ANGRY', \
        'THANKFUL', 'PRIDE', 'CARE', 'FIRE', 'HUNDRED']

    # get all comments to posts
    df_pages = pd.DataFrame()

    for page in pages:
        df_posts = pd.DataFrame()
        page_posts = graph.get_object(id=page['id'], fields="feed")['feed']
        while True:
            for post in page_posts['data']:
                comment = {}
                # # use the line below to get reactions info
                # stats = graph.get_object(id=post["id"], fields="message,shares,reactions.summary(true)")
                stats_to_save = arrange_facebook_replies(post, comment, page['name'])
                df_posts = df_posts.append(pd.Series(stats_to_save), ignore_index=True)

                df_comments = pd.DataFrame()
                comments = {'data': []}
                # TODO: add time sleep as in drought get ENSO
                try:
                    comments = graph.get_object(id=post["id"], fields="comments")['comments']
                except KeyError:
                    continue

                while True:
                    for comment in comments['data']:
                        stats_to_save = arrange_facebook_replies(post, comment, page['name'])
                        df_comments = df_comments.append(pd.Series(stats_to_save), ignore_index=True)

                        comments = {'data': []}
                        try:
                            comments = graph.get_object(id=comment["id"], fields="comments")['comments']
                        except KeyError:
                            continue

                        while True:
                            for comment in comments['data']:
                                stats_to_save = arrange_facebook_replies(post, comment, page['name'])
                                df_comments = df_comments.append(pd.Series(stats_to_save), ignore_index=True)
                                
                            try:
                                comments = requests.get(comments["paging"]["next"]).json()
                            except KeyError:
                                break

                    try:
                        comments = requests.get(comments["paging"]["next"]).json()
                    except KeyError:
                        break
                
                df_posts = df_posts.append(df_comments, ignore_index=True)
                
            try:
                page_posts = requests.get(page_posts["paging"]["next"]).json()
            except KeyError:
                break

        df_pages = df_pages.append(df_posts, ignore_index=True)
    
    df_pages.reset_index(inplace=True)
    df_pages['id'] = df_pages.index

    save_data(f"{config['country-code']}_FB_posts_{end_date}",
        "facebook", 
        df_pages, 
        "id",
        "FB",
        config)


def get_telegram(config, days):

    logging.info("Getting telegram data")

    end_date = datetime.datetime.today().date()
    start_date = end_date - pd.Timedelta(days=days)

    # load channel links
    blob_client = get_blob_service_client(config["telegram-channels-file"], config)
    with open(config["telegram-channels-file"], "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    with open(config["telegram-channels-file"], "r") as file:
        channels_file = yaml.load(file, Loader=yaml.FullLoader)
    telegram_channels = channels_file["telegram-channels"]

    #get data from telegram
    telegram_secrets = get_secret_keyvault("telegram-secret", config)
    telegram_secrets = json.loads(telegram_secrets)

    telegram_client = TelegramClient(
        StringSession(telegram_secrets["session-string"]),#"rou-smm-bot",
        telegram_secrets["api-id"],
        telegram_secrets["api-hash"]
    )
    telegram_client.connect()  # start(bot_token=telegram_secrets["bot-token"])
    logging.info("Telegram client connected")

    df_messages = pd.DataFrame()
    df_member_counts = pd.DataFrame()
    for channel in telegram_channels:
        logging.info(f"getting in telegram channel {channel}")
        try:
            channel_entity = telegram_client.get_entity(channel)
            channel_full_info = telegram_client(GetFullChannelRequest(channel=channel_entity))

            for message in telegram_client.iter_messages(
                channel_entity,
                offset_date=start_date,
                reverse=True,
                wait_time = 5
            ):
                reply = None
                df_messages = arrange_telegram_messages(df_messages, message, reply, channel)
                if channel_entity.broadcast and message.post and message.replies:
                    df_replies = pd.DataFrame()
                    for reply in telegram_client.iter_messages(
                        channel_entity,
                        reply_to=message.id,
                        wait_time = 2
                    ):
                        df_replies = arrange_telegram_messages(df_replies, message, reply, channel)
                        time.sleep(5)
                    df_messages = df_messages.append(df_replies, ignore_index=True)

            idx = len(df_member_counts)
            df_member_counts.at[idx, 'source'] = channel
            member_count = channel_full_info.full_chat.participants_count
            df_member_counts.at[idx, 'member_count'] = member_count
            df_member_counts.at[idx, 'date'] = end_date
            df_member_counts.at[idx, 'message_count'] = len(df_messages[df_messages['source']==channel])
            
            logging.info(f"finish telegram channel {channel}, sleep 10 seconds")
            time.sleep(10)
        except Exception as e:
            logging.error(f"in getting in telegram channel {channel}: {e}")

    telegram_client.disconnect()
    logging.info("Telegram client disconnected")

    # Add index column
    df_member_counts.reset_index(inplace=True)
    df_member_counts['id'] = df_member_counts.index
    df_messages.reset_index(inplace=True)
    df_messages['id'] = df_messages.index
    df_messages['date'] = pd.to_datetime(df_messages['datetime']).dt.strftime('%Y-%m-%d')

    logging.info("Saving Telegram data")
    save_data(f"{config['country-code']}_TL_messages_{start_date}_{end_date}",
              "telegram",
              df_messages,
              "id",
              "TL",
              config)
    save_data(f"{config['country-code']}_TL_membercount_{start_date}_{end_date}",
              "telegram",
              df_member_counts,
              "id",
              "TL",
              config)
