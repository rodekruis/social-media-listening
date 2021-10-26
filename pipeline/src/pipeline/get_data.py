import datetime
from azure.storage.blob import BlobServiceClient, BlobClient
import tweepy
import pandas as pd
from requests.exceptions import Timeout, ConnectionError
from requests.packages.urllib3.exceptions import ReadTimeoutError
import ssl
import time
import os
from google.oauth2 import service_account
import googleapiclient.discovery
from pipeline.utils import get_blob_service_client, get_secret_keyvault
import logging

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

    # save
    tweets_path = twitter_data_path + "/tweets_latest.csv"
    df_tweets = df_tweets
    df_tweets.to_csv(tweets_path, index=False)

    # upload to datalake
    if not config["skip-datalake"]:
        blob_client = get_blob_service_client('twitter/tweets_latest.csv', config)
        with open(tweets_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    # append to existing twitter dataframe
    tweets_all_path = twitter_data_path + "/tweets_all.csv"
    try:
        if not config["skip-datalake"]:
            blob_client = get_blob_service_client('twitter/tweets_all.csv', config)
            with open(tweets_all_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        df_old_tweets = pd.read_csv(tweets_all_path, lines=True)
        df_all_tweets = df_old_tweets.append(df_tweets, ignore_index=True)
    except:
        df_all_tweets = df_tweets.copy()

    # drop duplicates and save
    df_all_tweets = df_all_tweets.drop_duplicates(subset=['id'])
    df_all_tweets.to_csv(tweets_all_path, index=False)

    # upload to datalake
    if not config["skip-datalake"]:
        blob_client = get_blob_service_client('twitter/tweets_all.csv', config)
        with open(tweets_all_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)


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

    # save
    youtube_data_path = "./youtube"
    os.makedirs(youtube_data_path, exist_ok=True)
    videos_path = youtube_data_path + "/videos_latest.csv"
    df_videos = df_videos
    df_videos.to_csv(videos_path, index=False)

    # upload to datalake
    if not config["skip-datalake"]:
        blob_client = get_blob_service_client('youtube/videos_latest.csv', config)
        with open(videos_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    # append to existing twitter dataframe
    videos_all_path = youtube_data_path + "/videos_all.csv"
    try:
        if not config["skip-datalake"]:
            blob_client = get_blob_service_client('youtube/videos_all.csv', config)
            with open(videos_all_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        df_old_videos = pd.read_csv(videos_all_path, lines=True)
        df_all_videos = df_old_videos.append(df_videos, ignore_index=True)
    except:
        df_all_videos = df_videos.copy()

    # drop duplicates and save
    df_all_videos = df_all_videos.drop_duplicates(subset=['id'])
    df_all_videos.to_csv(videos_all_path, index=False)

    # upload to datalake
    if not config["skip-datalake"]:
        blob_client = get_blob_service_client('youtube/videos_all.csv', config)
        with open(videos_all_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)



