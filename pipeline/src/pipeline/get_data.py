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
from pipeline.utils import get_blob_service_client

# -*- coding: utf-8 -*-
try:
    import json
except ImportError:
    import simplejson as json


def get_twitter(tw_users, skip_datalake):

    # initialize twitter API
    with open("../credentials/twitter_secrets.json") as file:
        twitter_secrets = json.load(file)
    auth = tweepy.OAuthHandler(twitter_secrets['CONSUMER_KEY'], twitter_secrets['CONSUMER_SECRET'])
    auth.set_access_token(twitter_secrets['ACCESS_TOKEN'], twitter_secrets['ACCESS_SECRET'])
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

    twitter_data_path = "./twitter"
    os.makedirs(twitter_data_path, exist_ok=True)

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

        for tweet in all_tweets:
            try:
                with open(save_file, 'a') as tf:
                    tf.write('\n')
                    json.dump(tweet._json, tf)
            except tweepy.TweepError:
                raise

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
    if not skip_datalake:
        blob_client = get_blob_service_client('twitter/tweets_latest.csv')
        with open(tweets_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    # append to existing twitter dataframe
    tweets_all_path = twitter_data_path + "/tweets_all.csv"
    try:
        if not skip_datalake:
            blob_client = get_blob_service_client('twitter/tweets_all.csv')
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
    if not skip_datalake:
        with open(tweets_all_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)


def get_youtube(channel_ids, skip_datalake):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
    api_service_name = "youtube"
    api_version = "v3"
    service_account_info = "../credentials/google_service_account_secrets.json"

    # Get credentials and create an API client
    credentials = service_account.Credentials.from_service_account_file(service_account_info)
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
            viewCount = response['statistics']['viewCount']
            likeCount = response['statistics']['likeCount']
            dislikeCount = response['statistics']['dislikeCount']
            commentCount = response['statistics']['commentCount']
            publishedAt = response['snippet']['publishedAt']
            source = response['snippet']['channelTitle']
            url = f"https://www.youtube.com/watch?v={videoId}"
            df_videos = df_videos.append(pd.Series({
                'title': title,
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
    if not skip_datalake:
        blob_client = get_blob_service_client('youtube/videos_latest.csv')
        with open(videos_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    # append to existing twitter dataframe
    videos_all_path = youtube_data_path + "/videos_all.csv"
    try:
        if not skip_datalake:
            blob_client = get_blob_service_client('youtube/videos_all.csv')
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
    if not skip_datalake:
        with open(videos_all_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)



