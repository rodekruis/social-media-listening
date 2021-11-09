import os
import ast
import pandas as pd
import numpy as np
from pipeline.utils import clean_text, translate_dataframe, geolocate_dataframe, filter_by_keywords, \
    get_blob_service_client, html_decode, predict_topic, predict_sentiment, save_data
import logging


def get_retweet(row_):
    if row_['retweeted']:
        if not pd.isna(row_['retweeted_status']):
            return row_['retweeted_status']['full_text']
        else:
            return row_['full_text']
    else:
        return row_['full_text']


def get_url_from_entities(entities):
    try:
        return entities['urls'][0]['expanded_url']
    except:
        return np.nan


def get_url_from_tweet(row):
    return f"https://twitter.com/{row['screen_name']}/status/{row['id']}"


def parse_kobo(config):

    # load and parse tweets
    kobo_data_path = "./kobo"
    form_data_path = kobo_data_path + "/form_data_latest.csv"
    df_kobo = pd.read_csv(form_data_path)
    next_text_value = config["text-field-kobo-form"]
    if next_text_value not in df_kobo.columns:
        raise ValueError("text-field-kobo-form not in kobo form data, check config")

    # translate tweets
    if config["translate"]:
        df_kobo = translate_dataframe(df_kobo, next_text_value, 'full_text_en', config)
        next_text_value = 'full_text_en'

    # filter by keywords
    if config["filter-by-keywords"]:
        df_keywords = pd.read_csv('../config/keywords.csv')
        keywords = df_keywords.dropna()['keyword'].tolist()
        df_kobo = filter_by_keywords(df_kobo, [next_text_value], keywords)

    # geolocate
    if config["geolocate"]:
        df_kobo = geolocate_dataframe(df_kobo,
                                      config['geodata-locations'],
                                      config['geodata-country-boundaries'],
                                      config['location-input'],
                                      config['location-output'],
                                      [next_text_value],
                                      config,
                                      'place')

    # sentiment analysis
    if config["analyse-sentiment"]:
        df_kobo = predict_sentiment(df_kobo, next_text_value, config)

    # topic analysis
    if config["analyse-topic"]:
        df_kobo = predict_topic(df_kobo, next_text_value, config)

    save_data("form_data_processed", "kobo", df_kobo, "_id", config)
    return "./kobo/form_data_processed_all.csv"


def parse_twitter(config):

    # load and parse tweets
    twitter_data_path = "./twitter"
    tweets_path = twitter_data_path + "/tweets_latest.csv"
    df_tweets = pd.read_csv(tweets_path)
    df_tweets['user'] = df_tweets['user'].astype(str).apply(ast.literal_eval)
    df_tweets['source'] = df_tweets['user'].apply(lambda x: x['name'])
    df_tweets['screen_name'] = df_tweets['user'].apply(lambda x: x['screen_name'])
    df_tweets['entities'] = df_tweets['entities'].astype(str).apply(ast.literal_eval)
    df_tweets['url'] = df_tweets['entities'].apply(get_url_from_entities)
    df_tweets['twitter_url'] = df_tweets.apply(get_url_from_tweet, axis=1)
    df_tweets['url'] = df_tweets['url'].fillna(df_tweets['twitter_url'])
    df_tweets = df_tweets.drop(columns={'entities', 'screen_name', 'twitter_url', 'user', 'extended_entities'})
    next_text_value = 'full_text'

    # if retweet, get full text of original tweet
    df_tweets[next_text_value] = df_tweets.apply(get_retweet, axis=1)
    # clean text
    df_tweets[next_text_value] = df_tweets.apply(clean_text, args=(next_text_value,), axis=1)

    # translate tweets
    if config["translate"]:
        df_tweets = translate_dataframe(df_tweets, next_text_value, 'full_text_en', config)
        next_text_value = 'full_text_en'

    # filter by keywords
    if config["filter-by-keywords"]:
        df_keywords = pd.read_csv('../config/keywords.csv')
        keywords = df_keywords.dropna()['keyword'].tolist()
        df_tweets = filter_by_keywords(df_tweets, [next_text_value], keywords)

    # geolocate
    if config["geolocate"]:
        df_tweets = geolocate_dataframe(df_tweets,
                                        config['geodata-locations'],
                                        config['geodata-country-boundaries'],
                                        config['location-input'],
                                        config['location-output'],
                                        [next_text_value],
                                        config,
                                        'place')

    # sentiment analysis
    if config["analyse-sentiment"]:
        df_tweets = predict_sentiment(df_tweets, next_text_value, config)

    # topic analysis
    if config["analyse-topic"]:
        df_tweets = predict_topic(df_tweets, next_text_value, config)

    # remove unnecessary data
    for col in ['truncated', 'display_text_range', 'entities', 'extended_entities',
                'metadata', 'in_reply_to_status_id', 'in_reply_to_status_id_str',
                'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name',
                'user', 'geo', 'coordinates', 'contributors', 'is_quote_status',
                'possibly_sensitive', 'place', 'retweet_count', 'favorite_count',
                'retweeted_status', 'quoted_status_id', 'quoted_status_id_str', 'quoted_status',
                'favorited', 'retweeted', 'coord']:
        if col in df_tweets.columns:
            df_tweets = df_tweets.drop(columns=[col])

    save_data("tweets_processed", "twitter", df_tweets, "id", config)
    return "./twitter/tweets_processed_all.csv"


def parse_youtube(config):

    # load and parse youtube
    youtube_data_path = "./youtube"
    videos_path = youtube_data_path + "/videos_latest.csv"
    df_videos = pd.read_csv(videos_path)
    next_text_value = 'full_text'

    df_videos[next_text_value] = df_videos.apply(html_decode, args=(next_text_value,), axis=1)

    # translate videos title
    if config["translate"]:
        df_videos = translate_dataframe(df_videos, next_text_value, 'full_text_en', config)
        next_text_value = 'full_text_en'

    # filter by keywords
    if config["filter-by-keywords"]:
        df_keywords = pd.read_csv('../config/keywords.csv')
        keywords = df_keywords.dropna()['keyword'].tolist()
        df_videos = filter_by_keywords(df_videos, [next_text_value], keywords)

    if config["geolocate"]:
        df_videos = geolocate_dataframe(df_videos,
                                        config['geodata-locations'],
                                        config['geodata-country-boundaries'],
                                        config['location-input'],
                                        config['location-output'],
                                        [next_text_value],
                                        config)

    save_data("videos_processed", "youtube", df_videos, "id", config)
    return "./youtube/videos_processed_all.csv"


def merge_sources(data_to_merge, config):

    # merge everything
    merged_data_path = "./merged"
    os.makedirs(merged_data_path, exist_ok=True)
    merged_path = merged_data_path + "/merged_latest.csv"

    if len(data_to_merge) == 0:
        raise ValueError("No datasets to merge")
    elif len(data_to_merge) == 1:
        df_merged = pd.read_csv(data_to_merge[0])
        df_merged.to_csv(merged_path, index=False)
    else:
        df_merged = pd.read_csv(data_to_merge[0])
        for data in data_to_merge[1:]:
            df_merged = df_merged.append(pd.read_csv(data), ignore_index=True)
        df_merged.to_csv(merged_path, index=False)

    # upload to datalake
    if not config['skip-datalake']:
        blob_client = get_blob_service_client('merged/merged_all.csv', config)
        with open(merged_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)


def prepare_final_dataset(blob_service_client):

    logging.info('prepare_final_dataset: merging')
    in_file = './tweets/tweets_latest_topic.csv'
    if not os.path.exists(in_file):
        logging.error("Error: no processed tweets found")
    df_tweets = pd.read_csv(in_file)

    # datetime to date
    df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'])
    df_tweets['created_at'] = df_tweets['created_at'].dt.date
    df_tweets['full_text_clean'] = df_tweets['full_text_clean'].str.replace(': ', '')
    df_tweets['full_text_en'] = df_tweets['full_text_en'].str.replace(': ', '')
    df_tweets = df_tweets[(~df_tweets['full_text_clean'].str.contains('#NAME?')) & (~df_tweets['full_text_en'].str.contains('#NAME?'))]

    if len(df_tweets) > df_tweets.id.nunique():
        logging.info('re-assigning id')
        for ix, row in df_tweets.iterrows():
            ran_id = random.randint(1, 1E18)
            if ran_id not in df_tweets.id.unique():
                df_tweets.at[ix, 'id'] = ran_id
        logging.info(f"{len(df_tweets)}, {df_tweets.id.nunique()}")

    # save locally
    os.makedirs('./powerbi', exist_ok=True)
    out_file = './powerbi/powerbi_latest.xlsx'
    df_tweets.to_excel(out_file, index=False, encoding='utf8')

    # merge with existing datasets
    blob_client = blob_service_client.get_blob_client(container='covid-nam-rumor-tracker',
                                                      blob='powerbi/powerbi_latest.xlsx')
    try:
        with open('./powerbi/powerbi_old.xlsx', "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        df_old = pd.read_excel('./powerbi/powerbi_old.xlsx')
        if 'Unnamed: 0' in df_old.columns:
            df_old = df_old.drop(columns=['Unnamed: 0'])
        res = pd.concat([df_tweets, df_old])
        res = res.drop_duplicates(subset=['id'])
        res.to_excel('./powerbi/powerbi_merged.xlsx', index=False)
    except:
        df_tweets.to_excel('./powerbi/powerbi_merged.xlsx', index=False, encoding='utf8')

    # upload processed tweets
    with open('./powerbi/powerbi_merged.xlsx', "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    # upload a timestamped copy for future reference
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=7)
    reference_filename = 'powerbi_' + today.strftime("%d-%m-%Y") + '_' + week_ago.strftime("%d-%m-%Y") + '.xlsx'
    blob_client = blob_service_client.get_blob_client(container='covid-nam-rumor-tracker',
                                                      blob='powerbi/'+reference_filename)
    with open('./powerbi/powerbi_latest.xlsx', "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

