import os
import ast
import pandas as pd
import numpy as np
from pipeline.utils import clean_text, translate_dataframe, geolocate_dataframe, \
    filter_by_keywords, get_blob_service_client, html_decode, predict_topic, \
    predict_sentiment, save_data, get_word_frequency, get_daily_messages, \
    previous_weekday, read_db, remove_pii
import logging
import datetime
import random


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


def parse_azure_table(df, config):
    next_text_value = config["text-field-azure-table"]

    if type(next_text_value) != str and len(next_text_value) > 0:
        col0 = next_text_value[0]
        df[col0] = df[col0].replace(r'^\s*$', np.nan, regex=True)
        if col0 not in df.columns:
            raise ValueError(f"text-field-azure-table {col0} not in data, check config")
        for col in next_text_value[1:]:
            if col not in df.columns:
                raise ValueError(f"text-field-azure-table {col} not in data, check config")
            df[col0] = df[col0].fillna(df[col])
        next_text_value = col0

    # translate tweets
    if config["translate"]:
        df = translate_dataframe(df, next_text_value, 'full_text_en', config)
        next_text_value = 'full_text_en'

    # filter by keywords
    if config["filter-by-keywords"]:
        df_keywords = pd.read_csv('../config/keywords.csv')
        keywords = df_keywords.dropna()['keyword'].tolist()
        df = filter_by_keywords(df, [next_text_value], keywords)

    # geolocate
    if config["geolocate"]:
        df = geolocate_dataframe(df,
                                 config['geodata-locations'],
                                 config['geodata-country-boundaries'],
                                 config['location-input'],
                                 config['location-output'],
                                 [next_text_value],
                                 config)

    # sentiment analysis
    if config["analyse-sentiment"]:
        df = predict_sentiment(df, next_text_value, config)

    # topic analysis
    if config["analyse-topic"]:
        df = predict_topic(df, next_text_value, config)

    return df


def parse_facebook(config):
    # load and parse facebook comments
    data_dir = "./facebook"
    data_path = data_dir + "/facebook_comments_latest.csv"
    df_fb = pd.read_csv(data_path)
    next_text_value = "message"
    if next_text_value not in df_fb.columns:
        raise ValueError(f"{next_text_value} field not in facebook data")

    # translate tweets
    if config["translate"]:
        df_fb = translate_dataframe(df_fb, next_text_value, 'full_text_en', config)
        next_text_value = 'full_text_en'

    # filter by keywords
    if config["filter-by-keywords"]:
        df_keywords = pd.read_csv('../config/keywords.csv')
        keywords = df_keywords.dropna()['keyword'].tolist()
        df_fb = filter_by_keywords(df_fb, [next_text_value], keywords)

    # geolocate
    if config["geolocate"]:
        df_fb = geolocate_dataframe(df_fb,
                                    config['geodata-locations'],
                                    config['geodata-country-boundaries'],
                                    config['location-input'],
                                    config['location-output'],
                                    [next_text_value],
                                    config,
                                    'place')

    # sentiment analysis
    if config["analyse-sentiment"]:
        df_fb = predict_sentiment(df_fb, next_text_value, config)

    # topic analysis
    if config["analyse-topic"]:
        df_fb = predict_topic(df_fb, next_text_value, config)

    save_data("facebook_comments_processed", "facebook", df_fb, "id_comment", "FB", config)
    return "./facebook/facebook_comments_processed_all.csv"


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

    save_data("form_data_processed", "kobo", df_kobo, "_id", "Kobo", config)
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

    save_data("tweets_processed", "twitter", df_tweets, "id", "TW", config)
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

    save_data("videos_processed", "youtube", df_videos, "id", "YT", config)
    return "./youtube/videos_processed_all.csv"


def parse_telegram(config):
    start_date = os.environ["START_DATE"]
    end_date = os.environ["END_DATE"]

    telegram_data_path = "./telegram"
    sm_code = "TL"

    multiple_files = False  # True/ False  select True if using daily data instead of bi-weekly data

    # load telegram data
    if config['track-azure-database']:
        df_messages = read_db(sm_code, start_date, end_date, config)
    else:
        if multiple_files:
            df_messages = get_daily_messages(start_date, end_date, telegram_data_path, config)
        else:
            messages_filename = f"/{config['country-code']}_{sm_code}_messages_{start_date}_{end_date}_latest.csv"
            messages_path = telegram_data_path + messages_filename
            df_messages = pd.read_csv(messages_path)

    # Combine text of post and replies
    df_messages['text_post'] = df_messages['text_post'].fillna("")
    df_messages['text_reply'] = df_messages['text_reply'].fillna("")

    # Column that  takes the post text if message is post, else the reply text (needed for word frequencies)
    df_messages['text_combined'] = np.where(
        df_messages['post'],
        df_messages['text_post'],
        df_messages['text_reply']
    )
    # Column that merges the post and reply text (needed for filtering on keywords)
    df_messages['text_merged'] = df_messages['text_post'] + " " + df_messages['text_reply']

    # get distribution of wordss
    if config["get-word-freq"]:
        get_word_frequency(df_messages, 'text_combined', sm_code, start_date, end_date, config)

    if config["filter-by-keywords"]:
        keyword_files = config["keyword-files"]

        for file in keyword_files:
            topic = file.split("_")[0]

            df_keywords = pd.read_csv(f"../config/{file}")
            keywords = df_keywords.dropna()['keyword'].tolist()
            df_messages = filter_by_keywords(df_messages, ['text_merged'], keywords, topic)

    # Remove combined and merged text columns
    df_messages = df_messages.drop(columns=['text_combined', 'text_merged'])

    # translate telegram messages
    if config["translate"]:
        keyword_files = config["keyword-files"]

        topics = []
        for file in keyword_files:
            topic = file.split("_")[0]
            topics.append(topic)

        df_to_translate = df_messages.copy()

        # removing messages which don't belong to any topic
        for ix, row in df_to_translate.iterrows():
            if all(row[topic] is False for topic in topics):
                df_to_translate.drop(ix, inplace=True)

        df_messages = translate_dataframe(
            df_to_translate,
            ['text_post', 'text_reply'],
            ['text_post_en', 'text_reply_en'],
            config
        )

        df_messages['text_post_en'] = df_messages['text_post_en'].fillna(method='ffill')

    # # sentiment analysis
    # if config["analyse-sentiment"]:
    #     df_messages = predict_sentiment(df_messages, next_text_value, config)

    # topic analysis
    if config["analyse-topic"]:
        # Create combined translated text column to train model on
        df_messages['text_combined_en'] = np.where(
            df_messages['post'],
            df_messages['text_post_en'],
            df_messages['text_reply_en']
        )

        # analyse topic RED CROSS and CVA
        df_messages_1 = df_messages[
            (df_messages["rcrc"]) &
            (df_messages["cva"])
            ]
        df_messages_1 = predict_topic(df_messages_1, 'text_combined_en', sm_code, start_date, end_date, config, "cva")
        # analyse topic RED CROSS and not CVA
        df_messages_2 = df_messages[(df_messages["rcrc"]) & (~df_messages["cva"])]
        df_messages_2 = predict_topic(df_messages_2, 'text_combined_en', sm_code, start_date, end_date, config, "rcrc")
        # merge all into one single df
        df_messages_topics = df_messages_1.append(df_messages_2, ignore_index=True)

        # Drop combined translated text column
        df_messages_topics = df_messages_topics.drop(columns=['text_combined_en'])
        df_messages = df_messages.drop(columns=['text_combined_en'])

        # assign topics to messages that were not clustered
        df_messages.drop(df_messages[df_messages['rcrc']].index, inplace=True)

        df_messages['topic'] = ""
        df_messages = df_messages.append(df_messages_topics, ignore_index=True)
        df_messages.drop(df_messages[(df_messages['rcrc']) & (df_messages['topic'] == "")].index, inplace=True)

    # remove personally identifiable information
    if config["remove-pii"]:
        df_messages = remove_pii(df_messages, ['text_post_en', 'text_reply_en'])

    save_data(f"{config['country-code']}_{sm_code}_messagesprocessed_{start_date}_{end_date}",
              "telegram",
              df_messages,
              "id",
              sm_code,
              config)

    return f"./telegram/{config['country-code']}_{sm_code}_messagesprocessed_all.csv"


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
    df_tweets = df_tweets[
        (~df_tweets['full_text_clean'].str.contains('#NAME?')) & (~df_tweets['full_text_en'].str.contains('#NAME?'))]

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
                                                      blob='powerbi/' + reference_filename)
    with open('./powerbi/powerbi_latest.xlsx', "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
