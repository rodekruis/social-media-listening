from cgitb import text
import preprocessor as tp
import pandas as pd
import numpy as np
import os
import operator
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector
import stopwordsiso
import gensim
from google.cloud import language_v1
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from time import sleep
from requests.exceptions import ReadTimeout, ConnectionError
import requests, uuid, json
from shapely.geometry import Polygon, Point
import geopandas as gpd
import json
import enchant
import transformers
import pyodbc
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
np.random.seed(2018)
import nltk
nltk.download('wordnet')
import pickle
stemmer = PorterStemmer()
from spellchecker import SpellChecker
spell = SpellChecker()
from gensim.parsing.preprocessing import STOPWORDS
from pipeline.GSDMM import MovieGroupProcess
import ast
from azure.storage.blob import BlobServiceClient, PartialBatchErrorException
from azure.data.tables import TableServiceClient
from tqdm import tqdm
tqdm.pandas()
tp.set_options(tp.OPT.URL, tp.OPT.EMOJI, tp.OPT.MENTION)
en_dict = enchant.Dict("en_US")
nltk.download('punkt')
import logging
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import datetime
from urllib.error import HTTPError


def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)  # We use the seed 42


def get_secret_keyvault(secret_name, config):
    kv_url = config["keyvault-url"]
    # Authenticate with Azure
    az_credential = DefaultAzureCredential()
    # Retrieve primary key for blob from the Azure Keyvault
    kv_secretClient = SecretClient(vault_url=kv_url, credential=az_credential)
    secret_value = kv_secretClient.get_secret(config[secret_name]).value
    return secret_value


class BreakIt(Exception):
    pass


def preprocess(text):
    result = []
    token_list = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) > 2 and 'haha' not in token and token not in STOPWORDS:
            result.append(stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v')))
            token_list.append(token)
    return result, dict(zip(result, token_list))


def produce_mapping(mapping_list):
    mapping_pairs = pd.concat([pd.DataFrame([(k, v) for k, v in d.items()]) for d in mapping_list])
    mapping_pairs['count'] = 1
    mapping121 = mapping_pairs.groupby(by=[0, 1]).count().reset_index().sort_values(by=[0, 'count'],
                                                                                    ascending=False).groupby(by=0).head(1)
    mapping12many = mapping_pairs.drop(columns=['count']).drop_duplicates()
    return mapping121, mapping12many


def get_blob_service_client(blob_path, config):
    blobstorage_secrets = get_secret_keyvault('blobstorage-secret', config)
    blobstorage_secrets = json.loads(blobstorage_secrets)
    blob_service_client = BlobServiceClient.from_connection_string(blobstorage_secrets['connection_string'])
    container = blobstorage_secrets['container']
    return blob_service_client.get_blob_client(container=container, blob=blob_path)


def get_table_service_client(table, config):
    table_secret = get_secret_keyvault('table-secret', config)
    table_service_client = TableServiceClient.from_connection_string(table_secret)
    return table_service_client.get_table_client(table_name=table)


def match_location(x, gdf, target_column, loc_column, locations):
    """ if "coord" is empty do string matching, else find coord """
    sentence = x[target_column]
    coords = x['coord']
    if pd.isna(coords):
        if not pd.isna(sentence):
            loc_match = [loc for loc in locations if loc in sentence.lower()]
            # now taking the first match; TBI resolve ambiguities
            if len(loc_match) > 0:
                gdf_match = gdf[gdf[loc_column] == loc_match[0]]
                match_loc = gdf_match[loc_column].values[0]
                match_geo = gdf_match['geometry'].values[0]
                return match_geo, match_loc
            else:
                return np.nan, np.nan
        else:
            return np.nan, np.nan
    else:
        gdf_x = gpd.GeoDataFrame(pd.DataFrame(x).transpose(), geometry='coord', crs="EPSG:4326")
        gdf_x = gdf_x.drop(columns=[loc_column])
        res_union = gpd.overlay(gdf_x, gdf, how='intersection')
        if len(res_union) > 0:
            match_token = res_union[loc_column].values[0]
            match_geo = res_union.geometry.values[0]
            return match_geo, match_token
        else:
            return np.nan, np.nan


def extract_coordinates(x):
    """ extract coordinates from tweet's place field """
    if not pd.isna(x):
        x = ast.literal_eval(x)
        bbox = Polygon(x['bounding_box']['coordinates'][0])
        centroid = bbox.centroid.coords
        return Point(centroid)
    else:
        return np.nan


def extract_location(x):
    """ extract locations from tweet's place field """
    if not pd.isna(x):
        x = ast.literal_eval(x)
        location = x['name']
        return location
    else:
        return np.nan


def point_to_xy(point):
    """ extract x,y coordinates from shapely point """
    if not pd.isna(point):
        try:
            return point.x, point.y
        except:
            return point.centroid.x, point.centroid.y
    else:
        return np.nan, np.nan


def geolocate_dataframe(df_tweets, location_file, adm0_file,
                        location_input, location_output,
                        target, config, tw_place_column=""):
    logging.info("geolocating")

    # download geodata if not present
    if not os.path.exists(location_file):
        blob_client = get_blob_service_client('geodata/' + location_file, config)
        with open(location_file, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    if not os.path.exists(adm0_file):
        blob_client = get_blob_service_client('geodata/' + adm0_file, config)
        with open(adm0_file, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

    # select locations
    gdf = gpd.read_file(location_file, encoding='utf8')
    for loc_col in location_input:
        if loc_col not in gdf.columns:
            logging.warning(f"{loc_col} not in location file {location_file}, check config")
            continue
        gdf['is_english'] = gdf[loc_col].apply(en_dict.check)
        gdf = gdf[~gdf['is_english']].drop(columns=['is_english'])
        gdf[loc_col] = gdf[loc_col].str.lower()

    df_tweets[location_output] = np.nan

    # extract locations from "place" field (only for twitter)
    if tw_place_column in df_tweets.columns:
        df_tweets['coord'] = df_tweets[tw_place_column].apply(extract_coordinates)
    else:
        df_tweets['coord'] = np.nan

    # search for locations
    for loc_col in location_input:
        if loc_col not in gdf.columns:
            continue
        locations = [loc.lower() for loc in gdf[loc_col].values]
        for target_col in target:
            df_tweets['temp_coord'], df_tweets['temp_location'] = \
                zip(*df_tweets.progress_apply(lambda x: match_location(x, gdf, target_col, loc_col, locations), axis=1))
            df_tweets['coord'] = df_tweets['coord'].fillna(df_tweets['temp_coord'])
            df_tweets[location_output] = df_tweets[location_output].fillna(df_tweets['temp_location'])

    df_tweets = df_tweets.drop(columns=['temp_coord', 'temp_location'])
    count_geolocated = df_tweets.coord.count()

    # filter by country
    gdf_tweets = gpd.GeoDataFrame(df_tweets[~pd.isna(df_tweets.coord)], geometry='coord')
    gdf_country = gpd.read_file(adm0_file, encoding='utf8')
    gdf_tweets = gdf_tweets[gdf_tweets.geometry.within(gdf_country.geometry[0])]
    count_geolocated_filtered = gdf_tweets.coord.count()
    if count_geolocated_filtered < count_geolocated:
        df_tweets = df_tweets[df_tweets['id'].isin(gdf_tweets.id.tolist())]

    try:
        df_tweets['longitude'], df_tweets['latitude'] = zip(*df_tweets['coord'].apply(point_to_xy))
    except:
        df_tweets['longitude'], df_tweets['latitude'] = np.nan, np.nan
    df_tweets = df_tweets.drop(columns=['coord'])
    return df_tweets


def clean_text(row_, text_column):
    text_clean = tp.clean(row_[text_column]).lower().replace(': ', '')
    return text_clean


def html_decode(row_, text_column):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    """
    htmlCodes = (
        ("'", '&#39;'),
        ('"', '&quot;'),
        ('>', '&gt;'),
        ('<', '&lt;'),
        ('&', '&amp;')
    )
    text = row_[text_column]
    for code in htmlCodes:
        text = text.replace(code[1], code[0])
    return text


def translate_string(row_, translate_client, text_field, model):
    text = row_[text_field]
    if 'lang' in row_.keys():
        lang = row_['lang']
    else:
        lang = 'unknown'
    if lang != 'en':
        trans = None
        if model == "Google":
            try:
                response = translate_client.translate(text, target_language="en")
            except ReadTimeout or ConnectionError:
                sleep(60)
                try:
                    response = translate_client.translate(text, target_language="en")
                except ReadTimeout or ConnectionError:
                    sleep(60)
                    response = translate_client.translate(text, target_language="en")
            trans = response["translatedText"]
        elif "HuggingFace" in model:
            response = translate_client(text)[0]
            trans = response["translation_text"]
        elif model == "Microsoft":
            constructed_url = translate_client[0]
            params = translate_client[1]
            headers = translate_client[2]
            body = [{
                'text': text
            }]
            translation_done = False
            retry_times = 0
            while (not translation_done) and (retry_times <= 5):
                try:
                    request = requests.post(constructed_url, params=params, \
                        headers=headers, json=body)
                    response = request.json()
                    trans = response[0]['translations'][0]['text']
                    translation_done = True
                except ReadTimeout or ConnectionError as e:
                    retry_times += 1
                    sleep(60)
            if not translation_done:
                logging.warning(f"unable to translate {text}: {e}")

        if pd.isna(trans):
            return text
        else:
            return trans
    else:
        return text


def remove_pii(df, text_columns):
    """
    remove PII from dataframe
    """

    for ix, row in tqdm(df.iterrows(), total=len(df)):
        for text_column in text_columns:
            url = 'https://anonymization-app.azurewebsites.net/anonymize/'
            text_to_anonymize = row[text_column]
            if pd.isna(text_to_anonymize) or text_to_anonymize == "":
                df.at[ix, text_column] = text_to_anonymize
                # continue
            else:
                response = requests.post(url, json={"text": text_to_anonymize, "model": "ensemble"}).json()
                # print(response)
                if 'anonymized_text' in response.keys():
                    df.at[ix, text_column] = response['anonymized_text']
                    # else:
                    #     df.at[ix, text_column] = text_to_anonymize
                else:
                    logging.WARNING(f"Error with anonymization API: {response}")
                    print(text_to_anonymize)

    return df


def translate_dataframe(df_tweets, text_column, text_column_en, config, original_language=None):
    model = 'Google'  # default model
    if 'translation-model' in config.keys():
        model = config['translation-model']

    logging.info(f'translating with {model}')

    translate_client = None
    if model == 'Google':
        service_account_info = get_secret_keyvault('google-secret', config)
        credentials = service_account.Credentials.from_service_account_info(json.loads(service_account_info))
        translate_client = translate.Client(credentials=credentials)
    elif 'HuggingFace' in model:
        model_tag = model.replace("HuggingFace:", "")
        translate_client = transformers.pipeline("translation", model=model_tag)
    elif model == 'Microsoft':
        subcription_info = get_secret_keyvault('mscognitive-secret', config)
        subcription_info = json.loads(subcription_info)
        constructed_url = config['mscognitive-url']

        params = {
            'api-version': '3.0',
            'to': ['en'],
        }
        if original_language:
            params['from'] = [original_language]
        headers = {
            'Ocp-Apim-Subscription-Key': subcription_info["subscription_key"],
            'Ocp-Apim-Subscription-Region': subcription_info["location"],
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        translate_client = [constructed_url, params, headers]

    for idx, column in enumerate(text_column):
        # initialize empty column for translation in original dataframe
        df_tweets.at[text_column_en[idx]] = np.nan

        # initialize dataframe for translation
        df_tweets_nona = df_tweets.dropna(subset=[column])
        df_texts = df_tweets_nona.drop_duplicates(subset=[column])

        # translate to english
        df_texts[text_column_en[idx]] = df_texts.progress_apply(
            lambda x: translate_string(x, translate_client, column, model),
            axis=1)

        # copy back to original dataframe
        for ix, row in df_texts.iterrows():
            df_tweets.loc[df_tweets[column] == row[column], text_column_en[idx]] = row[text_column_en[idx]]

    # remove empty rows at the bottom
    df_tweets.dropna(subset=['id'], inplace=True)

    return df_tweets


def filter_by_keywords(df_tweets, text_columns, keywords, filter_name='is_conflict'):
    logging.info("Filtering by keywords")
    df_tweets[filter_name] = False

    for text_column in text_columns:
        df_tweets[text_column] = df_tweets[text_column].fillna("")

        df_tweets[filter_name] = df_tweets[text_column].apply(
            lambda x:
            True if any(len(word.split()) == 1 and
                        re.search(r"\b" + re.escape(word.lower()) + r"\b", x.lower())
                        for word in keywords)
            else (
                True if any(len(word.split()) >= 2 and
                            word.lower() in x.lower()
                            for word in keywords)
                else False
            )
        )

        # df_tweets['PII'] = df_tweets['text'].apply(lambda x: re.match(r'.\d\d\d\d\d+', x))

    # df_tweets = df_tweets[df_tweets['is_conflict']].drop(columns=['is_conflict'])
    logging.info("Done with filtering")
    return df_tweets


def get_word_frequency(df_tweets, text_column, sm_code, start_date, end_date, config):
    logging.info('Calculating word frequencies')

    # Get messages
    df_tweets[text_column] = df_tweets[text_column].astype(str)
    text = ''

    for msg_text in df_tweets[text_column]:
        text = text + " " + msg_text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace(",", " ")

    # Declare a dictionary
    dict_word_freq = {}

    # Split all the words of the string.
    text_list = text.split()

    # Lemmatize all the words of the string
    text_list_lemmatized = []
    language_detector = spacy.load("en_core_web_sm")
    Language.factory("language_detector", func=get_lang_detector)
    language_detector.add_pipe('language_detector', last=True)
    nlp_uk = spacy.load("uk_core_news_sm", disable=["tokenizer", "parser", "ner"])
    nlp_ru = spacy.load("ru_core_news_sm", disable=["tokenizer", "parser", "ner"])
    logging.info('Lemmatizing (it might take a while)')
    for text in tqdm(text_list):
        doc = language_detector(text)
        detected_language = doc._.language
        if 'language' in detected_language.keys():
            if detected_language['language'] == 'uk':
                lemmatized_text = " ".join([token.lemma_ for token in nlp_uk(text)])
            elif detected_language['language'] == 'ru':
                lemmatized_text = " ".join([token.lemma_ for token in nlp_ru(text)])
            else:
                lemmatized_text = text
            text_list_lemmatized.append(lemmatized_text)
    text_list = text_list_lemmatized.copy()

    # Remove stopwords and numbers
    stop_words = stopwordsiso.stopwords(['uk', 'ru'])
    text_list = [word for word in text_list if word not in stop_words and not word.isnumeric()]

    # Take each word from text_list and count occurence
    for element in text_list:
        # check if each word has '.' at its last. If so then ignore '.'
        if element[-1] == '.':
            element = element[0:len(element) - 1]

        # if there exists a key as "elements" then simply
        # increase its value.
        if element in dict_word_freq:
            dict_word_freq[element] += 1
        else:
            dict_word_freq.update({element: 1})

    # sort dict
    dict_word_freq = {
        key: value for key, value in sorted(
            dict_word_freq.items(),
            key=lambda item: item[1],
            reverse=True
        )
    }

    df_word_freq = pd.DataFrame.from_dict(dict_word_freq, orient='index')
    df_word_freq.reset_index(inplace=True)
    df_word_freq.columns = ['Word', 'Frequency']
    df_word_freq['id'] = df_word_freq.index

    # Delete everything with freq lower than 10
    threshold = 20
    if config['freq-threshold']:
        threshold = config['freq-threshold']

    df_word_freq = df_word_freq[df_word_freq['Frequency'] >= threshold]

    # Add translations
    df_word_freq = translate_dataframe(df_word_freq, ['Word'], ['Translation_Russian'], config, original_language='ru')
    df_word_freq = translate_dataframe(df_word_freq, ['Word'], ['Translation_Ukrainian'], config, original_language='uk')

    df_word_freq.drop(columns=['id'], inplace=True)

    word_freq_filename = f'{config["country-code"]}_{sm_code}_wordfrequencies_{start_date}_{end_date}.csv'
    word_freq_path = './word_frequencies'
    os.makedirs(word_freq_path, exist_ok=True)
    word_freq_filepath = os.path.join(word_freq_path, word_freq_filename)
    word_freq_blob_path = "word_frequencies"

    logging.info(f'Storing word frequencies at {word_freq_path}/{word_freq_filename}')
    df_word_freq.to_csv(word_freq_filepath, index=False)

    if not config["skip-datalake"]:
        logging.info(f'Uploading word frequencies for later use')
        blob_client = get_blob_service_client(os.path.join(word_freq_blob_path, word_freq_filename), config)
        with open(word_freq_filepath, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    return


def arrange_telegram_messages(df_messages, message, reply, channel):
    '''
    Arrange posts and their replies from Telegram channel
    '''
    ix = len(df_messages)
    df_messages.at[ix, "source"] = channel
    df_messages.at[ix, "id_post"] = message.id
    df_messages.at[ix, "text_post"] = str(message.text)
    if reply:
        df_messages.at[ix, "text_reply"] = str(reply.text)
        df_messages.at[ix, "datetime"] = reply.date
        df_messages.at[ix, "post"] = False
    else:
        df_messages.at[ix, "text_reply"] = reply
        df_messages.at[ix, "datetime"] = message.date
        df_messages.at[ix, "post"] = True
    return df_messages


def arrange_facebook_replies(post, comment, source):
    '''
    Arrange a post, their replies (and reactions, shares) 
    of a facebook post
    '''
    stats_to_save = {'id_post': post["id"]}
    stats_to_save['source'] = source
    # stats_to_save['id_comment'] = comment['id']
    if comment.keys():
        stats_to_save['datetime'] = comment['created_time']
        stats_to_save['text_post'] = post['message']
        stats_to_save['post'] = "False"
        if 'message' in comment.keys():
            stats_to_save['text_reply'] = comment['message']
        else:
            stats_to_save['text_reply'] = ""
    elif post.keys():
        stats_to_save['datetime'] = post['updated_time']
        stats_to_save['text_reply'] = ""
        stats_to_save['post'] = "True"
        if 'message' in post.keys():
            stats_to_save['text_post'] = post['message']
        else:
            stats_to_save['text_post'] = ""

    # # these commented lines used when shares and reactions needed
    # if 'shares' in stats.keys():
    #     stats_to_save['shares'] = stats['shares']['count']
    # if 'reactions' in stats.keys():
    #     stats_to_save['reaction_count'] = stats_comment['reactions']['summary']['total_count']
    #     if stats_comment['reactions']['summary']['viewer_reaction'] in reaction_types:
    #         for reaction in [stats_comment['reactions']['summary']['viewer_reaction']]:
    #             stats_to_save[reaction.lower()] = stats_comment['reactions']['summary']['total_count']
    # # better, use reactions.type(TYPE).summary(total_count) for TYPE=LIKE, LOVE, WOW, HAHA, SORRY, ANGRY
    return stats_to_save


def detect_sentiment(row, nlp_client, text_column, model="HuggingFace"):
    text = row[text_column]
    if pd.isna(text):
        return np.nan, np.nan
    else:
        if model == "Google":
            TYPE_ = language_v1.Document.Type.PLAIN_TEXT
            ENCODING_ = language_v1.EncodingType.UTF8
            document = {"content": text, "type_": TYPE_, "language": "en"}
            response = nlp_client.analyze_sentiment(request={'document': document, 'encoding_type': ENCODING_})
            return response.document_sentiment.score, response.document_sentiment.magnitude
        elif "HuggingFace" in model:
            response = nlp_client(text, return_all_scores=True)[0]
            weights = []
            if len(response) == 2:
                weights = [-1, 1]
            elif len(response) == 3:
                weights = [-1, 0, 1]
            score, maxscore = 0, 0
            for ix, label in enumerate(response):
                score += label['score'] * weights[ix]
                if label['score'] > maxscore:
                    maxscore = label['score']
            return score, maxscore


def predict_sentiment(df_tweets, text_column, config):
    model = 'HuggingFace'  # default model
    if 'sentiment-model' in config.keys():
        model = config['sentiment-model']

    logging.info(f'predicting sentiment with {model}')

    nlp_client = None
    if model == 'Google':
        service_account_info = get_secret_keyvault('google-secret', config)
        credentials = service_account.Credentials.from_service_account_info(json.loads(service_account_info))
        nlp_client = language_v1.LanguageServiceClient(credentials=credentials)
    elif 'HuggingFace' in model:
        model_tag = model.replace("HuggingFace:", "")
        nlp_client = transformers.pipeline('sentiment-analysis', model=model_tag)

    df_texts = df_tweets.drop_duplicates(subset=[text_column])

    # detect sentiment
    df_texts['sentiment_score'], df_texts['sentiment_magnitude'] = \
        zip(*df_texts.progress_apply(lambda x: detect_sentiment(x, nlp_client, text_column, model), axis=1))

    for ix, row in df_tweets.iterrows():
        df_texts_ = df_texts[df_texts['id'] == row['id']]
        if len(df_texts_) > 0:
            df_tweets.at[ix, 'sentiment_score'] = df_texts_['sentiment_score'].values[0]
            df_tweets.at[ix, 'sentiment_magnitude'] = df_texts_['sentiment_magnitude'].values[0]

    return df_tweets


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


def keywords_to_topic(df, df_topics):
    """assign a description ('theme') to each topic based on keywords"""
    for ix, row in df.iterrows():
        df_topics_ = df_topics[df_topics['topic number'] == row['topic number']]
        df.at[ix, 'topic'] = df_topics_['topic'].values[0]
    return df


def predict_topic(df_tweets, text_column, sm_code, start_date, end_date, config, filter_name='all'):
    logging.info('predicting topic')
    model_filename = f'{config["model-filename"].split(".")[0]}_{filter_name}.{config["model-filename"].split(".")[-1]}'
    keys_to_topic_filename = f'{config["keys-to-topics-filename"].split(".")[0]}_{filter_name}.{config["keys-to-topics-filename"].split(".")[-1]}'
    refit = True  # True/ False
    models_path = "./models"
    os.makedirs(models_path, exist_ok=True)
    model_filepath = os.path.join(models_path, model_filename)
    models_blob_path = "models"
    if "model-directory" in config.keys():
        models_blob_path = config["model-directory"]

    text = df_tweets[text_column]
    text = text[text != 'None'].astype(str)
    text = text[text.str.len() > 4]
    text = text.drop_duplicates()
    len_original = len(text)

    processed_ser = text.map(preprocess)
    processed_docs = [item[0] for item in processed_ser]
    mapping_list = [item[1] for item in processed_ser]
    mapping121, mapping12many = produce_mapping(mapping_list)

    # initialize and fit GSDMM model
    if not refit:
        # download topic model
        blob_client = get_blob_service_client(os.path.join(models_blob_path, model_filename), config)
        if os.path.exists(model_filepath):
            os.remove(model_filepath)
        with open(model_filepath, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

        if os.path.exists(model_filepath):
            logging.info('loading existing topic model')
            model = CustomUnpickler(open(model_filepath, "rb")).load()
        else:
            logging.error("Error: no topic model found")
    else:
        logging.info('initialize and fit topic model')
        model = MovieGroupProcess(K=6, alpha=0.3, beta=0.05, n_iters=500)
        y = model.fit(processed_docs, len(processed_docs))
        pickle.dump(model, open(model_filepath, "wb"))
        # upload topic model for later use
        blob_client = get_blob_service_client(os.path.join(models_blob_path, model_filename), config)
        with open(model_filepath, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    # create list of topic descriptions (lists of keywords) and scores
    matched_topic_score_list = [model.choose_best_label(i) for i in processed_docs]
    matched_topic_list = [t[0] for t in matched_topic_score_list]
    score_list = [t[1] for t in matched_topic_score_list]
    text = pd.DataFrame({'text': text.values, 'topic_num': matched_topic_list, 'score': score_list})

    # create list of human-readable topic descriptions (de-lemmatize)
    logging.info('create list of human-readable topics (de-lemmatize)')
    topic_list = [list(reversed(sorted(x.items(), key=operator.itemgetter(1))[-5:])) for x in
                  model.cluster_word_distribution]
    topic_list_flat = [[l[0] for l in x] for x in topic_list]
    topic_list_human_readable = topic_list_flat.copy()
    for ixt, topic in enumerate(topic_list_human_readable):
        for ixw, word in enumerate(topic):
            try:
                for raw in text.text.values:
                    for token in gensim.utils.simple_preprocess(raw):
                        if word in token:
                            topic_list_human_readable[ixt][ixw] = token
                            raise BreakIt
            except BreakIt:
                pass
    topic_list_human_readable = [[spell.correction(t) for t in l] for l in topic_list_human_readable]

    # create dataframe with best example per topic and topic description
    logging.info('create dataframe with best example per topic and topic description')
    df = pd.DataFrame()

    for topic_num, topic in enumerate(topic_list_human_readable):
        text_topic = text[text.topic_num == topic_num].sort_values(by=['score'], ascending=False).reset_index()
        frequency = len(text[text.topic_num == topic_num]) / len_original
        responses = len(text[text.topic_num == topic_num])
        if not text_topic.empty:
            representative_texts = text_topic.iloc[:10]['text'].tolist()

            for rep_text in representative_texts:
                df = df.append(pd.Series({"topic number": int(topic_num),
                                          "example": rep_text,
                                          "keywords": ', '.join(topic),
                                          "frequency (%)": frequency * 100.,
                                          "number of responses": responses}), ignore_index=True)

    df = df.sort_values(by=['frequency (%)'], ascending=False)
    df = df[['topic number', 'example', 'keywords', 'frequency (%)', 'number of responses']]
    if filter_name != 'all':
        df['topic'] = filter_name + '_' + df['topic number'].astype(int).astype(str)

    if not refit:
        # add topic descriptions and save topics locally
        blob_client = get_blob_service_client(os.path.join(models_blob_path, keys_to_topic_filename), config)
        topics_file_path = os.path.join(models_path, keys_to_topic_filename)
        if os.path.exists(topics_file_path):
            os.remove(topics_file_path)
        with open(topics_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        df_topics = pd.read_csv(topics_file_path)
        df = keywords_to_topic(df, df_topics)

    topic_dir = './topics'
    os.makedirs(topic_dir, exist_ok=True)
    df.to_csv(
        os.path.join(
            topic_dir,
            f'{config["country-code"]}_{sm_code}_topicslatestselect_{filter_name}_{start_date}_{end_date}.csv'
        ),
        index=False,
        decimal=','
    )

    if (not refit) or (filter_name != 'all'):
        # assign topic to tweets
        logging.info('assign topic to tweets')
        for ix, row in text.iterrows():
            topic = df[df['topic number'] == row['topic_num']]["topic"].values[0]
            df_tweets.at[df_tweets[text_column] == row["text"], 'topic'] = topic

    return df_tweets


def save_data(name, directory, data, id, sm_code, config):
    """
    both locally and in the datalake:
    1. save data as {directory}/{name}_latest.csv and append to {directory}/{name}_all.csv
    2. drop duplicates in {directory}/{name}_all.csv based on {id}
    3. save {directory}/{name}_all.csv
    """

    os.makedirs(f"./{directory}", exist_ok=True)
    tweets_path = f"./{directory}/{name}_latest.csv"
    data.to_csv(tweets_path, index=False, encoding="utf-8")

    # upload to datalake
    if not config["skip-datalake"]:
        blob_client = get_blob_service_client(f'{directory}/{name}_latest.csv', config)
        with open(tweets_path, "rb") as upload_file:
            blob_client.upload_blob(upload_file, overwrite=True)

    # upload to database
    if not config["skip-database"]:
        save_to_db(sm_code, data, config)

    # append to existing twitter dataframe
    final_table_columns = ["index", "source", "member_count", "message_count", \
        "text", "datetime", "id", "date", "rcrc", "cva", "full_text_en"]
    data.drop(columns=[col for col in data if col not in final_table_columns], inplace=True)
    if containsNumber(name):
        name = "_".join(name.split('_')[0:3])
    data_all_path = f"./{directory}/{name}_all.csv"
    try:
        if not config["skip-datalake"]:
            blob_client = get_blob_service_client(f'{directory}/{name}_all.csv', config)
            with open(data_all_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        data_old = pd.read_csv(data_all_path)  # , lines=True)
        data_all = data_old.append(data, ignore_index=True)
    except:
        data_all = data.copy()

    # drop duplicates and save
    data_all = data_all.drop_duplicates(subset=[id])
    data_all.to_csv(data_all_path, index=False, encoding="utf-8")

    # upload to datalake
    if not config["skip-datalake"]:
        blob_client = get_blob_service_client(f'{directory}/{name}_all.csv', config)
        with open(data_all_path, "rb") as upload_file:
            blob_client.upload_blob(upload_file, overwrite=True)


def read_db(sm_code, start_date, end_date, config):
    '''
    Retrieve messages between a certain period from AZ Database
    '''

    connection, cursor = connect_to_db(config)

    table_name = config["azure-database-name"]
    query = f"""SELECT * \
        FROM {table_name} \
        WHERE sm_code = '{sm_code}' \
        AND date \
        BETWEEN '{start_date}' AND '{end_date}' \
        """
    try:
        df_messages = pd.read_sql(query, connection)
        logging.info(f"Succesfully retrieve {sm_code} messages \
            from {start_date} to {end_date} from table {table_name}")
    except Exception:
        df_messages = None
        logging.error(f"Failed to retrieve SQL table")
    finally:
        cursor.close()
        connection.close()
        logging.info("AZ Database connection is closed")

    return df_messages


def save_to_db(sm_code, data, config):

    # Prepare for storing in Azure db
    data['post'] = np.where(
        data['post'],
        1,
        0
    )

    data = data.astype(object).where(pd.notnull(data), None)

    current_datetime = datetime.datetime.now()

    # Make connection to Azure datbase
    connection, cursor = connect_to_db(config)

    try:
        mySql_insert_query =f"""INSERT INTO {config['azure-database-name']} (id_post, country, sm_code, source, datetime_scraped,\
         datetime, date, post, text_post, text_reply) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        for idx, row in data.iterrows():

            cursor.execute(
                mySql_insert_query,
                row['id'],
                config['country-code'],
                sm_code,
                row['source'],
                current_datetime,
                row['datetime'],
                row['date'],
                row['post'],
                row['text_post'],
                row['text_reply']
            )

            connection.commit()

        logging.INFO(f"Succesfully inserted {len(data)} entries into table 'smm.messages'")

    except pyodbc.Error as error:
        logging.WARNING("Failed to insert into SQL table {}".format(error))

    finally:
        cursor.close()
        connection.close()
        logging.INFO("Pyodbc connection is closed")


def connect_to_db(config):
    # Get credentials
    database_secret = get_secret_keyvault("azure-database-secret", config)
    database_secret = json.loads(database_secret)

    try:
        # Connect to db
        driver = '{ODBC Driver 18 for SQL Server}'
        connection = pyodbc.connect(
            f'DRIVER={driver};'
            f'SERVER=tcp:{database_secret["SQL_DB_SERVER"]};'
            f'PORT=1433;DATABASE={database_secret["SQL_DB"]};'
            f'UID={database_secret["SQL_USER"]};'
            f'PWD={database_secret["SQL_PASSWORD"]}'
        )

        cursor = connection.cursor()
        logging.INFO("Successfully connected to database")
    except pyodbc.Error as error:
        logging.INFO("Failed to connect to database {}".format(error))

    return connection, cursor


def containsNumber(string):
    for character in string:
        if character.isdigit():
            return True
    return False


def get_daily_messages(start_date, end_date, telegram_data_path, config):
    '''
    Download daily scraped Telegram files from storage
    and merge into one single dataframe
    '''

    dates = [start_date + datetime.timedelta(days=x) \
    for x in range((end_date - start_date).days)]
    dates.append(end_date)

    df_messages = pd.DataFrame()
    # i = 1
    for i in range(len(dates)-1):
        date_1 = dates[i].strftime('%Y-%m-%d')
        date_2 = dates[i+1].strftime('%Y-%m-%d')
        messages_path = telegram_data_path + f"/{config['country-code']}_TL_messages_{date_1}_{date_2}_latest.csv"
        try:
            blob_client = get_blob_service_client(messages_path, config)
            with open(messages_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            df = pd.read_csv(messages_path)
            df_messages = df_messages.append(df, ignore_index=True)
            df_messages.drop_duplicates(subsets=['text_post', 'text_reply', 'datetime'])
        except PartialBatchErrorException as e:
            logging.warning(f"unable to get {messages_path}: {e}")
        # i += 1

    return df_messages


def previous_weekday(d, weekday):
    '''
    Find the closest past weekday from a given day
    i.e. what is the closest past Wednesday from today?
    d: a date in datetime type
    weekday: 0=Mon, 1=Tue, 2=Wed, ect.
    '''
    days_behind = weekday - d.weekday()
    if days_behind > 0: # Target day already happened this week
        days_behind = days_behind - 7
    return d + datetime.timedelta(days_behind)