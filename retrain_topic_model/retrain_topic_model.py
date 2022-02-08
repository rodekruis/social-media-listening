import preprocessor as tp
import pandas as pd
import numpy as np
import os
import operator
import gensim
import enchant
import transformers
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
STOPWORDS = list(STOPWORDS) # TBI move to config
STOPWORDS.append('covid')
STOPWORDS.append('vaccine')
STOPWORDS.append('vaccines')
STOPWORDS.append('vaccinated')
STOPWORDS.append('namibia')
STOPWORDS.append('says')
STOPWORDS.append('because')
STOPWORDS.append('like')
STOPWORDS.append('get')
from GSDMM import MovieGroupProcess
from tqdm import tqdm
tqdm.pandas()
tp.set_options(tp.OPT.URL, tp.OPT.EMOJI, tp.OPT.MENTION)
en_dict = enchant.Dict("en_US")
nltk.download('punkt')
import logging
import click


class BreakIt(Exception):
    pass


def preprocess(text):
    result = []
    token_list = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) > 2 and 'haha' not in token and token.lower() not in STOPWORDS:
            result.append(stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v')))
            token_list.append(token)
    return result, dict(zip(result, token_list))


def produce_mapping(mapping_list):
    mapping_pairs = pd.concat([pd.DataFrame([(k, v) for k, v in d.items()]) for d in mapping_list])
    mapping_pairs['count'] = 1
    mapping121 = mapping_pairs.groupby(by=[0, 1]).count().reset_index().sort_values(by=[0, 'count'], ascending=False).groupby(by=0).head(1)
    mapping12many = mapping_pairs.drop(columns=['count']).drop_duplicates()
    return mapping121, mapping12many


@click.command()
@click.option('--data', help='input data file (csv)')
@click.option('--textcolumn', help='text column', multiple=True)
def main(data, textcolumn):

    print('predicting topic')
    models_path = "./models"
    model_filename = 'gsdmm-model-v2.pickle'
    os.makedirs(models_path, exist_ok=True)
    model_filepath = os.path.join(models_path, model_filename)

    df_tweets = pd.read_csv(data)

    if type(textcolumn) == tuple:
        col0 = textcolumn[0]
        for col in textcolumn[1:]:
            df_tweets[col0] = df_tweets[col0].fillna(df_tweets[col])
        textcolumn = col0
        # rumors_cg
    text = df_tweets[textcolumn]


    text = text[text != 'None'].astype(str)
    text = text[text.str.len() > 4]
    text = text.drop_duplicates()
    len_original = len(text)

    print(f'found {len_original} valid entries (out of {len(df_tweets)})')

    processed_ser = text.map(preprocess)
    processed_docs = [item[0] for item in processed_ser]
    mapping_list = [item[1] for item in processed_ser]
    mapping121, mapping12many = produce_mapping(mapping_list)

    # initialize and fit GSDMM model
    print('initialize and fit topic model')
    model = MovieGroupProcess(K=6, alpha=0.3, beta=0.05, n_iters=500)
    y = model.fit(processed_docs, len(processed_docs))
    # save model
    pickle.dump(model, open(model_filepath, "wb"))


    # create list of topic descriptions (lists of keywords) and scores
    matched_topic_score_list = [model.choose_best_label(i) for i in processed_docs]
    matched_topic_list = [t[0] for t in matched_topic_score_list]
    score_list = [t[1] for t in matched_topic_score_list]
    text = pd.DataFrame({'text': text.values, 'topic_num': matched_topic_list, 'score': score_list})

    # create list of human-readable topic descriptions (de-lemmatize)
    logging.info('create list of human-readable topics (de-lemmatize)')
    topic_list = [list(reversed(sorted(x.items(), key=operator.itemgetter(1))[-5:])) for x in model.cluster_word_distribution]
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
            representative_text = ';'.join(text_topic.iloc[:10]['text'].values.tolist())

            df = df.append(pd.Series({"topic number": int(topic_num),
                                      "example": representative_text,
                                      "keywords": ', '.join(topic),
                                      "frequency (%)": frequency * 100.,
                                      "number of responses": responses}), ignore_index=True)
    df = df.sort_values(by=['frequency (%)'], ascending=False)

    topic_dir = './topics'
    os.makedirs(topic_dir, exist_ok=True)
    df.to_csv(os.path.join(topic_dir, 'topics_latest_select.csv'))

    # # assign topic to tweets
    # logging.info('assign topic to tweets')
    # for ix, row in text.iterrows():
    #     topic = df[df['topic number'] == row['topic_num']]["topic"].values[0]
    #     df_tweets.at[df_tweets[textcolumn] == row["text"], 'topic'] = topic

    df_tweets.to_csv(os.path.join(topic_dir, 'rumors.csv'))


if __name__ == "__main__":
    main()