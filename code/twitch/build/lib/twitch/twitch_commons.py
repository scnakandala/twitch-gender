import csv

from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn import metrics

import numpy as np

from nltk.corpus import stopwords as nltk_stopwords
from wordcloud import STOPWORDS as wordcloud_stopwords

import matplotlib

import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_word_count_file(file_path):
    with open(file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        counts = {rows[0]: int(rows[1]) for rows in reader}
    return counts


def get_stopwords():
    # selecting words which appear only in the web text corpus and not in stop words
    stopwords_list = list(nltk_stopwords.words('english'))
    stopwords_list = stopwords_list + list(wordcloud_stopwords)

    return stopwords_list


def get_word_channel_count(base_path, broadcaster_list):
    return get_token_channel_count(base_path, broadcaster_list, 'word')


def get_bigram_channel_count(base_path, broadcaster_list):
    return get_token_channel_count(base_path, broadcaster_list, 'bigram')


def get_token_channel_count(base_path, broadcaster_list, type):
    channel_word_occurrences_count = {}
    for broadcaster in broadcaster_list:
        with open(base_path + "/" + broadcaster + "_" + type + "_counts.csv", mode='r',
                  encoding='utf-8') as infile:
            reader = csv.reader(infile)
            boradcaster_word_counts = {rows[0]: int(rows[1]) for rows in reader}

            for key in boradcaster_word_counts:
                if boradcaster_word_counts[key] >= 100:
                    try:
                        channel_word_occurrences_count[key] += 1
                    except:
                        channel_word_occurrences_count[key] = 1
    return channel_word_occurrences_count


def calculate_log_odds_idp(global_counts, female_corpus_counts,
                           female_channel_word_occurances_count, male_corpus_counts,
                           male_channel_word_occurances_count):
    global_df = pd.DataFrame(list(global_counts.items()), columns=['word', 'global_count'])

    global_df['female_corpus_counts'] = global_df.word.apply(lambda word: female_corpus_counts[word]
    if word in female_corpus_counts.keys() else 0)
    global_df['male_corpus_counts'] = global_df.word.apply(lambda word: male_corpus_counts[word]
    if word in male_corpus_counts.keys() else 0)
    global_df['female_channel_counts'] = global_df.word.apply(lambda word: female_channel_word_occurances_count[word]
    if word in female_channel_word_occurances_count.keys() else 0)
    global_df['male_channel_counts'] = global_df.word.apply(lambda word: male_channel_word_occurances_count[word]
    if word in male_channel_word_occurances_count.keys() else 0)

    # Log odds ratio infromatice Dirichlet prior calculation
    # female -> i, male -> j
    ni = global_df.female_corpus_counts.sum()
    nj = global_df.male_corpus_counts.sum()
    a0 = global_df.global_count.sum()

    yiw = global_df.female_corpus_counts
    yjw = global_df.male_corpus_counts
    aw = global_df.global_count

    sd = (1 / (yiw + aw) + 1 / (yjw + aw)).apply(np.sqrt)

    delta = (((yiw + aw) / (ni + a0 - (yiw + aw))) / ((yjw + aw) / (nj + a0 - (yjw + aw)))).apply(np.log)

    global_df['log_odds_z_score'] = delta / sd

    global_df = global_df.sort_values(by=['log_odds_z_score'], ascending=False)

    return global_df


def reduce_dim(feature_vectors, model_type, n_components=2):
    transformed_vectors = []
    if model_type == 'pca':
        pca = decomposition.PCA()
        pca.fit([v[0] for v in feature_vectors])
        pca.n_components = n_components
        pca_transformed_vectors = pca.fit_transform(feature_vectors)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        transformed_vectors = pca_transformed_vectors
    elif model_type == 'tsne':
        pca = decomposition.PCA()
        pca.fit([v[0] for v in feature_vectors])
        pca.n_components = 50
        pca_transformed_vectors = pca.fit_transform(feature_vectors)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        transformed_vectors = TSNE(n_components=n_components, verbose=2).fit_transform(pca_transformed_vectors)
    elif model_type == 'mds':
        transformed_vectors = MDS(n_components=n_components).fit_transform(feature_vectors)

    return transformed_vectors


def plot_roc_curve(fpr, tpr):
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()