import csv
import os
import re
import string
import math

import numpy as np
import statistics
from scipy import interp
import pandas as pd

from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

from nltk.corpus import stopwords as nltk_stopwords
from wordcloud import STOPWORDS as wordcloud_stopwords


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_list_raw_chatlogs(raw_data_path):
    """ returns a set of raw chat log (.log) files """
    files = set()
    for fname in os.listdir(raw_data_path):
        if re.match(r'.+\.log', fname):
            files.add(os.path.join(raw_data_path, fname))
    return files

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
        pca.fit(feature_vectors)
        pca.n_components = n_components
        pca_transformed_vectors = pca.fit_transform(feature_vectors)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        transformed_vectors = pca_transformed_vectors
    elif model_type == 'tsne':
        # if dimensionality is very high we first reduce it using PCA
        if len(feature_vectors[0]) > 100:
            pca = decomposition.PCA()
            pca.fit(feature_vectors)
            pca.n_components = 100
            feature_vectors = pca.fit_transform(feature_vectors)
        transformed_vectors = TSNE(n_components=n_components, verbose=2).fit_transform(feature_vectors)
    elif model_type == 'mds':
        transformed_vectors = MDS(n_components=n_components).fit_transform(feature_vectors)

    return transformed_vectors


def plot_roc_curve(fpr, tpr):
    roc_auc = metrics.auc(fpr, tpr)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size

    # plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def do_message_word_replaces(message):
    replaces = {"let's": "let us", "'s": " is", "'re": " are", "don’t": "do not", "won’t": "will not",
                "can’t": "can not", "didn't": "did not", "aren’t": "are not", "couldn’t": "could not",
                "'ll": " will", "’ve": " have", "'d": " would",

                # words that I added to get cleaner results
                "hiiii ": "hi ", " hiiii ": " hi ", "hiii ": "hi ", " hiii ": " hi ", "hii ": "hi ", " hii ": " hi ",
                "shes ": "she ", " shes ": " she ", "boob ": "boobs ", " boob ": "boobs ", "boobies ": "boobs ",
                " boobies ": " boobs"
                }

    for key in replaces:
        message = re.sub(key, replaces[key], message)

    return message


def strip_non_ascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


def clean_message(message):
    message = message.lower().strip()
    message = do_message_word_replaces(message)

    # stripping links
    message = ' '.join(re.sub("(\w+:\/\/\S+)", " ", message).split())
    message = strip_non_ascii(message)
    exclude = set(string.punctuation)
    exclude.add('\n')
    exclude.add('\r')
    message = ''.join(ch for ch in message if ch not in exclude)
    message = re.sub(' +', ' ', message)
    message = re.sub('\d+', ' ', message)
    return message


def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def build_lr_classification_model(train_arrays, train_labels, test_arrays, test_labels):
    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)
    print('model accuracy : ' + str(classifier.score(test_arrays, test_labels)))
    predict_score = classifier.predict_proba(test_arrays)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, predict_score)
    plot_roc_curve(fpr, tpr)

    return classifier, classifier.coef_, classifier.intercept_


def build_lr_classification_model_cv(x, y, n_folds):
    cv = StratifiedKFold(y, n_folds=n_folds)
    classifier = LogisticRegression()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    scores = []
    for i, (train, test) in enumerate(cv):
        classifier.fit(x[train], y[train])
        scores.append(classifier.score(x[test], y[test]))

        probas_ = classifier.predict_proba(x[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

    mean = statistics.mean(scores)
    std = statistics.pstdev(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (mean, std * 2))

    return classifier, classifier.coef_, classifier.intercept_
