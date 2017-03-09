from gensim.models import Doc2Vec
from sklearn import decomposition
import numpy as np
import sys
import operator
from sklearn import metrics
from sklearn import preprocessing
from sklearn import cross_validation

import os
import heapq
import pandas as pd
import random as rnd
from random import shuffle
import collections
import matplotlib.ticker as ticker

from heapq import nlargest


df = pd.read_csv('./user_chat_counts.csv.dat', header=None, names=['user', 'total', 'male', 'female'])
df['female_chat_percentage'] = (df.female*100)/df.total

print('all users: ' + str(len(df.index)))

print('users with atleast 100 messages: ' + str(len(df[df.total >= 100].index)))

all_users = df[(df.total >= 100)]
user_docvec_ids = df[(df.total >= 100)]# & ((df.female_chat_percentage == 100) | (df.female_chat_percentage == 0))]
users = user_docvec_ids.user.tolist()
users_dic = {}
for u in users:
    users_dic[u] = 1

female_chat_percentages = user_docvec_ids.female_chat_percentage.tolist()

no_of_users = len(users)
print('filtered users: ' + str(no_of_users))

channel_docvec_ids = [line.strip().split(',')[0] for line in open('../female_channels.csv', 'r')] \
        + [line.strip().split(',')[0] for line in open('../male_channels.csv', 'r')]
print(len(channel_docvec_ids))

popular_female_users = set()
for file_name in os.listdir("../../data/channel_chat_logs/cleaned"):
    if file_name.endswith('.csv'):
        stream = file_name.replace('_chat_log.csv', '')
        if stream in channel_docvec_ids[0:100]:
            file_path = "../../data/channel_chat_logs/cleaned/" + file_name
            with open(file_path, 'r') as fp:
                for line in fp:
                    splits = line.split(",")
                    stream = splits[1]
                    user = splits[2]
                    message = splits[3]
                    if user not in users_dic:
                        continue
                    if len(message) == 0:
                        continue
                    popular_female_users.add(user)

print(len(popular_female_users))

less_popular_female_users = set()
for file_name in os.listdir("../../data/channel_chat_logs/cleaned"):
    if file_name.endswith('.csv'):
        stream = file_name.replace('_chat_log.csv', '')
        if stream in channel_docvec_ids[100:200]:
            file_path = "../../data/channel_chat_logs/cleaned/" + file_name
            with open(file_path, 'r') as fp:
                for line in fp:
                    splits = line.split(",")
                    stream = splits[1]
                    user = splits[2]
                    message = splits[3]
                    if user not in users_dic:
                        continue
                    if len(message) == 0:
                        continue
                    less_popular_female_users.add(user)

print(len(less_popular_female_users))

print(len(popular_female_users) - len(popular_female_users - less_popular_female_users))