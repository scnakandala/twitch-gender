from gensim.models import Doc2Vec
from sklearn import decomposition
import numpy as np
import sys
import operator
from sklearn import metrics
from sklearn import preprocessing
from sklearn import cross_validation
from adjustText import adjust_text
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import os
import heapq
import pandas as pd
import random as rnd
from random import shuffle
import collections
import matplotlib.ticker as ticker

from heapq import nlargest

from twitch import commons

matplotlib.use('Agg')
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

df = pd.read_csv('./user_chat_counts.csv.dat', header=None, names=['users', 'total', 'male', 'female'])
df['female_chat_percentage'] = (df.female*100)/df.total

print('all users: ' + str(len(df.index)))

all_female = df[(df.total >= 100)].users.values.tolist()

female_only = df[(df.total >= 100) & (df.female_chat_percentage == 100)].users.values.tolist()

female_90 = df[(df.total >= 100) & ((df.female_chat_percentage >= 85)
                                    & (df.female_chat_percentage < 95))].users.values.tolist()

female_80 = df[(df.total >= 100) & ((df.female_chat_percentage >= 75)
                                    & (df.female_chat_percentage < 85))].users.values.tolist()

female_70 = df[(df.total >= 100) & ((df.female_chat_percentage >= 65)
                                    & (df.female_chat_percentage < 75))].users.values.tolist()

female_60 = df[(df.total >= 100) & ((df.female_chat_percentage >= 55)
                                    & (df.female_chat_percentage < 65))].users.values.tolist()

female_50 = df[(df.total >= 100) & ((df.female_chat_percentage >= 45)
                                    & (df.female_chat_percentage < 55))].users.values.tolist()

print(len(all_female))
print(len(female_only))
print(len(female_90))
print(len(female_80))
print(len(female_70))
print(len(female_60))
print(len(female_50))

temp = [line.strip().split(',')[0] for line in open('../female_channels.csv', 'r')]
female_channels = {}
for t in temp:
    female_channels[t] = 1

gendered_terms = [r'\bhe\b', r'\bhes', r'\bshe\b', r'\bshes\b', r'\bhis\b', r'\bher\b', r'\bbro\b',
                  r'\bman\b', r'\bsir\b', r'\bdude\b', r'\bgirl\b', r'\bgirls\b', r'\blady\b',
                  r'\bgurl\b', r'\bhers\b', r'\bhisself\b', r'\bherself\b', r'\bman\b', r'\bwoman\b']

select_users = {}
for t in all_female:
    select_users[t] = 1

user_chats = {}

for file_name in os.listdir("../../data/channel_chat_logs/cleaned"):
    if file_name.endswith('.csv'):
        file_path = "../../data/channel_chat_logs/cleaned/" + file_name
        with open(file_path, 'r') as fp:
            #print('reading : ' + file_path)
            for line in fp:
                splits = line.split(",")
                channel = splits[1].replace('#', '')
                user = splits[2]
                message = splits[3]

                # avoiding users with less number of messages
                if user not in select_users:
                    continue

                if channel not in female_channels:
                    continue

                for temp in gendered_terms:
                    message = re.sub(temp, '', message)

                if len(message.strip()) == 0:
                    continue

                try:
                    user_chats[user].append(message.strip())
                except KeyError:
                    user_chats[user] = [message.strip()]
print('done')

class LabeledLineSentence(object):
    def __init__(self, messages_dic):
        self.documents = []
        self.messages_dic = messages_dic

    def __iter__(self):
        for user in self.messages_dic:
            messages_list = self.messages_dic[user]
            yield LabeledSentence((' '.join(messages_list)).split(), [user])

    def to_array(self):
        for user in self.messages_dic:
            messages_list = self.messages_dic[user]
            message_in_one_line = ' '.join(messages_list)
            self.documents.append(LabeledSentence(message_in_one_line.split(), [user]))
        return self.documents

    def sentences_perm(self):
        shuffle(self.documents)
        return self.documents

sentences = LabeledLineSentence(user_chats)
model = Doc2Vec(min_count=10, window=5, size=100, sample=1e-5, negative=5, workers=8, dm=0, dbow_words=1)
model.build_vocab(sentences.to_array())

for epoch in range(10):
    print('doc2vec epoch : ' + str(epoch))
    model.train(sentences.sentences_perm())
    model.save('user_all_chats.d2v')