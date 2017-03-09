import csv
import os
import random
from random import shuffle
import re

import pandas as pd
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

gendered_terms = [r'\bhe\b', r'\bhes', r'\bshe\b', r'\bshes\b', r'\bhis\b', r'\bher\b', r'\bbro\b',
                  r'\bman\b', r'\bsir\b', r'\bdude\b', r'\bgirl\b', r'\bgirls\b', r'\blady\b',
                  r'\bgurl\b', r'\bhers\b', r'\bhisself\b', r'\bherself\b', r'\bman\b', r'\bwoman\b']


class LabeledLineSentence(object):
    def __init__(self, messages_dic, is_sample=True):
        self.documents = []
        self.messages_dic = messages_dic
        self.is_sample = is_sample

    def __iter__(self):
        for user in self.messages_dic:
            if self.is_sample:
                messages_list = random.sample(self.messages_dic[user], min(100, len(self.messages_dic[user])))
            else:
                messages_list = self.messages_dic[user]
            yield LabeledSentence((' '.join(messages_list)).split(), [user])

    def to_array(self):
        if self.is_sample:
            fp = open('user_random_message_sample.csv.dat', 'w')
        for user in self.messages_dic:
            if self.is_sample:
                messages_list = random.sample(self.messages_dic[user], min(100, len(self.messages_dic[user])))
            else:
                messages_list = self.messages_dic[user]
            message_in_one_line = ' '.join(messages_list)
            if self.is_sample:
                fp.write(user + "," + message_in_one_line + "\n")
            self.documents.append(LabeledSentence(message_in_one_line.split(), [user]))
        return self.documents

    def sentences_perm(self):
        shuffle(self.documents)
        return self.documents


def get_user_chats(user_list):
    select_users = {}
    for t in user_list:
        select_users[t] = 1

    user_chats = {}

    for file_name in os.listdir("../../data/channel_chat_logs/cleaned"):
        if file_name.endswith('.csv'):
            file_path = "../../data/channel_chat_logs/cleaned/" + file_name
            with open(file_path, 'r') as fp:
                print('reading : ' + file_path)
                for line in fp:
                    splits = line.split(",")
                    user = splits[2]
                    message = splits[3]

                    # avoiding users with less number of messages
                    if user not in select_users:
                        continue

                    for temp in gendered_terms:
                        message = re.sub(temp, '', message)

                    if len(message.strip()) == 0:
                        continue

                    try:
                        user_chats[user].append(message.strip())
                    except KeyError:
                        user_chats[user] = [message.strip()]
    return user_chats


def generate_user_chats_d2v(user_list, is_sample=True):
    user_chats = get_user_chats(user_list)
    sentences = LabeledLineSentence(user_chats, is_sample)
    model = Doc2Vec(min_count=10, window=5, size=100, sample=1e-5, negative=5, workers=8, dm=0, dbow_words=1)
    model.build_vocab(sentences.to_array())

    for epoch in range(10):
        print('doc2vec epoch : ' + str(epoch))
        model.train(sentences.sentences_perm())
        if is_sample:
            model.save('user_random_chats.d2v')
        else:
            model.save('user_all_chats.d2v')


def generate_user_all_chat_log(user_list):
    user_chats = get_user_chats(user_list)
    fp = open('user_all_message.csv.dat', 'w')
    for user in user_chats:
        fp.write(user + "," + ' '.join(user_chats[user]) + "\n")
    fp.close()


def generate_user_chat_counts():
    user_chat_counts = {}

    temp = [line.strip().split(',')[0] for line in open('../female_channels.csv', 'r')]
    female_channels = {}
    for t in temp:
        female_channels[t] = 1

    temp = [line.strip().split(',')[0] for line in open('../male_channels.csv', 'r')]
    male_channels = {}
    for t in temp:
        male_channels[t] = 1

    for i, file_name in enumerate(os.listdir("../../data/channel_chat_logs/cleaned")):
        if file_name.endswith('.csv'):
            file_path = "../../data/channel_chat_logs/cleaned/" + file_name
            with open(file_path, 'r') as fp:
                print('reading : ' + str(i) + " " + file_path)
                for line in fp:
                    splits = line.split(",")
                    stream = splits[1].replace('#', '')
                    user = splits[2]
                    message = splits[3]

                    for temp in gendered_terms:
                        message = re.sub(temp, '', message)

                    # not counting empty messages
                    if len(message.strip()) == 0:
                        continue

                    try:
                        if stream in female_channels:
                            user_chat_counts[user] = (user_chat_counts[user][0] + 1, user_chat_counts[user][1],
                                                      user_chat_counts[user][2] + 1)
                        elif stream in male_channels:
                            user_chat_counts[user] = (user_chat_counts[user][0] + 1, user_chat_counts[user][1] + 1,
                                                      user_chat_counts[user][2])
                    except KeyError:
                        if stream in female_channels:
                            user_chat_counts[user] = (1, 0, 1)
                        elif stream in male_channels:
                            user_chat_counts[user] = (1, 1, 0)

    with open("user_chat_counts.csv.dat", "w") as f:
        csv.writer(f).writerows((k,) + v for k, v in user_chat_counts.items())


def find_user_channel_count(selected_user_list):

    select_user_map = {}
    for x in select_user_list:
        select_user_map[x] = 1

    user_channel_counts = {}

    for i, file_name in enumerate(os.listdir("../../data/channel_chat_logs/cleaned")):
        if file_name.endswith('.csv'):
            file_path = "../../data/channel_chat_logs/cleaned/" + file_name
            with open(file_path, 'r') as fp:
                print('reading : ' + str(i) + " " + file_path)
                for line in fp:
                    splits = line.split(",")
                    stream = splits[1].replace('#', '')
                    user = splits[2]
                    if (user != stream) and (user in select_user_map):
                        try:
                            user_channel_counts[user].add(stream)
                        except KeyError:
                            user_channel_counts[user] = set()
                            user_channel_counts[user].add(stream)

    with open("user_channel_counts.csv.dat", "w") as f:
        csv.writer(f).writerows((k, str(len(v))) for k, v in user_channel_counts.items())


if __name__ == "__main__":
    # generate_user_chat_counts()
    df = pd.read_csv('./user_chat_counts.csv.dat', header=None, names=['user', 'total', 'male', 'female'])
    df = df[df.total >= 100]
    select_user_list = df.user.values.tolist()
    # print('no of users : ' + str(len(select_user_list)))
    # generate_user_chats_d2v(select_user_list, False)
    # generate_user_chats_d2v(select_user_list, True)
    # generate_user_all_chat_log(select_user_list)

    find_user_channel_count(select_user_list)
