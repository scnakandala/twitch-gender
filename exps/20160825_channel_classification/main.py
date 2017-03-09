import os
import random
import re
from random import shuffle

from gensim.models.doc2vec import LabeledSentence, Doc2Vec


class LabeledLineSentence(object):
    def __init__(self, messages_dic, is_sample=True):
        self.documents = []
        self.messages_dic = messages_dic
        self.is_sample = is_sample

    def __iter__(self):
        for user in self.messages_dic:
            if self.is_sample:
                for i in range(200):
                    messages_list = random.sample(self.messages_dic[user], 100)
                    yield LabeledSentence((' '.join(messages_list)).split(), [user + "_" + str(i)])
            else:
                messages_list = self.messages_dic[user]
                yield LabeledSentence((' '.join(messages_list)).split(), [user])

    def to_array(self):
        if self.is_sample:
            fp = open('channel_random_message_samples.csv.dat', 'w')

        for user in self.messages_dic:
            if self.is_sample:
                no_of_chat_samples = int(0.1 * len(self.messages_dic[user]))//100
                chat_sample = random.sample(self.messages_dic[user], no_of_chat_samples * 100)
                for i in range(no_of_chat_samples):
                    messages_list = chat_sample[i*100:(i+1)*100]
                    message_in_one_line = ' '.join(messages_list)
                    fp.write(user + "_" + str(i) + "," + message_in_one_line + "\n")
                    self.documents.append(LabeledSentence(message_in_one_line.split(), [user + "_" + str(i)]))
            else:
                messages_list = self.messages_dic[user]
                message_in_one_line = ' '.join(messages_list)
                self.documents.append(LabeledSentence(message_in_one_line.split(), [user]))
        return self.documents

    def sentences_perm(self):
        shuffle(self.documents)
        return self.documents


def generate_channel_chats_d2v(channel_list, is_sample=True):
    channel_chats = {}

    for file_name in os.listdir("../../data/channel_chat_logs/cleaned"):
        if file_name.endswith('.csv'):
            file_path = "../../data/channel_chat_logs/cleaned/" + file_name
            with open(file_path, 'r') as fp:
                print('reading : ' + file_path)
                for line in fp:
                    splits = line.split(",")
                    stream = splits[1].replace('#', '')
                    message = splits[3]

                    if len(message) == 0:
                        continue

                    gendered_terms = [r'\bhe\b', r'\bhes', r'\bshe\b', r'\bshes\b', r'\bhis\b', r'\bher\b', r'\bbro\b',
                          r'\bman\b', r'\bsir\b', r'\bdude\b', r'\bgirl\b', r'\bgirls\b', r'\blady\b',
                          r'\bgurl\b', r'\bhers\b', r'\bhisself\b', r'\bherself\b', r'\bman\b', r'\bwoman\b']
                    for temp in gendered_terms:
                        message = re.sub(temp, '', message)

                    try:
                        channel_chats[stream].append(message.strip())
                    except KeyError:
                        channel_chats[stream] = [message.strip()]

    sentences = LabeledLineSentence(channel_chats, is_sample)
    model = Doc2Vec(min_count=10, window=5, size=100, sample=1e-5, negative=5, workers=8, dm=0, dbow_words=1)
    model.build_vocab(sentences.to_array())

    for epoch in range(10):
        print('doc2vec epoch : ' + str(epoch))
        model.train(sentences.sentences_perm())

    if is_sample:
        model.save('channel_random_chats.d2v')
    else:
        model.save('channel_all_chats.d2v')

if __name__ == '__main__':
    channels = [line.strip().split(',')[0] for line in open('../female_channels.csv', 'r')] \
               + [line.strip().split(',')[0] for line in open('../male_channels.csv', 'r')]
    # generate_channel_chats_d2v(channels, True)
    generate_channel_chats_d2v(channels, False)