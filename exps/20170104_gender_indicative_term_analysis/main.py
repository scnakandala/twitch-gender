import os
import csv
from nltk.tokenize import TweetTokenizer
import pandas as pd

pronouns = ['i', 'we', 'me', 'us', 'you', 'she', 'her', 'he', 'him', 'it', 'they', 'them', 'u', 'ur', 'yr']

# FIXME Not Complete List
emotions = ['sad', 'love', 'lov', 'luv', 'glad', 'mad', 'scared', 'fun', 'funny', 'angry', 'proud', 'calm', 'brave']

emoticons = [':)', ':)', ':D', ':3', ':-3', ':]', ':-]', ':>', ':->', '8-)', '8)', ':}', ':-}', '<3', ':*']

cmc_hesitation = ['lol', 'omg', 'ah', 'hmm', 'ugh', 'grr', 'um', 'umm']

assent = ['okay', 'yes', 'yess', 'yesss', 'yessss']

taboo = ['shit', 'ass', 'arse', 'cock', 'boong', 'gook', 'faggot', 'bitch', 'coon', 'coonass', 'fuck', 'cunt']


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

    for i, file_name in enumerate(os.listdir("../../data/channel_chat_logs/raw")):
        if file_name.endswith('.csv'):
            file_path = "../../data/channel_chat_logs/raw/" + file_name
            with open(file_path, 'r') as fp:
                print('reading : ' + str(i) + " " + file_path)
                for line in fp:
                    splits = line.split(",")
                    stream = splits[1].replace('#', '')
                    user = splits[2]
                    message = splits[3]

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

    with open("user_chat_counts.csv", "w") as f:
        csv.writer(f).writerows((k,) + v for k, v in user_chat_counts.items())


def generate_feature_vectors(user_list):
    tknzr = TweetTokenizer()
    user_feature_vectors = {}

    for i, file_name in enumerate(os.listdir("../../data/channel_chat_logs/raw")):
        if file_name.endswith('.csv'):
            file_path = "../../data/channel_chat_logs/raw/" + file_name
            with open(file_path, 'r') as fp:
                print('reading : ' + str(i) + " " + file_path)
                for line in fp:
                    splits = line.split(",")
                    user = splits[2]
                    message = splits[3]

                    if user not in user_list or len(message.strip()) == 0:
                        continue

                    if user not in user_feature_vectors:
                        user_feature_vectors[user] = [0] * 7

                    # Feature Ordering
                    # 1. is_female
                    # 2. pronouns
                    # 3. emotions
                    # 4. emoticons
                    # 5. cmc_hesitation
                    # 6. assent
                    # 7. taboo
                    # 8. numbers
                    tokens = tknzr.tokenize(message)
                    # tokens = message.split()
                    for token in tokens:
                        if token in pronouns:
                            user_feature_vectors[user][0] += 1
                        elif token in emotions:
                            user_feature_vectors[user][1] += 1
                        elif token in emoticons:
                            user_feature_vectors[user][2] += 1
                        elif token in cmc_hesitation:
                            user_feature_vectors[user][3] += 1
                        elif token in assent:
                            user_feature_vectors[user][4] += 1
                        elif token in taboo:
                            user_feature_vectors[user][5] += 1
                        elif token.isdigit():
                            user_feature_vectors[user][6] += 1

    with open("user_feature_vectors.csv", "w") as f:
        csv.writer(f).writerows((k, ",".join(map(str, v))) for k, v in user_feature_vectors.items())


if __name__ == "__main__":
    # generate_user_chat_counts()
    df = pd.read_csv('./user_chat_counts.csv', header=None, names=['user', 'total', 'male', 'female'])
    df = df[df.total >= 100]
    df = df.sample(n=10000)
    select_user_list = df.user.values.tolist()
    print(len(select_user_list))
    generate_feature_vectors(select_user_list)
