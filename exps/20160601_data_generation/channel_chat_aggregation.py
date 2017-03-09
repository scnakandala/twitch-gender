from twitch import commons
import string
import os
import re
import csv


def generate_users_per_channel_counts():
    users_per_channel = {}

    file_paths = []
    for s in os.listdir("../../data/raw_data"):
        if re.match('.+\.log', s):
            file_paths.append("../../data/raw_data/" + s)

    for file_path in file_paths:
        with open(file_path) as fp:
            for line in fp:
                splits = re.split('\x1e', line)
                if len(splits) == 4:
                    channel = splits[1]
                    user = splits[2]
                    if channel in users_per_channel:
                        users_per_channel[channel].add(user)
                    else:
                        users_per_channel[channel] = set()
                        users_per_channel[channel].add(user)

    for k in users_per_channel:
        users_per_channel[k] = str(len(users_per_channel[k]))

    with open("../../data/users_per_channel_counts.csv", "w") as f:
        csv.writer(f).writerows(x for x in users_per_channel.items())


def generate_user_chat_counts():
    user_chat_counts = {}

    file_paths = []
    for s in os.listdir("../../data/raw_data"):
        if re.match('.+\.log', s):
            file_paths.append("../../data/raw_data/" + s)

    for file_path in file_paths:
        with open(file_path) as fp:
            for line in fp:
                splits = re.split('\x1e', line)
                if len(splits) == 4:
                    user = splits[2]
                    if user in user_chat_counts:
                        user_chat_counts[user] += 1
                    else:
                        user_chat_counts[user] = 1

    with open("../../data/user_chat_counts.csv", "w") as f:
        csv.writer(f).writerows(x for x in user_chat_counts.items())


def generate_separate_channel_logs(channel_list):
    chat_dic = {}
    for channel in channel_list:
        chat_dic[channel] = []

    file_paths = []
    for s in os.listdir("../../data/raw_data"):
        if re.match('.+\.log', s):
            file_paths.append("../../data/raw_data/" + s)

    for file_path in file_paths:
        with open(file_path) as fp:
            for line in fp:
                splits = re.split('\x1e', line)
                if len(splits) == 4 and splits[1].replace('#', '') in chat_dic:
                    chat_dic[splits[1].replace('#', '')].append(line.replace('\x1e', ','))

    for channel in chat_dic:
        chat_list = chat_dic[channel]
        with open('../../data/channel_chat_logs/raw/' + channel + '_chat_log.csv', 'w') as fp:
            fp.writelines(chat_list)

        with open('../../data/channel_chat_logs/cleaned/' + channel + '_chat_log.csv', 'w') as fp:
            for chat_line in chat_list:
                splits = chat_line.split(',')
                cleaned_message = commons.clean_message(splits[3])
                if len(cleaned_message) > 0:
                    fp.write(splits[0] + "," + splits[1] + "," + splits[2] + "," + cleaned_message + "\n")


if __name__ == "__main__":
    channels = [line.strip().split(',')[0] for line in open("../female_channels.csv", 'r')] \
                + [line.strip().split(',')[0] for line in open("../male_channels.csv", 'r')]
    generate_separate_channel_logs(channels)
    generate_user_chat_counts()
    generate_users_per_channel_counts()