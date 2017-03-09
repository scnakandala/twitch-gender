""" doit script for workflow """
#!/usr/bin/env python
# encoding: utf-8
import twitch.commons as tw
from collections import defaultdict
import string
import os
import re

def task_generate_separate_channel_logs():
    """ to be implemented """

    def generate_separate_channel_logs(channel_list):
        chat_dic = defaultdict(list) 
        files = tw.get_list_raw_chatlogs('data/raw_data')
        for fname in files:
            with open(fname) as fp:
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

    return {
        'actions': [generate_separate_channel_logs],
        'file_dep': ["data/female_channels.csv",
                     "data/male_channels.csv"],
        'targets': [],
    }
