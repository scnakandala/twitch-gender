import csv
import re


def count_words(file_paths):
    result_dict = {}
    for file_path in file_paths:
        with open(file_path) as fp:
            for line in fp:
                splits = re.split(',', line)
                if len(splits) == 4:
                    streamer = splits[1]
                    commenter = splits[2]
                    message = splits[3]
                    if commenter != streamer:
                        tokens = message.split()
                        for i in range(0, len(tokens)):
                            token = tokens[i]
                            try:
                                result_dict[token] += 1
                            except KeyError:
                                result_dict[token] = 1
    return result_dict


def count_bigrams(file_paths):
    result_dict = {}
    for file_path in file_paths:
        with open(file_path, 'r') as fp:
            for line in fp:
                splits = re.split(',', line)
                if len(splits) == 4:
                    streamer = splits[1]
                    commenter = splits[2]
                    message = splits[3]
                    if commenter != streamer:
                        tokens = message.split()
                        for itr in range(0, len(tokens) - 1):
                            token1 = tokens[itr]
                            token2 = tokens[itr + 1]
                            token = token1 + " " + token2
                            try:
                                result_dict[token] += 1
                            except KeyError:
                                result_dict[token] = 1
    return result_dict


def generate_token_counts():
    gender_category_types = ['female', 'male']

    for gender_category in gender_category_types:

        all_names = [line.strip().split(',')[0] for line in open("../" + gender_category + "_channels.csv", 'r')]

        divided_names = {'top_' + gender_category : all_names[0:100], 'bottom_' + gender_category : all_names[100:200]}
        count_types = ['bigram', 'word']

        for n in divided_names:
            file_names = divided_names[n]
            for count_type in count_types:
                final_result = {}
                for file_name in file_names:
                    if count_type == 'bigram':
                        result = count_bigrams(["../../data/channel_chat_logs/cleaned/" + file_name + "_chat_log.csv"])
                    else:
                        result = count_words(["../../data/channel_chat_logs/cleaned/" + file_name + "_chat_log.csv"])
                    sorted_dic = ((k, result[k]) for k in sorted(result, key=result.get, reverse=True))
                    for key, value in sorted_dic:
                        if int(value) < 5:
                            break
                        try:
                            final_result[key] += int(value)
                        except KeyError:
                            final_result[key] = int(value)
                    sorted_dic = ((k, result[k]) for k in sorted(result, key=result.get, reverse=True))
                    with open("../../data/word_counts/channel_"+count_type+"_counts/"+ file_name + "_"+count_type+"_counts.csv", 'w') as fp:
                        writer = csv.writer(fp, delimiter=',')
                        writer.writerows(sorted_dic)
                sorted_dic = ((k, final_result[k]) for k in sorted(final_result, key=final_result.get, reverse=True))
                with open("../../data/word_counts/" + n + "_"+count_type+"_counts.csv", 'w') as fp:
                    writer = csv.writer(fp, delimiter=',')
                    writer.writerows(sorted_dic)


if __name__ == '__main__':
    generate_token_counts()