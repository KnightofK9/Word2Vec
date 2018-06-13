import operator
import os
import pickle
import datetime
import json
import pandas as pd

import preprocessor


def save_simple_object(obj, path):
    with open(path, 'wb') as stream:  # Overwrites any existing file.
        pickle.dump(obj, stream, pickle.HIGHEST_PROTOCOL)


def load_simple_object(path):
    with open(path, 'rb') as stream:
        data = pickle.load(stream)
        return data


def save_json_object(obj, path):
    with open(path, 'w') as outfile:
        json.dump(obj, outfile)


def load_json_object(path):
    with open(path, 'r') as infile:
        return json.load(infile)


def save_string(str, path):
    with open(path, "w") as text_file:
        text_file.write(str)


def load_string(path):
    with open(path, "r") as text_file:
        return text_file.read()


def print_current_datetime():
    print(datetime.datetime.now())


def exists(saved_vocabulary_path):
    return os.path.exists(saved_vocabulary_path)


def build_vocab(data_model, max_vocab_size):
    dict_count = {}
    for one_gram in data_model:
        for word in one_gram:
            if word in dict_count:
                dict_count[word] += 1
            else:
                dict_count[word] = 0
    sorted_x = sorted(dict_count.items(), key=operator.itemgetter(1))
    sorted_x = list(reversed(sorted_x))
    word_count_len = len(sorted_x)
    print("word count len {}".format(word_count_len))
    assert (max_vocab_size < word_count_len)
    print("creating dictionary with len {}".format(max_vocab_size))
    sorted_x = sorted_x[:max_vocab_size - 1]
    count = [['UNK', -1]]
    count.extend(list(sorted_x))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for one_gram in data_model:
        for word in one_gram:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)

    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return (count, dictionary, reversed_dictionary)


def sub_array_soft(array, index, front_skip, end_skip):
    front_skip = max(0, index - front_skip)
    end_skip = max(len(array) - 1, index + end_skip)
    return array[front_skip:end_skip + 1]


def sub_array_hard(array, index, front_skip, end_skip):
    front_skip = index - front_skip
    end_skip = index + end_skip
    if front_skip < 0 or end_skip >= len(array):
        return None
    return array[front_skip:end_skip + 1]


def read_csv_by_index_post(reversed_mapper):
    org_idx = reversed_mapper[0]
    csv_path = reversed_mapper[1]
    line_number = reversed_mapper[2]
    return read_csv_at(org_idx, csv_path, line_number)


def read_csv_at(org_idx, csv_path, line_number):
    post_idx, title, tags, content, catId = extract_info_from_csv(org_idx, csv_path, line_number)
    show_result = None
    if content is not None:
        show_result = content
    elif tags is not None:
        show_result = tags
    return post_idx, title, show_result


def extract_info_from_csv(org_idx, csv_path, line_number):
    accepted_line = [0, line_number]
    # df = pd.read_csv(csv_path, nrows=2, sep=',', header=0, encoding="utf8", usecols=["id", "title", "content"],
    df = pd.read_csv(csv_path, nrows=2, sep=',', header=0, encoding="utf8", usecols=["id", "title", "tags"],
                     skiprows=lambda x: x not in accepted_line)
    title = df["title"].tolist()[0]
    post_idx = df["id"].tolist()[0]
    tags = None
    content = None
    catId = None
    if 'tags' in df.columns:
        tags = df["tags"].tolist()[0]
    if 'content' in df.columns:
        content = df["content"].tolist()[0]
    if 'catId' in df.columns:
        catId = df["catId"].tolist()[0]
    assert post_idx == int(org_idx)
    return post_idx, title, tags, content, catId


def extract_query_from_csv(org_idx, csv_path, line_number):
    post_idx, title, tags = read_csv_at(org_idx, csv_path, line_number)
    return preprocessor.get_query_word_from_title_and_tags(title, tags)
