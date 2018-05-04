import collections
import operator
import random

import pandas as pd
import re
from nltk import ngrams
import numpy as np


class Data_Model:
    def __init__(self, csv_path, chunk_size=1):
        self.chunk_size = chunk_size
        self.csv_path = csv_path

    def __iter__(self):
        for df in pd.read_csv(self.csv_path, sep=',', header=0, chunksize=self.chunk_size, encoding="utf8"):
            post = df["content"].tolist()
            if len(post) == 0:
                continue
            post = post[0]
            for row in post.split("."):
                if len(row) == 0:
                    continue
                transformed = transform_row(row)
                one_gram = kieu_ngram(transformed, 1)
                yield one_gram
                # two_gram = kieu_ngram(transformed, 2)
                # yield two_gram


class Iter_Batch_Data_Model:
    def __init__(self, csv_path, max_vocab_size=10000, batch_size=128, num_skip=2, skip_window=2, chunk_size=1):
        self.chunk_size = chunk_size
        self.data_model = Data_Model(csv_path)
        self.batch_size = batch_size
        self.num_skip = num_skip
        self.max_vocab_size = max_vocab_size
        self.skip_window = skip_window

        (count, dictionary, reversed_dictionary) = self.build_vocabulary()
        self.count = count
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary

    def build_vocabulary(self):
        dict_count = {}
        for one_gram in self.data_model:
            for word in one_gram:
                if word in dict_count:
                    dict_count[word] += 1
                else:
                    dict_count[word] = 0
        sorted_x = sorted(dict_count.items(), key=operator.itemgetter(1))
        sorted_x = sorted_x.reverse()[:self.max_vocab_size - 1]
        count = [['UNK', -1]]
        count.extend(list(sorted_x))

        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for one_gram in self.data_model:
            for word in one_gram:
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                    unk_count += 1
                data.append(index)

        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return count, dictionary, reversed_dictionary

    def __iter__(self):
        batch = []
        contexts = []

        span = 2 * self.skip_window + 1
        for one_gram in self.data_model:
            iterable = Iter_Sentences(one_gram, self.num_skip, self.skip_window)
            for (word, context) in iterable:
                batch.append(word)
                contexts.append(context)
                if len(batch) == self.batch_size:
                    yield (batch, context)
                    batch = []
                    contexts = []

        yield (batch, contexts)


class Iter_Sentences:
    def __init__(self, sentences, num_skips, skip_window):
        self.sentences = sentences
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.data_index = 0

    def __iter__(self):
        data = self.sentences
        data_index = self.data_index
        skip_window = self.skip_window
        num_skips = self.num_skips

        span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(0, len(data) // span):
            target = skip_window  # input word at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                word = buffer[skip_window]  # this is the input word
                context = buffer[target]  # these are the context words
                yield (word, context)

            buffer.append(data[data_index])
            data_index = (data_index + 1)


def transform_row(row):
    # row = row.encode("utf-8")
    # Xóa số dòng ở đầu câu
    row = row.lower()

    row = re.sub(r"^[0-9\.]+", "", row)

    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$", "", row)

    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ")

    row = row.strip()
    return row


def kieu_ngram(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [" ".join(gram).lower() for gram in gram_str]
