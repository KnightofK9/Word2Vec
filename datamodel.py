import collections
import numbers
import operator
import random

import pandas as pd
import re
from nltk import ngrams
import numpy as np
import preprocessor
import math

class DataModel:
    def __init__(self, csv_path, chunk_size=1, print_percentage=False):
        self.chunk_size = chunk_size
        self.csv_path = csv_path
        self.print_percentage = print_percentage

    def __iter__(self):
        count = 0
        for df in pd.read_csv(self.csv_path, sep=',', header=0, chunksize=self.chunk_size, encoding="utf8"):
            post = df["content"].tolist()
            if len(post) == 0:
                continue
            count = count + 1
            if self.print_percentage:
                print("Reading post {}".format(count))
            post = post[0]
            for row in post.split("."):
                if len(row) == 0:
                    continue
                transformed = preprocessor.nomalize_uni_string(row)
                one_gram = preprocessor.split_row_to_word(transformed, 1)
                yield one_gram
                # two_gram = kieu_ngram(transformed, 2)
                # yield two_gram


class PreloadDataModel:
    def __init__(self, csv_path, print_percentage=False):
        self.preload_data = []
        count = 0
        df = pd.read_csv(csv_path, sep=',', header=0, encoding="utf8")
        posts = df["content"].tolist()
        self.total_sentences = 0
        self.print_percentage = print_percentage
        for post in posts:
            if isinstance(post, numbers.Number) or len(post) == 0:
                continue
            count = count + 1
            print("Reading post {}".format(count))
            for row in post.split("."):
                if len(row) == 0:
                    continue
                transformed = preprocessor.nomalize_uni_string(row)
                one_gram = preprocessor.split_row_to_word(transformed, 1)
                self.preload_data.append(one_gram)
                self.total_sentences = self.total_sentences + 1

    def __iter__(self):
        count = 0
        for one_gram in self.preload_data:
            count = count + 1
            if self.print_percentage:
                percentage = (count / (1.0 * self.total_sentences)) * 100
                print("Reading sentence {0:.02f} %".format(percentage))

            yield one_gram


class IterBatchDataModel:
    def __init__(self, csv_path, max_vocab_size=10000, batch_size=128, num_skip=2, skip_window=2, chunk_size=1,
                 preload_data=False, print_percentage = False):
        self.chunk_size = chunk_size
        if preload_data:
            self.data_model = PreloadDataModel(csv_path,print_percentage=print_percentage)
        else:
            self.data_model = DataModel(csv_path,print_percentage=print_percentage)
        self.batch_size = batch_size
        self.num_skip = num_skip
        self.max_vocab_size = max_vocab_size
        self.skip_window = skip_window

        (count, dictionary, reversed_dictionary) = self.build_vocabulary()
        self.count = count
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary

    def drop_train_text(self):
        self.data_model = None

    def build_vocabulary(self):
        dict_count = {}
        for one_gram in self.data_model:
            for word in one_gram:
                if word in dict_count:
                    dict_count[word] += 1
                else:
                    dict_count[word] = 0
        sorted_x = sorted(dict_count.items(), key=operator.itemgetter(1))
        sorted_x = list(reversed(sorted_x))[:self.max_vocab_size - 1]
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
        batch = np.ndarray(shape=self.batch_size, dtype=np.int32)
        contexts = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        count = 0
        for one_gram in self.data_model:
            iterable = IterSentences(one_gram, self.num_skip, self.skip_window)
            for (word, context) in iterable:
                batch[count] = self.word_to_id(word)
                contexts[count] = self.word_to_id(context)
                count = count + 1
                if count == self.batch_size:
                    yield (batch, contexts)
                    batch = np.ndarray(shape=self.batch_size, dtype=np.int32)
                    contexts = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
                    count = 0

    def word_to_id(self, word):
        if word in self.dictionary:
            return self.dictionary.get(word)
        else:
            return self.dictionary.get("UNK")


class IterSentences:
    def __init__(self, sentences, num_skips=2, skip_window=2):
        self.sentences = sentences
        self.num_skips = num_skips
        self.skip_window = skip_window

    def __iter__(self):
        data = self.sentences
        skip_window = self.skip_window
        num_skips = self.num_skips

        data_length = len(data)
        for word_index in range(0, data_length):
            word = data[word_index]
            front_skip = skip_window if word_index - skip_window >= 0 else 0
            end_skip = skip_window if word_index + skip_window <= data_length - 1 else data_length - (
                    word_index + skip_window)
            all_context_index_array = list(range(word_index - front_skip + 1, word_index)) + list(
                range(word_index + 1, word_index + end_skip))

            for context_index in random.sample(all_context_index_array,
                                               num_skips if num_skips < len(all_context_index_array) else len(
                                                   all_context_index_array)):
                yield word, data[context_index]
