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
import glob

import utilities

class DataModel:
    def __init__(self, csv_path, chunk_size=1, print_percentage=False):
        self.chunk_size = chunk_size
        self.csv_path = csv_path
        self.print_percentage = print_percentage
        self.progress = {
            "current_index": 0,
            "current_row_index": 0
        }

    def __iter__(self):
        count = 0
        start_index = self.progress["current_index"]
        for df in pd.read_csv(self.csv_path, sep=',', header=0, skiprows=range(1, start_index),
                              chunksize=self.chunk_size,
                              encoding="utf8"):
            self.progress["current_index"] += 1
            post = df["content"].tolist()
            if len(post) == 0 or isinstance(post[0], numbers.Number):
                continue
            count = count + 1
            if self.print_percentage:
                print("Reading post {}".format(count))
            post = post[0]
            start_row_index = self.progress["current_row_index"]
            row_list = post.split(".")
            for i in range(start_row_index, len(row_list)):
                self.progress["current_row_index"] = i
                row = row_list[i]
                if len(row) == 0:
                    continue
                transformed = preprocessor.nomalize_uni_string(row)
                one_gram = preprocessor.split_row_to_word(transformed, 1)
                yield one_gram
                # two_gram = kieu_ngram(transformed, 2)
                # yield two_gram
            self.progress["current_row_index"] = 0

        self.progress["current_index"] = 0

    def load_progress(self, progress):
        self.progress = progress


class FolderDataModel:
    def __init__(self, csv_folder_path, chunk_size=1, print_percentage=False):
        self.chunk_size = chunk_size
        self.csv_data_model_list = []
        self.print_percentage = print_percentage
        self.progress = {
            "csv_index": 0,
            "csv_data_model_progress": {}
        }
        for csv_path in glob.glob(csv_folder_path):
            self.csv_data_model_list.append(DataModel(csv_path, chunk_size, print_percentage))

    def __iter__(self):
        start_csv_index = self.progress["csv_index"]
        for i in range(start_csv_index, len(self.csv_data_model_list)):
            self.progress["csv_index"] = i
            csv_data_model = self.csv_data_model_list[i]
            if self.print_percentage:
                print("Processing file {}".format(csv_data_model.csv_path))
            for one_gram in csv_data_model:
                self.progress["csv_data_model_progress"] = {csv_data_model.csv_path: csv_data_model.progress}
                yield one_gram
        self.progress["csv_index"] = 0

    def load_progress(self, progress):
        self.progress = progress
        for csv_data_model in self.csv_data_model_list:
            if csv_data_model.csv_path not in self.progress["csv_data_model_progress"]:
                continue
            csv_data_model_progress = self.progress["csv_data_model_progress"][csv_data_model.csv_path]
            csv_data_model.progress = csv_data_model_progress


# Deprecated
class PreloadDataModel:
    def __init__(self, csv_path, print_percentage=False):
        self.preload_data = []
        self.csv_path = csv_path
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

    def load_progress(self, progress):
        # TODO: Load progress for preload datamodel
        pass


class IterBatchDataModel:
    def __init__(self, max_vocab_size=10000, batch_size=128, num_skip=2, skip_window=2, chunk_size=1):
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.num_skip = num_skip
        self.max_vocab_size = max_vocab_size
        self.skip_window = skip_window

        self.data_model = None

        self.count = None
        self.dictionary = None
        self.reversed_dictionary = None

    def init_data_model(self, csv_path, preload_data=False, print_percentage=False, is_folder_path=False):
        if preload_data:
            self.data_model = PreloadDataModel(csv_path, print_percentage=print_percentage)
        else:
            if is_folder_path:
                self.data_model = FolderDataModel(csv_path, print_percentage=print_percentage)
            else:
                self.data_model = DataModel(csv_path, print_percentage=print_percentage)

    def get_progress(self):
        return self.data_model.progress

    def load_progress(self, progress):
        self.data_model.load_progress(progress)

    def drop_train_text(self):
        self.data_model = None

    def get_vocabulary(self):
        return {
            "count": self.count,
            "dictionary": self.dictionary,
            "reversed_dictionary": self.reversed_dictionary
        }

    def set_vocabulary(self, vocabulary_data):
        self.count = vocabulary_data["count"]
        self.dictionary = vocabulary_data["dictionary"]
        self.reversed_dictionary = vocabulary_data["reversed_dictionary"]

    def build_vocabulary(self):
        (count, dictionary, reversed_dictionary) = utilities.build_vocab(self.data_model, self.max_vocab_size)
        self.count = count
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary

    def __iter__(self):
        batch = np.ndarray(shape=self.batch_size, dtype=np.int32)
        contexts = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        count = 0
        for one_gram in self.data_model:
            # TODO: Saving iterable process here, currently ignore it
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
