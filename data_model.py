import glob
import numbers
import operator
import random
import numpy as np
import pandas as pd
import os.path

import preprocessor
import utilities
from datamodel import IterSentences
from serializer import JsonClassSerialize


class Saver:
    def __init__(self, save_folder_path):
        self.serializer = JsonClassSerialize()
        self.save_folder_path = save_folder_path
        pass

    def get_config_path(self):
        return os.path.join(self.save_folder_path, "config.json")

    def get_progress_path(self):
        return os.path.join(self.save_folder_path, "progress.json")

    def get_word_mapper_path(self):
        return os.path.join(self.save_folder_path, "word_mapper.json")

    def get_word_embedding_path(self):
        return os.path.join(self.save_folder_path, "word_embedding.vec")

    def save_config(self, data_model, path=None):
        if path is None:
            path = self.get_config_path()
        self.serializer.save(data_model.config, path)

    def save_progress(self, progress_data_model, path=None):
        if path is None:
            path = self.get_progress_path()
        self.serializer.save(progress_data_model.progress, path)

    def save_word_mapper(self, data_model, path=None):
        if path is None:
            path = self.get_word_mapper_path()
        self.serializer.save(data_model.word_mapper, path)

    def save_word_embedding(self, np_final_embedding, reversed_dictionary, path=None):
        list_embedding = np_final_embedding.tolist()
        if path is None:
            path = self.get_word_embedding_path()
        with open(path, "w") as file:
            for index in range(0, len(list_embedding)):
                word = reversed_dictionary[str(index)]
                if word == "UNK":
                    continue
                embedding = list_embedding[index]
                line = [word] + embedding
                file.write(" ".join(map(str,line)) + "\n")

        # word_embedding = WordEmbedding(final_embedding, reversed_dictionary)
        # utilities.save_string(word_embedding,path)

    def restore_config(self, data_model, path=None):
        if path is None:
            path = self.get_config_path()
        data_model.config = self.serializer.load(path)

    def restore_progress(self, progress_data_model, path=None):
        if path is None:
            path = self.get_progress_path()
        progress_data_model.progress = self.serializer.load(path)

    def init_progress(self, progress_data_model, csv_folder_path):
        empty_progress = Progress()
        empty_progress.build_csv_list(csv_folder_path)
        progress_data_model.progress = empty_progress

    def restore_word_mapper(self, data_model, path=None):
        if path is None:
            path = self.get_word_mapper_path()
        data_model.word_mapper = self.serializer.load(path)

class WordCount(object):
    def __init__(self,word_count):
        self.word_count = word_count
        self.word_count_length = len(self.word_count)

    def get_vocab(self, max_vocab_size):
        sorted_x = sorted(self.word_count.items(), key=operator.itemgetter(1))
        sorted_x = list(reversed(sorted_x))
        sorted_x = sorted_x[:max_vocab_size - 1]
        count = [['UNK', -1]]
        count.extend(list(sorted_x))

        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reversed_dictionary = dict(zip(map(str, dictionary.values()), dictionary.keys()))

        return WordMapper(dictionary, reversed_dictionary)

class WordEmbedding(object):
    def __init__(self, np_final_embedding, reversed_dictionary):
        list_embedding = np_final_embedding.tolist()
        self.word_embedding = {reversed_dictionary[str(index)]: value for (index, value) in enumerate(list_embedding)}


class Progress(object):
    def __init__(self):
        self.csv_list = []
        self.current_csv_index = 0
        self.current_post_index = 0
        self.current_row_index = 0
        self.current_iteration = 0
        self.current_epoch = 0
        self.word_index = 0
        self.finish = False

    def build_csv_list(self, csv_folder_path):
        self.csv_list = glob.glob(csv_folder_path)

    def increase_iteration(self):
        self.current_iteration += 1

    def set_finish(self):
        self.finish = True


class Config(object):
    def __init__(self):
        self.batch_size = 128
        self.epoch_size = 1
        self.vocabulary_size = 10000
        self.csv_folder_path = None
        self.save_folder_path = None
        self.save_model_name = "train_model"
        self.num_skips = 2  # How many times to reuse an input to generate a label.
        self.skip_window = 2  # How many words to consider left and right.
        self.save_every_iteration = 10000
        self.embedding_size = 300

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.num_sampled = 64  # Number of negative examples to sample.

    def generate_valid_examples(self):
        valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        return valid_examples

    def get_save_model_path(self):
        return os.path.join(self.save_folder_path, self.save_model_name)


class WordMapper(object):
    def __init__(self, dictionary, reversed_dictionary):
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary

    def word_to_id(self, word):
        if word in self.dictionary:
            return self.dictionary.get(word)
        else:
            return self.dictionary.get("UNK")


class ProgressDataModel:
    def __init__(self):
        self.config = None
        self.progress = None
        self.word_mapper = None

    def __iter__(self):
        batch_size = self.config.batch_size
        skip_window = self.config.skip_window
        num_skips = self.config.num_skips
        epoch_size = self.config.epoch_size
        csv_list = self.progress.csv_list
        word_mapper = self.word_mapper
        word_batch, context_batch = init_batch(batch_size)
        batch_count = 0
        for epoch in range(self.progress.current_epoch, epoch_size):
            self.progress.current_epoch = epoch
            for csv_index in range(self.progress.current_csv_index, len(csv_list)):
                self.progress.current_csv_index = csv_index
                csv_path = csv_list[csv_index]
                for df in pd.read_csv(csv_path, sep=',', header=0, skiprows=range(1, self.progress.current_post_index),
                                      chunksize=1,
                                      encoding="utf8"):
                    self.progress.current_post_index += 1
                    row_list = get_row_list_from_df(df)
                    if row_list is None:
                        continue
                    for row_index in range(self.progress.current_row_index, len(row_list)):
                        self.progress.current_row_index = row_index
                        row = row_list[row_index]
                        if len(row) == 0:
                            continue
                        transformed = preprocessor.nomalize_uni_string(row)
                        data = preprocessor.split_row_to_word(transformed)
                        data_length = len(data)
                        for word_index in range(self.progress.word_index, data_length):
                            self.progress.word_index = word_index
                            word = data[word_index]
                            front_skip = skip_window if word_index - skip_window >= 0 else 0
                            end_skip = skip_window if word_index + skip_window <= data_length - 1 else data_length - (
                                    word_index + skip_window)
                            all_context_index_array = list(range(word_index - front_skip + 1, word_index)) + list(
                                range(word_index + 1, word_index + end_skip))

                            for context_index in random.sample(all_context_index_array,
                                                               num_skips if num_skips < len(
                                                                   all_context_index_array) else len(
                                                                   all_context_index_array)):
                                context = data[context_index]
                                word_batch[batch_count] = word_mapper.word_to_id(word)
                                context_batch[batch_count] = word_mapper.word_to_id(context)
                                batch_count += 1
                                if batch_count == batch_size:
                                    self.progress.increase_iteration()
                                    yield (word_batch, context_batch)
                                    word_batch, context_batch = init_batch(batch_size)
                                    batch_count = 0

                        self.progress.word_index = 0
                    self.progress.current_row_index = 0
                self.progress.current_post_index = 0
            self.progress.current_csv_index = 0
        self.progress.current_epoch = 0


class SimpleDataModel:
    def __init__(self, csv_folder_path):
        self.csv_list = glob.glob(csv_folder_path)

    def __iter__(self):
        csv_list = self.csv_list
        for csv_path in csv_list:
            for df in pd.read_csv(csv_path, sep=',', header=0,
                                  chunksize=1,
                                  encoding="utf8"):
                row_list = get_row_list_from_df(df)
                if row_list is None:
                    continue
                for row in row_list:
                    transformed = preprocessor.nomalize_uni_string(row)
                    data = preprocessor.split_row_to_word(transformed)
                    yield data


def init_batch(batch_size):
    word_batch = np.ndarray(shape=batch_size, dtype=np.int32)
    context_batch = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    return (word_batch, context_batch)


def get_row_list_from_df(df):
    post = df["content"].tolist()
    if len(post) == 0 or isinstance(post[0], numbers.Number):
        return None
    post = post[0]
    row_list = post.split(".")
    return row_list


def build_word_count(csv_folder_path):
    data_model = SimpleDataModel(csv_folder_path)
    dict_count = {}
    for one_gram in data_model:
        for word in one_gram:
            if word in dict_count:
                dict_count[word] += 1
            else:
                dict_count[word] = 0
    word_count_len = len(dict_count)
    print("word count len {}".format(word_count_len))
    return WordCount(dict_count)
