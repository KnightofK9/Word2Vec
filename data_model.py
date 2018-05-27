import glob
import numbers
import operator
import random
import numpy as np
import pandas as pd
import os.path

from matplotlib.mlab import PCA
import matplotlib.pyplot as plt

import preprocessor
import utilities
from datamodel import IterSentences
from serializer import JsonClassSerialize


class Saver:
    def __init__(self, save_folder_path = None):
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

    def save_word_embedding(self, word_embedding, reversed_dictionary, path=None):
        list_embedding = word_embedding.tolist()
        if path is None:
            path = self.get_word_embedding_path()
        with open(path, "w") as file:
            for index in range(0, len(list_embedding)):
                word = reversed_dictionary[str(index)]
                if word == "UNK":
                    continue
                embedding = list_embedding[index]
                line = [word] + embedding
                file.write(" ".join(map(str, line)) + "\n")

    def load_word_embedding(self, word_mapper, path=None):
        dictionary = word_mapper.dictionary
        if path is None:
            path = self.get_word_embedding_path()
        dictionary_length = len(dictionary)
        np_embedding = None
        with open(path, "r") as file:
            line = file.readline().split(" ")
            if np_embedding is None:
                embedding_size = len(line) - 1
                np_embedding = np.ndarray(shape=(dictionary_length, embedding_size), dtype=np.int32)
            np_embedding[dictionary[line[0]], :] = line[1:]
        return WordEmbedding(np_embedding, word_mapper)

    def restore_config(self, data_model, path=None):
        if path is None:
            path = self.get_config_path()
        data_model.config = self.serializer.load(path)

    def load_config(self, config_path):
        config = self.serializer.load(config_path)
        self.save_folder_path = config.save_folder_path
        return config

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
    def __init__(self, word_count):
        self.word_count = word_count
        self.word_count_length = len(self.word_count)

    def get_vocab(self, min_count = 5):
        sorted_x = sorted(self.word_count.items(), key=operator.itemgetter(1))
        sorted_x = list(reversed(sorted_x))
        sorted_x = list(filter(lambda x: x[1] >= min_count, sorted_x))
        # sorted_x = sorted_x[:max_vocab_size - 1]
        count = [['UNK', -1]]
        count.extend(list(sorted_x))

        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reversed_dictionary = dict(zip(map(str, dictionary.values()), dictionary.keys()))

        return WordMapper(dictionary, reversed_dictionary)

    def draw_histogram(self):
        bins = [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400]
        plt.hist(self.word_count.values(),bins)
        plt.show()




class WordEmbedding(object):
    def __init__(self, np_final_embedding, word_mapper):
        self.embedding = np_final_embedding
        self.word_mapper = word_mapper

    def similar_by(self, word, top_k=8):
        dictionary = self.word_mapper.dictionary
        reversed_dictionary = self.word_mapper.reversed_dictionary

        norm = np.sqrt(np.sum(np.square(self.embedding), 1))
        norm = np.reshape(norm, (len(dictionary), 1))
        normalized_embeddings = self.embedding / norm
        valid_embeddings = normalized_embeddings[dictionary[word]]
        similarity = np.matmul(
            valid_embeddings, np.transpose(normalized_embeddings), )

        nearest = (-similarity[:]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % word
        for k in range(top_k):
            close_word = reversed_dictionary[str(nearest[k])]
            log_str = '%s %s,' % (log_str, close_word)
        return log_str

    def draw(self):
        embeddings = self.embedding
        reversed_dictionary = self.word_mapper.reversed_dictionary
        words_np = []
        words_label = []
        for i in range(0, len(embeddings)):
            words_np.append(embeddings[i])
            words_label.append(reversed_dictionary[i])

        pca = PCA(n_components=2)
        pca.fit(words_np)
        reduced = pca.transform(words_np)

        plt.rcParams["figure.figsize"] = (20, 20)
        for index, vec in enumerate(reduced):
            if index < 1000:
                x, y = vec[0], vec[1]
                plt.scatter(x, y)
                plt.annotate(words_label[index], xy=(x, y))
        plt.show()


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
        self.use_preprocessor = True

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

    def get_visualization_path(self):
        return os.path.join(self.save_folder_path, "tensorboard")


class WordMapper(object):
    def __init__(self, dictionary, reversed_dictionary):
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary

    def word_to_id(self, word):
        if word in self.dictionary:
            return self.dictionary.get(word)
        else:
            return self.dictionary.get("UNK")

    def get_len(self):
        return len(self.dictionary)


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
                        if self.config.use_preprocessor:
                            data = preprocessor.split_preprocessor_row_to_word(row)
                        else:
                            data = preprocessor.split_row_to_word(row)
                        data_length = len(data)
                        # print(row)
                        for word_index in range(self.progress.word_index, data_length):
                            self.progress.word_index = word_index
                            word = data[word_index]
                            front_skip = skip_window if word_index - skip_window >= 0 else 0
                            end_skip = skip_window if word_index + skip_window <= data_length - 1 else data_length - (
                                    word_index + skip_window)
                            # all_context_index_array = list(range(word_index - front_skip + 1, word_index)) + list(
                            #     range(word_index + 1, word_index + end_skip))
                            all_context_index_array = list(range(word_index - front_skip , word_index)) + list(
                                range(word_index + 1, word_index + end_skip + 1))

                            for context_index in random.sample(all_context_index_array,
                                                               num_skips if num_skips < len(
                                                                   all_context_index_array) else len(
                                                                   all_context_index_array)):
                                context = data[context_index]
                                word_batch[batch_count] = word_mapper.word_to_id(word)
                                context_batch[batch_count] = word_mapper.word_to_id(context)
                                # print("({},{})".format(word,context))
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
    def __init__(self, csv_folder_path, use_preprocessor=True):
        self.csv_list = glob.glob(csv_folder_path)
        self.use_preprocessor = use_preprocessor

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
                    if self.use_preprocessor:
                        data = preprocessor.split_preprocessor_row_to_word(row)
                    else:
                        data = preprocessor.split_row_to_word(row)
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


def build_word_count(csv_folder_path,use_preprocessor):
    data_model = SimpleDataModel(csv_folder_path,use_preprocessor)
    dict_count = {}
    for one_gram in data_model:
        for word in one_gram:
            if word in dict_count:
                dict_count[word] += 1
            else:
                dict_count[word] = 1
    word_count_len = len(dict_count)
    print("word count len {}".format(word_count_len))
    return WordCount(dict_count)
