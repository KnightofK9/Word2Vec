import glob
import numbers
import operator
import random
import numpy as np
import pandas as pd
import os.path
import math

from matplotlib.mlab import PCA
import matplotlib.pyplot as plt

import preprocessor
import utilities
from serializer import JsonClassSerialize
import collections


class Saver:
    def __init__(self, save_folder_path=None):
        self.serializer = JsonClassSerialize()
        self.save_folder_path = save_folder_path
        pass

    def get_config_path(self):
        return os.path.join(self.save_folder_path, "short_data_config.json")

    def get_progress_path(self):
        return os.path.join(self.save_folder_path, "progress.json")

    def get_word_mapper_path(self):
        return os.path.join(self.save_folder_path, "word_mapper.json")

    def get_word_embedding_path(self):
        return os.path.join(self.save_folder_path, "word_embedding.vec")

    def get_doc_embedding_path(self):
        return os.path.join(self.save_folder_path, "doc_embedding.vec")

    def get_doc_mapper_path(self):
        return os.path.join(self.save_folder_path, "doc_mapper.json")

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
        with open(path, "w", encoding='utf-8') as file:
            for index in range(0, len(list_embedding)):
                word = reversed_dictionary[str(index)]
                if word == "UNK":
                    continue
                embedding = list_embedding[index]
                line = [word] + embedding
                file.write(" ".join(map(str, line)) + "\n")

    def save_doc_embedding(self, np_doc_embedding, reversed_dictionary, path=None):
        list_embedding = np_doc_embedding.tolist()
        if path is None:
            path = self.get_doc_embedding_path()
        with open(path, "w", encoding='utf-8') as file:
            for index in range(0, len(list_embedding)):
                word = reversed_dictionary[str(index)][0]
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

    def load_doc_embedding(self, doc_mapper, path):
        dictionary = doc_mapper.doc_mapper
        dictionary_length = len(dictionary)
        np_embedding = None
        array = []
        with open(path, "r") as file:
            for line in file:
                array.append(line.split(" "))
        array = np.array(array)
        for ele in array:
            ele[0] = int(dictionary[str(ele[0])])
        array = array.astype(np.float64)
        np_embedding = array[array[:, 0].argsort()]
        np_embedding = np_embedding[:, 1:]
        return DocEmbedding(np_embedding, doc_mapper)

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

    def init_progress(self, progress_data_model, config):
        csv_folder_path = config.csv_folder_path
        empty_progress = Progress()
        if (config.mode == "word2vec" or config.mode == "doc2vec") and config.model == "cbow":
            empty_progress.word_index = config.get_start_word_index()
        empty_progress.build_csv_list(csv_folder_path)
        progress_data_model.progress = empty_progress

    def restore_word_mapper(self, data_model, path=None):
        if path is None:
            path = self.get_word_mapper_path()
        data_model.word_mapper = self.serializer.load(path)

    def save_doc_mapper(self, doc_mapper, path=None):
        if path is None:
            path = self.get_doc_mapper_path()
        self.serializer.save(doc_mapper, path)


class CategoryMapper(object):
    def __init__(self):
        self.dictionary = None
        self.reversed_dictionary = None
        self.length = None

    def build_mapper(self, csv_folder_path):
        count_mapper = {}
        for csv_path in glob.glob(csv_folder_path):
            df = pd.read_csv(csv_path, sep=',', header=0, encoding="utf8", usecols=["catId"])
            unique_cat_id_list = df.catId.unique()
            for unique_id in unique_cat_id_list:
                if unique_id not in count_mapper:
                    count_mapper[unique_id] = 1

        dictionary = {}
        for unique_id in count_mapper.keys():
            dictionary[str(unique_id)] = str(len(dictionary))

        self.dictionary = dictionary
        self.reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        self.length = len(dictionary)


class DocMapper(object):
    def __init__(self):
        self.doc_mapper = None
        self.reversed_doc_mapper = None  # [[0]:post_id,[1]:csv_path,[2], line_number]
        self.total_doc = None

    def build_mapper(self, csv_folder_path):
        mapper = {}
        count = 0
        for csv_path in glob.glob(csv_folder_path):
            df = pd.read_csv(csv_path, sep=',', header=0, encoding="utf8", usecols=["id"])
            for index, row in df.iterrows():
                line_number = index + 1
                id = row['id']
                mapper[str(count)] = [str(id), csv_path, line_number]
                count += 1
        self.reversed_doc_mapper = mapper
        self.total_doc = count
        self.doc_mapper = dict(zip(map(str, [x[0] for x in mapper.values()]), mapper.keys()))

    def set_doc_mapper(self, doc_mapper):
        self.doc_mapper = doc_mapper
        self.total_doc = len(doc_mapper)
        self.reversed_doc_mapper = dict(zip(doc_mapper.values(), doc_mapper.keys()))

    def id_to_doc(self, idx):
        return self.reversed_doc_mapper[str(idx)][0]

    def doc_to_id(self, doc_id):
        return self.doc_mapper[doc_id]

class WordCount(object):
    def __init__(self, word_count):
        self.word_count = word_count
        self.word_count_length = len(self.word_count)

    def get_vocab_by_min_count(self, min_count=5):
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

    def get_vocab_by_size(self, vocabulary_size):
        sorted_x = sorted(self.word_count.items(), key=operator.itemgetter(1))
        sorted_x = list(reversed(sorted_x))
        sorted_x = sorted_x[:vocabulary_size - 1]
        count = [['UNK', -1]]
        count.extend(list(sorted_x))

        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reversed_dictionary = dict(zip(map(str, dictionary.values()), dictionary.keys()))

        return WordMapper(dictionary, reversed_dictionary)

    def draw_histogram(self):
        bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
        plt.hist(self.word_count.values(), bins)
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


class DocEmbedding(object):
    def __init__(self, np_final_embedding, doc_mapper):
        self.embedding = np_final_embedding
        self.doc_mapper = doc_mapper

    def similar_by(self, org_idx, top_k=8):
        dictionary = self.doc_mapper.doc_mapper
        reversed_dictionary = self.doc_mapper.reversed_doc_mapper
        idx = dictionary[org_idx]
        norm = np.sqrt(np.sum(np.square(self.embedding), 1))
        norm = np.reshape(norm, (len(dictionary), 1))
        normalized_embeddings = self.embedding / norm
        valid_embeddings = normalized_embeddings[int(idx)]
        similarity = np.matmul(
            valid_embeddings, np.transpose(normalized_embeddings), )
        sort_similarity = (-similarity[:])
        nearest = sort_similarity.argsort()[1:top_k + 1]
        # org_idx, org_title, org_content = utilities.read_csv_by_index_post(reversed_dictionary[str(idx)])
        org_idx, org_title, org_content = utilities.read_csv_by_index_post(reversed_dictionary[str(idx)])
        log_str = "_________________\nNearst to doc:\n{}\n--------------\n".format(
            self.format_doc(org_idx, org_title, org_content))
        for k in range(top_k):
            close_doc_mapper = reversed_dictionary[str(nearest[k])]
            similarity_percent = utilities.format_percentage(-sort_similarity[nearest[k]])
            close_idx, close_title, close_content = utilities.read_csv_by_index_post(close_doc_mapper)
            log_str += "{0}\n{1}\n--------------\n".format(similarity_percent,
                                                                self.format_doc(close_idx, close_title, close_content))
        return log_str

    def similar_by_embedding(self, query, query_embedding, top_k=8):
        dictionary = self.doc_mapper.doc_mapper
        reversed_dictionary = self.doc_mapper.reversed_doc_mapper
        idx = self.embedding.shape[0]
        new_embedding = np.concatenate((self.embedding, [query_embedding]))
        norm = np.sqrt(np.sum(np.square(new_embedding), 1))
        norm = np.reshape(norm, (len(dictionary) + 1, 1))
        normalized_embeddings = new_embedding / norm
        valid_embeddings = normalized_embeddings[int(idx)]
        similarity = np.matmul(
            valid_embeddings, np.transpose(normalized_embeddings), )
        sort_similarity = (-similarity[:])
        nearest = sort_similarity.argsort()[1:top_k + 1]
        # org_idx, org_title, org_content = utilities.read_csv_by_index_post(reversed_dictionary[str(idx)])
        log_str = "_________________\nNearst to query: {}\n--------------\n".format(
            query)
        for k in range(top_k):
            close_doc_mapper = reversed_dictionary[str(nearest[k])]
            similarity_percent = utilities.format_percentage(-sort_similarity[nearest[k]])
            close_idx, close_title, close_content = utilities.read_csv_by_index_post(close_doc_mapper)
            log_str += "{0}\n{1}\n--------------\n".format(similarity_percent,
                                                                self.format_doc(close_idx, close_title, close_content))
        return log_str

    def format_doc(self, org_idx, org_title, org_content):
        return "Id: {}, title: {}\n".format(org_idx, org_title)

    def draw(self):
        embeddings = self.embedding
        reversed_dictionary = self.doc_mapper.reversed_dictionary
        words_np = []
        words_label = []
        for i in range(0, len(embeddings)):
            words_np.append(embeddings[i])
            words_label.append(reversed_dictionary[i][0])

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


class ConfigFactory:
    @staticmethod
    def generate_config(save_folder_path, csv_folder_path, train_model, train_mode):
        config = Config()
        config.csv_folder_path = csv_folder_path
        config.save_folder_path = save_folder_path
        if train_model == "cbow" and train_mode == "doc2vec":
            config.use_lt_window_only = True
            config.skip_window = 3
        if train_model == "cbow" and train_mode == "word2vec":
            config.use_lt_window_only = False
            config.skip_window = 1
        return config

    @staticmethod
    def generate_cnn_config(save_folder_path, csv_folder_path):
        config = CNNConfig()
        config.csv_folder_path = csv_folder_path
        config.save_folder_path = save_folder_path
        return config


class CNNConfig(object):
    def __init__(self):
        self.batch_size = 50
        self.epoch_size = 1
        self.csv_folder_path = None
        self.save_folder_path = None
        self.save_model_name = "train_model"
        self.save_every_iteration = 10000
        self.embedding_size = 300
        self.learning_rate = 1.0
        self.kernel_size = [3, 4, 5]
        self.sequence_length = 35
        self.num_filters = 128
        self.dropout_keep_prob = 0.5
        self.l2_reg_lambda = 0.0  # L2 regularization lambda (default: 0.0)
        self.model = "cnn"
        self.mode = "docrelevant"

    def get_save_model_path(self):
        return os.path.join(self.save_folder_path, self.save_model_name)

    def get_visualization_path(self):
        return os.path.join(self.save_folder_path, "tensorboard")


class Config(object):
    def __init__(self):
        self.batch_size = 128
        self.epoch_size = 1
        # self.vocabulary_size = 10000 # Move vocabulary size to word_mapper
        self.csv_folder_path = None
        self.save_folder_path = None
        self.save_model_name = "train_model"
        self.num_skips = 2  # How many times to reuse an input to generate a label.
        self.skip_window = 2  # How many words to consider left and right.
        self.save_every_iteration = 10000
        self.embedding_size = 300
        self.doc_embedding_size = 100
        self.use_preprocessor = True
        self.model = "skipgram"
        self.mode = "word2vec"
        self.learning_rate = 1.0
        self.use_lt_window_only = False
        self.use_lt_window_only = False

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.num_sampled = 64  # Number of negative examples to sample.

    def get_start_word_index(self):
        return self.skip_window

    def generate_valid_examples(self):
        valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        return valid_examples

    def get_valid_examples(self, dictionary):
        examples = ["xây_dựng", "hay", "giảm", "tuổi", "trung_quốc", "việt_nam", "tỷ", "người", "bạn", "nói", "công_ty",
                    "hà_nội", "với", "tốt", "mua", "trường"]
        valid_examples = []
        for example in examples:
            if example in dictionary:
                valid_examples.append(dictionary[example])
        return valid_examples

    def get_save_model_path(self):
        return os.path.join(self.save_folder_path, self.save_model_name)

    def get_visualization_path(self):
        return os.path.join(self.save_folder_path, "tensorboard")

    def is_skipgram(self):
        return self.model == "skipgram"

    def is_cbow(self):
        return self.model == "cbow"

    def is_doc2vec(self):
        return self.mode == "doc2vec"

    def is_word2vec(self):
        return self.mode == "word2vec"

    def get_span_size(self):
        if self.use_lt_window_only:
            return self.skip_window + 1
        return self.skip_window * 2 + 1
        # if self.is_cbow():
        #     if self.is_doc2vec():
        #         return self.skip_window + 1
        #     return self.skip_window * 2 + 1
        # if self.is_skipgram():
        #     return self.skip_window * 2 + 1

    def get_train_input_size(self):
        assert self.is_skipgram() is False
        doc_size_increment = 1 if self.is_doc2vec() else 0
        if self.use_lt_window_only:
            return self.skip_window + doc_size_increment
        return self.skip_window * 2 + doc_size_increment
        # if self.is_doc2vec():
        #     return self.skip_window + 1
        # else:
        #     return self.skip_window * 2


class WordMapper(object):
    def __init__(self, dictionary, reversed_dictionary):
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary
        self.total_word = len(dictionary)

    def word_to_id(self, word):
        if word in self.dictionary:
            return self.dictionary.get(word)
        else:
            return self.dictionary.get("UNK")

    def id_to_word(self, wid):
        return self.reversed_dictionary[str(wid)]

    def get_vocabulary_size(self):
        return self.total_word


class ProgressDataModelCbow:
    def __init__(self):
        self.config = None
        self.progress = None
        self.word_mapper = None
        self.doc_mapper = None

    def set_doc_mapper_data(self, doc_mapper):
        self.doc_mapper = doc_mapper

    def __iter__(self):
        is_doc2vec = self.config.is_doc2vec()
        batch_size = self.config.batch_size
        skip_window = self.config.skip_window
        epoch_size = self.config.epoch_size
        csv_list = self.progress.csv_list
        word_mapper = self.word_mapper
        doc_dictionary = None
        span_size = self.config.get_span_size()
        window_size = self.config.skip_window
        start_word_index = self.config.get_start_word_index()

        front_skip = skip_window

        if is_doc2vec:
            doc_dictionary = self.doc_mapper.doc_mapper
            end_skip = skip_window
        else:
            end_skip = 0
        word_batch, context_batch = init_cbow_batch(batch_size, skip_window + 1)
        batch_count = 0
        # if self.progress.word_index < skip_window:
        #     self.progress.word_index = skip_window

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
                    if is_doc2vec:
                        idx = df["id"].tolist()[0]
                        idx = doc_dictionary[str(idx)]
                    if row_list is None:
                        continue
                    for row_index in range(self.progress.current_row_index, len(row_list)):
                        self.progress.current_row_index = row_index
                        row = row_list[row_index]
                        if len(row) == 0:
                            continue
                        if self.config.use_preprocessor:
                            data = preprocessor.split_preprocessor_row_to_word_v2(row)
                        else:
                            data = preprocessor.split_row_to_word(row)
                        data = list(map(word_mapper.word_to_id, data))
                        data_length = len(data)

                        # print(row)
                        word_index = self.progress.word_index  # Don't use progress word

                        if data_length < span_size:
                            self.progress.word_index = start_word_index
                            continue

                        array = utilities.sub_array_hard(data, word_index, front_skip, end_skip)
                        if array is None:
                            self.progress.word_index = self.config.get_span_size()
                            continue
                        deque = collections.deque(array, maxlen=span_size)

                        while word_index < data_length:
                            input_array = [token for idx, token in enumerate(deque) if idx != skip_window]

                            if is_doc2vec:
                                word_batch[batch_count] = input_array + [idx]
                            else:
                                word_batch[batch_count] = input_array
                            context = deque[skip_window]
                            context_batch[batch_count] = context
                            batch_count += 1
                            if batch_count == batch_size:
                                self.progress.increase_iteration()
                                yield (word_batch, context_batch)
                                word_batch, context_batch = init_cbow_batch(batch_size, skip_window + 1)
                                batch_count = 0
                            word_index += 1
                            self.progress.word_index = word_index
                            if word_index + end_skip < data_length:
                                deque.append(data[word_index + end_skip])
                            else:
                                break

                        self.progress.word_index = start_word_index
                    self.progress.current_row_index = 0
                self.progress.current_post_index = 0
            self.progress.current_csv_index = 0
        self.progress.current_epoch = 0


class ProgressDataModelSkipgram:
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
                    id = df["id"].to_string()
                    if row_list is None:
                        continue
                    for row_index in range(self.progress.current_row_index, len(row_list)):
                        self.progress.current_row_index = row_index
                        row = row_list[row_index]
                        if len(row) == 0:
                            continue
                        if self.config.use_preprocessor:
                            data = preprocessor.split_preprocessor_row_to_word_v2(row)
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
                            all_context_index_array = list(range(word_index - front_skip, word_index)) + list(
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

    def set_doc_mapper_data(self, doc_mapper):
        pass


class ProgressDataModelDocRele:
    def __init__(self):
        self.config = None
        self.progress = None
        self.word_mapper = None
        self.category_mapper = None

    def set_category_mapper(self, category_mapper):
        self.category_mapper = category_mapper

    def __iter__(self):
        batch_size = self.config.batch_size
        epoch_size = self.config.epoch_size
        csv_list = self.progress.csv_list
        word_mapper = self.word_mapper
        category_mapper = self.category_mapper
        sequence_length = self.config.sequence_length
        word_batch, context_batch = init_cnn_batch(batch_size, sequence_length)
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
                    id = df.id.tolist()[0]
                    title = df.title.tolist()[0]
                    tags = df.tags.tolist()[0]
                    catId = df.catId.tolist()[0]
                    # print("{} & {}".format(title, tags))
                    train_word = preprocessor.split_preprocessor_title_to_word(title)
                    if isinstance(tags, str):
                        train_word += preprocessor.split_tag_to_word(tags)
                    for idx, word in enumerate(train_word):
                        if idx >= sequence_length:
                            break
                        word_batch[batch_count][idx] = word_mapper.word_to_id(word)
                    context_batch[batch_count] = category_mapper.dictionary[str(catId)]
                    batch_count += 1
                    if batch_count == batch_size:
                        self.progress.increase_iteration()
                        yield (word_batch, context_batch)
                        word_batch, context_batch = init_cnn_batch(batch_size, sequence_length)
                        batch_count = 0

                self.progress.current_post_index = 0
            self.progress.current_csv_index = 0
        self.progress.current_epoch = 0

    def set_doc_mapper_data(self, doc_mapper):
        pass


class DataModelFactory:
    @staticmethod
    def generate_data_model(config):
        if config.mode == "word2vec" or config.mode == "doc2vec":
            if config.is_cbow():
                train_data = ProgressDataModelCbow()
            else:
                train_data = ProgressDataModelSkipgram()
        elif config.mode == "docrelevant":
            train_data = ProgressDataModelDocRele()
        else:
            raise Exception("Not supported {}".format(config.mode))
        train_data.config = config
        return train_data


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
                        data = preprocessor.split_preprocessor_row_to_word_v2(row)
                    else:
                        data = preprocessor.split_row_to_word(row)
                    yield data


class SimpleBatchModel:
    def __init__(self, config, word_mapper, text,predict_epoch_size, doc_id, use_preprocessor=True):
        self.predict_epoch_size = predict_epoch_size
        self.config = config
        self.word_mapper = word_mapper
        self.doc_id = doc_id
        self.text = text
        self.use_preprocessor = use_preprocessor

    def __iter__(self):
        batch_size = self.config.batch_size
        skip_window = self.config.skip_window
        word_mapper = self.word_mapper
        span_size = self.config.get_span_size()
        start_word_index = self.config.get_start_word_index()

        front_skip = skip_window
        end_skip = skip_window
        word_batch, context_batch = init_cbow_batch(batch_size, skip_window + 1)
        batch_count = 0

        for epoch in range(self.predict_epoch_size):
            idx = self.doc_id
            for row in self.text.split("."):
                if len(row) == 0:
                    continue
                if self.use_preprocessor:
                    data = preprocessor.split_preprocessor_row_to_word_v2(row)
                else:
                    data = preprocessor.split_row_to_word(row)
                data = list(map(word_mapper.word_to_id, data))
                data_length = len(data)

                # print(row)
                word_index = start_word_index  # Don't use progress word

                if data_length < span_size:
                    continue

                array = utilities.sub_array_hard(data, word_index, front_skip, end_skip)
                if array is None:
                    continue
                deque = collections.deque(array, maxlen=span_size)

                while word_index < data_length:
                    input_array = [token for idx, token in enumerate(deque) if idx != skip_window]
                    word_batch[batch_count] = input_array + [idx]
                    context = deque[skip_window]
                    context_batch[batch_count] = context
                    batch_count += 1
                    if batch_count == batch_size:
                        yield (word_batch, context_batch)
                        word_batch, context_batch = init_cbow_batch(batch_size, skip_window + 1)
                        batch_count = 0
                    word_index += 1
                    if word_index + end_skip < data_length:
                        deque.append(data[word_index + end_skip])
                    else:
                        break


def init_batch(batch_size):
    word_batch = np.ndarray(shape=batch_size, dtype=np.int32)
    context_batch = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    return (word_batch, context_batch)


def init_cnn_batch(batch_size, sequence_length):
    word_batch = np.zeros(shape=(batch_size, sequence_length), dtype=np.int32)
    context_batch = np.zeros(shape=(batch_size, 1), dtype=np.int32)
    return (word_batch, context_batch)


def init_cbow_batch(batch_size, input_size):
    word_batch = np.ndarray(shape=(batch_size, input_size), dtype=np.int32)
    context_batch = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    return (word_batch, context_batch)


def get_row_list_from_df(df):
    row_list = []
    if "content" in df.columns:
        post = df["content"].tolist()
        if not (len(post) == 0 or isinstance(post[0], numbers.Number)):
            post = post[0]
            row_list = row_list + post.split(".")
    if "title" in df.columns:
        title = df["title"].tolist()
        if not (len(title) == 0 or isinstance(title[0], numbers.Number)):
            title = title[0]
            row_list = row_list + [title]
    if "tags" in df.columns:
        tags = df["tags"].tolist()
        if not (len(tags) == 0 or isinstance(tags[0], numbers.Number)):
            tags = " ".join(preprocessor.split_tag_to_word(tags[0]))
            row_list = row_list + [tags]
    if len(row_list) == 0:
        return None
    return row_list


def build_word_count(csv_folder_path, use_preprocessor):
    data_model = SimpleDataModel(csv_folder_path, use_preprocessor)
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
