import urllib.request
import collections
import math
import os
import random
import zipfile
import datetime as dt
import pandas as pd


import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from datamodel import DataModel, IterBatchDataModel


def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


class Tf_Word2Vec:
    def __init__(self):
        self.data_index = 0

        vocabulary_size = 10000

        batch_size = 128
        embedding_size = 300  # Dimension of the embedding vector.
        skip_window = 2  # How many words to consider left and right.
        num_skips = 2  # How many times to reuse an input to generate a label.

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        num_sampled = 64  # Number of negative examples to sample.
        self.session = None
        self.final_embeddings = None
        self.train_data = None
        graph = tf.Graph()

        with graph.as_default():
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_context,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size))

            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            init = tf.global_variables_initializer()

            self.nn_var = (
                train_inputs, train_context, valid_dataset, embeddings, nce_loss, optimizer, normalized_embeddings,
                similarity, init)
            self.saver = tf.train.Saver()


        self.var = (
            vocabulary_size, batch_size, embedding_size, skip_window,
            num_skips, valid_size, valid_window, valid_examples, num_sampled, graph)


    def load_data(self, csv_path, preload=False):
        (vocabulary_size, batch_size, embedding_size, skip_window,
         num_skips, valid_size, valid_window, valid_examples, num_sampled, graph) = self.var
        self.train_data = IterBatchDataModel(csv_path, max_vocab_size=vocabulary_size, batch_size=batch_size,
                                             num_skip=num_skips, skip_window=skip_window, preload_data=preload)

    def train(self, iteration=2):
        (vocabulary_size, batch_size, embedding_size, skip_window,
         num_skips, valid_size, valid_window, valid_examples, num_sampled, graph) = self.var

        (train_inputs, train_context, valid_dataset, embeddings, nce_loss, optimizer, normalized_embeddings, similarity,
         init) = self.nn_var

        num_steps = iteration
        nce_start_time = dt.datetime.now()

        session = tf.Session(graph=graph)
        # We must initialize all variables before we use them.
        init.run(session=session)
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            for (batch_inputs, batch_context) in self.train_data:
                feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, nce_loss], feed_dict=feed_dict)
                average_loss += loss_val

        self.final_embeddings = normalized_embeddings.eval(session=session)
        self.session = session
        nce_end_time = dt.datetime.now()
        print(
            "NCE method took {} seconds to run 100 iterations".format((nce_end_time - nce_start_time).total_seconds()))

    def save_model(self, path):
        save_path = self.saver.save(self.session, path)
        print("Model saved in path: %s" % save_path)

    def load_model(self, path):
        (vocabulary_size, batch_size, embedding_size, skip_window,
         num_skips, valid_size, valid_window, valid_examples, num_sampled, graph) = self.var
        self.session = tf.Session(graph=graph)
        self.saver.restore(self.session, path)

    def similar_by(self, word):
        dictionary = self.train_data.dictionary
        reversed_dictionary = self.train_data.reversed_dictionary

        norm = np.sqrt(np.sum(np.square(self.final_embeddings), 1))
        normalized_embeddings = self.final_embeddings / norm
        valid_embeddings = normalized_embeddings[dictionary[word]]
        similarity = np.matmul(
            valid_embeddings, np.transpose(normalized_embeddings), )

        top_k = 8  # number of nearest neighbors
        nearest = (-similarity[:]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % word
        for k in range(top_k):
            close_word = reversed_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        return log_str

    def draw(self):
        embeddings = self.final_embeddings
        reversed_dictionary = self.train_data.reversed_dictionary
        words_np = []
        words_label = []
        for embedding in embeddings:
            words_np.append(embedding)
            words_label.append(reversed_dictionary[embedding])

        pca = PCA(n_components=2)
        pca.fit(words_np)
        reduced = pca.transform(words_np)

        plt.rcParams["figure.figsize"] = (20, 20)
        for index, vec in enumerate(reduced):
            if index < 200:
                x, y = vec[0], vec[1]
                plt.scatter(x, y)
                plt.annotate(words_label[index], xy=(x, y))
        plt.show()

    def load_vocab(self, saved_vocabulary):
        self.train_data = saved_vocabulary
        pass
