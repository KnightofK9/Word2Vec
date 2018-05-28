import math
import os

import datetime as dt

import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import utilities
from data_model import WordEmbedding


class Tf_Word2Vec:
    def __init__(self):

        self.train_data = None
        self.train_data_saver = None

        self.session = None
        self.graph = None
        self.final_embeddings = None

        self.nn_var = None
        self.model_saver = None
        self.writer = None

    def init_graph(self):
        assert self.train_data is not None
        config = self.train_data.config
        vocabulary_size = config.vocabulary_size
        batch_size = config.batch_size
        embedding_size = config.embedding_size  # Dimension of the embedding vector.

        # valid_examples = config.generate_valid_examples()
        valid_examples = config.get_valid_examples(self.train_data.word_mapper.dictionary)
        num_sampled = config.num_sampled  # Number of negative examples to sample.
        graph = tf.Graph()

        self.graph = graph

        with graph.as_default():
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name="train_inputs")
            train_context = tf.placeholder(tf.int32, shape=[batch_size, 1], name="train_context")
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32, name="valid_dataset")

            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name="embeddings")
            embed = tf.nn.embedding_lookup(embeddings, train_inputs, name="embed")

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)),name="nce_weights")
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]),name="nce_biases")

            nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_context,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size),name="nce_loss")

            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True),name="norm")
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True, name="similarity")

            # Add variable initializer.
            init = tf.global_variables_initializer()

            self.nn_var = (
                train_inputs, train_context, valid_dataset, embeddings, nce_loss, optimizer, normalized_embeddings,
                similarity, init, valid_examples)
            self.model_saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.train_data.config.get_visualization_path(), graph)

    def restore_last_training_if_exists(self):
        if self.train_data.progress.finish:
            iteration = None
        else:
            iteration = self.train_data.progress.current_iteration
            if iteration == 0:
                return
        self.load_model_at_iteration(iteration)

    def set_train_data(self, train_data, train_data_saver):
        self.train_data = train_data
        self.train_data_saver = train_data_saver
        self.init_graph()

    def load_model_at_iteration(self, iteration=None):
        save_model_path = self.train_data.config.get_save_model_path()
        if iteration is None:
            path = "{}".format(save_model_path)
        else:
            path = "{}-{}".format(save_model_path, iteration)
        print("Trying to load model {}".format(path))
        assert os.path.exists(path + ".meta")
        print("Data found! Loading saved model {}".format(path))
        self.load_model(path)

    def train(self):

        (train_inputs, train_context, valid_dataset, embeddings, nce_loss, optimizer, normalized_embeddings, similarity,
         init, valid_examples) = self.nn_var
        graph = self.graph
        reversed_dictionary = self.train_data.word_mapper.reversed_dictionary
        config = self.train_data.config
        valid_size = config.valid_size
        save_every_iteration = config.save_every_iteration
        save_model_path = config.get_save_model_path()

        nce_start_time = dt.datetime.now()
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True
        session = tf.Session(graph=graph, config=tf_config)
        self.session = session
        # We must initialize all variables before we use them.
        init.run(session=session)
        print('Initialized')

        average_loss = 0
        for (batch_inputs, batch_context) in self.train_data:
            feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, nce_loss], feed_dict=feed_dict)
            average_loss += loss_val
            iteration = self.train_data.progress.current_iteration
            if save_every_iteration and iteration % save_every_iteration == 0:
                utilities.print_current_datetime()
                print("Saving iteration no {}".format(iteration))
                self.save_model(save_model_path, iteration)
                self.train_data_saver.save_progress(self.train_data)
                self.train_data_saver.save_word_embedding(normalized_embeddings.eval(session=session),
                                                          self.train_data.word_mapper.reversed_dictionary)

                sim = similarity.eval(session=session)
                for i in range(valid_size):
                    valid_word = reversed_dictionary[str(valid_examples[i])]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reversed_dictionary[str(nearest[k])]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        self.save_model(save_model_path)
        self.final_embeddings = normalized_embeddings.eval(session=session)
        self.train_data.progress.set_finish()
        self.train_data_saver.save_progress(self.train_data)
        nce_end_time = dt.datetime.now()
        print(
            "NCE method took {} seconds to run 100 iterations".format((nce_end_time - nce_start_time).total_seconds()))

    def save_model(self, path, global_step=None):
        save_path = self.model_saver.save(self.session, path, global_step=global_step)
        print("Model saved in path: %s" % save_path)

    def load_model(self, path):
        (train_inputs, train_context, valid_dataset, embeddings, nce_loss, optimizer, normalized_embeddings, similarity,
         init, valid_examples) = self.nn_var
        graph = self.graph

        self.session = tf.Session(graph=graph)
        self.model_saver.restore(self.session, path)
        self.final_embeddings = normalized_embeddings.eval(session=self.session)

    def get_word_embedding(self):
        return WordEmbedding(self.final_embeddings, self.train_data.word_mapper)
