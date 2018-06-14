import glob
import math
import os

import datetime as dt

import numpy as np
import pandas as pd
import tensorflow as tf

import preprocessor
from NNVar import *

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import utilities
from data_model import WordEmbedding, DocEmbedding, DocMapper, init_cnn_batch


class BaseTf:
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
        pass

    def restore_last_training_if_exists(self):
        if self.train_data.progress.finish:
            iteration = None
        else:
            iteration = self.train_data.progress.current_iteration
            if iteration == 0:
                return
        self.load_model_at_iteration(iteration)

    def init_session(self, graph):
        pass

    def set_train_data(self, train_data, train_data_saver):
        self.train_data = train_data
        self.train_data_saver = train_data_saver
        self.init_graph()
        self.init_session(self.graph)

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

    def save_progress_by_iteration(self, iteration):
        config = self.train_data.config
        save_model_path = config.get_save_model_path()

        utilities.print_current_datetime()
        print("Saving iteration no {}".format(iteration))
        self.save_model(save_model_path, iteration)
        self.train_data_saver.save_progress(self.train_data)

    def save_finish_progress(self):
        save_model_path = self.train_data.config.get_save_model_path()
        self.save_model(save_model_path)
        self.train_data.progress.set_finish()
        self.train_data_saver.save_progress(self.train_data)

    def print_evaluation(self):
        pass

    def train(self):
        pass

    def save_model(self, path, global_step=None):
        save_path = self.model_saver.save(self.session, path, global_step=global_step)
        print("Model saved in path: %s" % save_path)

    def load_model(self, path):
        self.model_saver.restore(self.session, path)


class Tf_DocRele(BaseTf):
    def __init__(self):
        super().__init__()
        self.doc_embedding = None

    def init_graph(self):
        nn_config = self.train_data.config
        word_mapper = self.train_data.word_mapper
        category_mapper = self.train_data.category_mapper
        sequence_length = nn_config.sequence_length
        num_classes = category_mapper.length
        vocab_size = word_mapper.total_word
        embedding_size = nn_config.embedding_size
        filter_sizes = nn_config.kernel_size
        num_filters = nn_config.num_filters
        l2_reg_lambda = nn_config.l2_reg_lambda

        # Placeholders for input, output and dropout
        train_inputs = tf.placeholder(tf.int32, [None, sequence_length], name="train_inputs")
        train_context = tf.placeholder(tf.int32, [None, 1], name="train_context")
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        context_one_hot = tf.one_hot(train_context, num_classes)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="W")
        embedded_chars = tf.nn.embedding_lookup(W, train_inputs)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            predictions = tf.argmax(scores, 1, name="predictions", output_type=tf.int32)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=context_one_hot)
            loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, train_context)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)

        init = tf.global_variables_initializer()

        self.nn_var = CNN_Var()
        self.nn_var.train_inputs = train_inputs
        self.nn_var.train_context = train_context
        self.nn_var.dropout_keep_prob = dropout_keep_prob
        self.nn_var.loss = loss
        self.nn_var.accuracy = accuracy
        self.nn_var.correct_predictions = correct_predictions
        self.nn_var.h_pool_flat = h_pool_flat
        self.nn_var.train_op = train_op
        self.nn_var.init = init

        self.model_saver = tf.train.Saver()

    def train(self):

        config = self.train_data.config
        save_every_iteration = config.save_every_iteration
        config_drop_out = config.dropout_keep_prob

        train_inputs = self.nn_var.train_inputs
        train_context = self.nn_var.train_context
        dropout_keep_prob = self.nn_var.dropout_keep_prob
        loss = self.nn_var.loss
        accuracy = self.nn_var.accuracy
        train_op = self.nn_var.train_op

        nce_start_time = dt.datetime.now()
        session = self.session
        # We must initialize all variables before we use them.

        print('Initialized')

        average_loss = 0
        for (batch_inputs, batch_context) in self.train_data:
            feed_dict = {train_inputs: batch_inputs, train_context: batch_context, dropout_keep_prob: config_drop_out}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val, accuracy_val = session.run([train_op, loss, accuracy], feed_dict=feed_dict)
            average_loss = loss_val
            iteration = self.train_data.progress.current_iteration
            if iteration & 1000 == 0:
                print("Step {} - Loss {} Accuracy {}".format(iteration, loss, accuracy))
            if save_every_iteration and iteration % save_every_iteration == 0:
                self.save_progress_by_iteration(iteration)
                print("Average Loss at {} : {}".format(iteration, average_loss))
                # self.print_evaluation()

        self.save_finish_progress()
        # self.print_evaluation()
        nce_end_time = dt.datetime.now()
        print(
            "NCE method took {} seconds to run 100 iterations".format((nce_end_time - nce_start_time).total_seconds()))

        self.get_doc_embedding()

    def save_progress_by_iteration(self, iteration):
        super().save_progress_by_iteration(iteration)

    def save_finish_progress(self):
        super().save_finish_progress()
        self.get_doc_embedding()
        self.train_data_saver.save_doc_embedding(self.doc_embedding.embedding,
                                                 self.doc_embedding.doc_mapper.reversed_doc_mapper)
        self.train_data_saver.save_doc_mapper(self.doc_embedding.doc_mapper)

    def print_evaluation(self):
        if self.doc_embedding is None:
            doc_embedding = self.get_doc_embedding()
            self.doc_embedding = doc_embedding
        else:
            doc_embedding = self.doc_embedding
        doc_embedding.similar_by(doc_embedding.doc_mapper.reversed_doc_mapper("0"))

    def restore_last_training_if_exists(self):
        super().restore_last_training_if_exists()
        if utilities.exists(self.train_data_saver.get_doc_embedding_path()) and utilities.exists(
                self.train_data_saver.get_doc_mapper_path()):
            print("Doc embedding and doc mapper found at {}. Loading".format(self.train_data_saver.save_folder_path))
            doc_mapper = self.train_data_saver.serializer.load(self.train_data_saver.get_doc_mapper_path())
            doc_embedding = self.train_data_saver.load_doc_embedding(doc_mapper,
                                                                     self.train_data_saver.get_doc_embedding_path())
            self.doc_embedding = doc_embedding

    def get_doc_embedding(self, csv_folder_path=None):
        if csv_folder_path is None:
            csv_folder_path = self.train_data.config.csv_folder_path

        np_doc_embedding = []

        mapper = {}
        count = 0
        for csv_path in glob.glob(csv_folder_path):
            df = pd.read_csv(csv_path, sep=',', header=0, encoding="utf8", usecols=["id", "title", "tags"])
            for index, row in df.iterrows():
                line_number = index + 1
                id = row['id']
                mapper[str(count)] = [str(id), csv_path, line_number]
                count += 1

                title = row["title"]
                tags = row["tags"]
                train_word = preprocessor.get_train_word_from_title_and_tags(title, tags)
                feature_vector = self.get_query_embedding(train_word)
                np_doc_embedding.append(feature_vector)

        reversed_doc_mapper = mapper
        total_doc = count
        doc_mapper_dict = dict(zip(map(str, [x[0] for x in mapper.values()]), mapper.keys()))

        doc_mapper = DocMapper()
        doc_mapper.doc_mapper = doc_mapper_dict
        doc_mapper.reversed_doc_mapper = reversed_doc_mapper
        doc_mapper.total_doc = total_doc

        doc_embedding = DocEmbedding(np.asarray(np_doc_embedding), doc_mapper)

        self.doc_embedding = doc_embedding
        return doc_embedding

    def get_query_embedding(self, train_word):
        train_inputs = self.nn_var.train_inputs
        train_context = self.nn_var.train_context
        dropout_keep_prob = self.nn_var.dropout_keep_prob
        h_pool_flat = self.nn_var.h_pool_flat
        word_mapper = self.train_data.word_mapper
        session = self.session

        sequence_length = self.train_data.config.sequence_length
        word_batch, context_batch = init_cnn_batch(1, sequence_length)
        for idx, word in enumerate(train_word):
            if idx >= sequence_length:
                break
            word_batch[0][idx] = word_mapper.word_to_id(word)
        feed_dict = {train_inputs: word_batch, train_context: context_batch, dropout_keep_prob: 1.0}
        feature_vector = session.run([h_pool_flat], feed_dict=feed_dict)
        feature_vector = feature_vector[0][0]
        return feature_vector

    def get_query_prediction(self, train_word):
        train_inputs = self.nn_var.train_inputs
        train_context = self.nn_var.train_context
        dropout_keep_prob = self.nn_var.dropout_keep_prob
        correct_predictions = self.nn_var.correct_predictions
        word_mapper = self.train_data.word_mapper
        session = self.session

        sequence_length = self.train_data.config.sequence_length
        word_batch, context_batch = init_cnn_batch(1, sequence_length)
        for idx, word in enumerate(train_word):
            if idx >= sequence_length:
                break
            word_batch[0][idx] = word_mapper.word_to_id(word)
        feed_dict = {train_inputs: word_batch, train_context: context_batch, dropout_keep_prob: 1.0}
        correct_predictions = session.run([correct_predictions], feed_dict=feed_dict)
        correct_predictions = correct_predictions[0][0]
        return correct_predictions

    def init_session(self, graph):
        init = self.nn_var.init
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(graph=graph, config=tf_config)
        self.session = session
        init.run(session=session)
        return session

    def retrieve_by_query(self, query_list):
        result = ""
        for query in query_list:
            processor_query = preprocessor.preprocess_row(query)
            query_embedding = self.get_query_embedding(preprocessor.split_query_to_train_word(processor_query))
            result += self.doc_embedding.similar_by_embedding(processor_query, query_embedding)

        return result


class Tf_Word2VecBase(BaseTf):
    def __init__(self):
        super().__init__()

    def load_model(self, path):
        super().load_model(path)
        self.build_word_embedding()

    def init_session(self, graph):
        init = self.nn_var.init
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True
        session = tf.Session(graph=graph, config=tf_config)
        self.session = session
        init.run(session=session)
        return session

    def build_word_embedding(self):
        normalized_embeddings = self.nn_var.normalized_embeddings
        self.final_embeddings = normalized_embeddings.eval(session=self.session)

    def get_word_embedding(self):
        return WordEmbedding(self.final_embeddings, self.train_data.word_mapper)

    def save_progress_by_iteration(self, iteration):
        super().save_progress_by_iteration(iteration)
        self.build_word_embedding()
        self.save_word_embedding()

    def save_finish_progress(self):
        super().save_finish_progress()
        self.build_word_embedding()
        self.save_word_embedding()

    def save_word_embedding(self):
        self.train_data_saver.save_word_embedding(self.final_embeddings,
                                                  self.train_data.word_mapper.reversed_dictionary)

    def print_evaluation(self):
        similarity = self.nn_var.similarity
        valid_examples = self.nn_var.valid_examples
        reversed_dictionary = self.train_data.word_mapper.reversed_dictionary
        session = self.session

        sim = similarity.eval(session=session)
        for i in range(len(valid_examples)):
            valid_word = reversed_dictionary[str(valid_examples[i])]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reversed_dictionary[str(nearest[k])]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    def train(self):

        train_inputs = self.nn_var.train_inputs
        train_context = self.nn_var.train_context
        optimizer = self.nn_var.optimizer
        nce_loss = self.nn_var.nce_loss
        config = self.train_data.config
        save_every_iteration = config.save_every_iteration

        nce_start_time = dt.datetime.now()
        session = self.session
        # We must initialize all variables before we use them.

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
                self.save_progress_by_iteration(iteration)
                self.print_evaluation()

        self.save_finish_progress()
        nce_end_time = dt.datetime.now()
        print(
            "NCE method took {} seconds to run 100 iterations".format((nce_end_time - nce_start_time).total_seconds()))


class Tf_CBOWWord2Vec(Tf_Word2VecBase):
    def __init__(self):
        super().__init__()

    def init_graph(self):
        assert self.train_data is not None
        config = self.train_data.config
        vocabulary_size = self.train_data.word_mapper.get_vocabulary_size()
        batch_size = config.batch_size
        embedding_size = config.embedding_size  # Dimension of the embedding vector.
        window_size = config.skip_window
        model_learning_rate = config.learning_rate
        train_input_size = config.get_train_input_size()

        # valid_examples = config.generate_valid_examples()
        valid_examples = config.get_valid_examples(self.train_data.word_mapper.dictionary)
        num_sampled = config.num_sampled  # Number of negative examples to sample.
        graph = tf.Graph()

        self.graph = graph

        with graph.as_default():
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size, train_input_size])
            train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            reduced_embed = tf.div(tf.reduce_sum(embed, 1), window_size * 2)
            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=reduced_embed, labels=train_context,
                               num_sampled=num_sampled, num_classes=vocabulary_size))
            # Create optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(nce_loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True), name="norm")
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True, name="similarity")

            # Add variable initializer.
            init = tf.global_variables_initializer()

            self.nn_var = NNVar()
            self.nn_var.train_inputs = train_inputs
            self.nn_var.train_context = train_context
            self.nn_var.valid_dataset = valid_dataset
            self.nn_var.embeddings = embeddings
            self.nn_var.nce_loss = nce_loss
            self.nn_var.optimizer = optimizer
            self.nn_var.normalized_embeddings = normalized_embeddings
            self.nn_var.similarity = similarity
            self.nn_var.init = init
            self.nn_var.valid_examples = valid_examples
            self.nn_var.doc_embeddings = None
            self.model_saver = tf.train.Saver()
            # self.writer = tf.summary.FileWriter(self.train_data.config.get_visualization_path(), graph)


class Tf_SkipgramWord2Vec(Tf_Word2VecBase):
    def __init__(self):
        super().__init__()

    def init_graph(self):
        assert self.train_data is not None
        config = self.train_data.config
        vocabulary_size = self.train_data.word_mapper.get_vocabulary_size()
        batch_size = config.batch_size
        model_learning_rate = config.learning_rate
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
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embeddings")
            embed = tf.nn.embedding_lookup(embeddings, train_inputs, name="embed")

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)), name="nce_weights")
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")

            nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_context,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size), name="nce_loss")

            optimizer = tf.train.GradientDescentOptimizer(model_learning_rate).minimize(nce_loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True), name="norm")
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True, name="similarity")

            # Add variable initializer.
            init = tf.global_variables_initializer()

            self.nn_var = NNVar()
            self.nn_var.train_inputs = train_inputs
            self.nn_var.train_context = train_context
            self.nn_var.valid_dataset = valid_dataset
            self.nn_var.embeddings = embeddings
            self.nn_var.nce_loss = nce_loss
            self.nn_var.optimizer = optimizer
            self.nn_var.normalized_embeddings = normalized_embeddings
            self.nn_var.similarity = similarity
            self.nn_var.init = init
            self.nn_var.valid_examples = valid_examples
            self.nn_var.doc_embeddings = None
            self.model_saver = tf.train.Saver()
            # self.writer = tf.summary.FileWriter(self.train_data.config.get_visualization_path(), graph)


class Tf_Doc2VecBase(Tf_Word2VecBase):
    def __init__(self):
        super().__init__()

    def save_progress_by_iteration(self, iteration):
        super().save_progress_by_iteration(iteration)
        self.save_doc_embedding()

    def save_doc_embedding(self):
        self.train_data_saver.save_doc_embedding(self.build_doc_embedding(),
                                                 self.train_data.doc_mapper.reversed_doc_mapper)

    def build_doc_embedding(self):
        doc_embeddings = self.nn_var.doc_embeddings
        return doc_embeddings.eval(session=self.session)

    def get_doc_embedding(self):
        return DocEmbedding(self.build_doc_embedding(), self.train_data.doc_mapper)

    def save_finish_progress(self):
        super().save_finish_progress()
        self.save_doc_embedding()


class Tf_CBOWDoc2Vec(Tf_Doc2VecBase):
    def __init__(self):
        super().__init__()

    def init_graph(self):
        assert self.train_data is not None
        config = self.train_data.config
        doc_mapper = self.train_data.doc_mapper
        vocabulary_size = self.train_data.word_mapper.get_vocabulary_size()
        batch_size = config.batch_size
        embedding_size = config.embedding_size  # Dimension of the embedding vector.
        doc_embedding_size = config.doc_embedding_size
        total_doc = doc_mapper.total_doc
        window_size = config.skip_window
        concatenated_size = embedding_size + doc_embedding_size
        model_learning_rate = config.learning_rate
        train_input_size = config.get_train_input_size()

        # valid_examples = config.generate_valid_examples()
        valid_examples = config.get_valid_examples(self.train_data.word_mapper.dictionary)
        num_sampled = config.num_sampled  # Number of negative examples to sample.
        graph = tf.Graph()

        self.graph = graph

        with graph.as_default():
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[None, train_input_size], name="train_inputs")
            train_context = tf.placeholder(tf.int32, shape=[None, 1], name="train_context")
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32, name="valid_dataset")

            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embeddings")
            doc_embeddings = tf.Variable(tf.random_uniform([total_doc, doc_embedding_size], -1.0, 1.0))

            # NCE loss parameters
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size],
                                                          stddev=1.0 / np.sqrt(concatenated_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            embed = tf.zeros([batch_size, embedding_size])
            for element in range(window_size):
                embed += tf.nn.embedding_lookup(embeddings, train_inputs[:, element])

            doc_indices = tf.slice(train_inputs, [0, window_size], [batch_size, 1])
            doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)

            # concatenate embeddings
            final_embed = tf.concat(axis=1, values=[embed, tf.squeeze(doc_embed)])

            nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                     biases=nce_biases,
                                                     labels=train_context,
                                                     inputs=final_embed,
                                                     num_sampled=num_sampled,
                                                     num_classes=vocabulary_size))

            # Create optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(nce_loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True), name="norm")
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True, name="similarity")

            # Add variable initializer.
            init = tf.global_variables_initializer()

            self.nn_var = NNVar()
            self.nn_var.train_inputs = train_inputs
            self.nn_var.train_context = train_context
            self.nn_var.valid_dataset = valid_dataset
            self.nn_var.embeddings = embeddings
            self.nn_var.nce_loss = nce_loss
            self.nn_var.optimizer = optimizer
            self.nn_var.normalized_embeddings = normalized_embeddings
            self.nn_var.similarity = similarity
            self.nn_var.init = init
            self.nn_var.valid_examples = valid_examples
            self.nn_var.doc_embeddings = doc_embeddings
            self.model_saver = tf.train.Saver()
            # self.writer = tf.summary.FileWriter(self.train_data.config.get_visualization_path(), graph)


class NetworkFactory:
    @staticmethod
    def generate_network(config):
        if config.mode == "docrelevant":
            return Tf_DocRele()
        if config.is_doc2vec() and config.is_cbow():
            return Tf_CBOWDoc2Vec()
        if config.is_word2vec() and config.is_cbow():
            return Tf_CBOWWord2Vec()
        if config.is_word2vec() and config.is_skipgram():
            return Tf_SkipgramWord2Vec()
        raise Exception("Not supported")
