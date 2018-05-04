from gensim.models import Word2Vec
import logging
import pandas as pd
import re
from nltk import ngrams
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import os

from data_model import Data_Model


class Gensim_Word2Vec:
    def __init__(self):
        self.train_data = None
        self.model = None


    def load_iter_data(self,csv_path):
        self.train_data = Data_Model(csv_path)

    def train(self, size=300, max_vocab_size=10000, window=5, min_count=5, workers=4, sg=1, iter=2):
        train_data = self.train_data
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.model = Word2Vec(train_data, max_vocab_size=max_vocab_size, size=size, window=window, min_count=min_count,
                              workers=workers, sg=sg, iter=iter)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = Word2Vec.load(path)

    def similar_by(self, word):
        return self.model.wv.similar_by_word(word)

    def draw(self):
        model = self.model

        words_np = []
        words_label = []
        for word in model.wv.vocab.keys():
            words_np.append(model.wv[word])
            words_label.append(word)

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
