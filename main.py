import os

import argparse

from data_model import Config, Saver, build_word_mapper, ProgressDataModel, Progress
from serializer import JsonClassSerialize
from tf_word2vec import Tf_Word2Vec
import utilities

# start_training = False

parser = argparse.ArgumentParser(description='Word2Vec training tool')

parser.add_argument('-train', action='store_true',
                    default=False,
                    dest='is_training',
                    help='Start training')

parser.add_argument('-create-mapper', action='store_true',
                    default=False,
                    dest='is_create_mapper',
                    help='Create word_mapper from list of csv')
parser.add_argument('-create-config', action='store_true',
                    default=False,
                    dest='is_create_config',
                    help='Create config file from list of csv folder')
parser.add_argument('-csv-folder-path', action='store',
                    dest='csv_folder_path',
                    default=None,
                    help='Path to csv folder. Eg: ./data/*csv')
parser.add_argument('-vocabulary_size', action='store',
                    dest='vocabulary_size',
                    default=10000,
                    help='Set vocabulary size for building vocabulary')
parser.add_argument('-save-path', action='store',
                    default="./",
                    dest='save_folder_path',
                    help="Set save path for creating mapper/config or loading training config")
parser.add_argument('-mapper-path', action='store',
                    dest='mapper_path',
                    default=None,
                    help='Set word_mapper path for training!')

results = parser.parse_args()

seri = JsonClassSerialize()


def main():
    if results.is_create_mapper:
        print("Creating mapper!")
        build_vocab(results.save_path, results.csv_folder_path, results.vocabulary_size)
        return
    if results.is_create_config:
        print("Creating config!")
        build_config(results.save_path, results.csv_folder_path)
        return

    save_folder_path = results.save_folder_path
    word_mapper_path = results.mapper_path
    train_data = ProgressDataModel()
    train_data_saver = Saver(save_folder_path)

    assert train_data_saver.get_config_path()
    train_data_saver.restore_config(train_data)

    assert utilities.exists(word_mapper_path)
    train_data_saver.restore_word_mapper(train_data, word_mapper_path)

    if utilities.exists(train_data_saver.get_progress_path()):
        train_data_saver.restore_progress(train_data, train_data_saver.get_progress_path())
    else:
        train_data_saver.init_progress(train_data, train_data.config.csv_folder_path)

    word2vec = Tf_Word2Vec()
    word2vec.set_train_data(train_data, train_data_saver)
    word2vec.restore_last_training_if_exists()

    if results.is_training:
        word2vec.train()
    else:
        print(word2vec.similar_by("người"))
        print(word2vec.similar_by("anh"))
        print(word2vec.similar_by("xã"))


def build_vocab(save_folder_path, csv_folder_path, max_vocab_size):
    word_mapper = build_word_mapper(csv_folder_path, max_vocab_size)
    seri.save(word_mapper, os.path.join(save_folder_path, "word_mapper.json"))


def build_config(save_folder_path, csv_folder_path):
    config = Config()
    config.csv_folder_path = csv_folder_path
    config.save_folder_path = save_folder_path
    seri.save(config, os.path.join(save_folder_path, "config.json"))


if __name__ == "__main__":
    main()
    # build_vocab("./temp/", "./data/longdata/*.csv", 10000)
    # build_config( "./temp/shortdata/", "./data/shortdata/*.csv")
