import os

import argparse

import data_model
from data_model import Config, Saver, ProgressDataModel, Progress
from serializer import JsonClassSerialize
from tf_word2vec import Tf_Word2Vec
import utilities

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Word2Vec training tool')

parser.add_argument('-train', action='store_true',
                    default=False,
                    dest='is_training',
                    help='Start training')
parser.add_argument('-create-embedding', action='store_true',
                    default=False,
                    dest='is_create_embedding',
                    help='Create embedding from training model')

parser.add_argument('-create-mapper', action='store_true',
                    default=False,
                    dest='is_create_mapper',
                    help='Create word_mapper from list of csv')
parser.add_argument('-create-word-count', action='store_true',
                    default=False,
                    dest='is_create_word_count',
                    help='Create word_count from list of csv')
parser.add_argument('-create-config', action='store_true',
                    default=False,
                    dest='is_create_config',
                    help='Create config file from list of csv folder')
parser.add_argument('-csv-folder-path', action='store',
                    dest='csv_folder_path',
                    default=None,
                    help='Path to csv folder. Eg: ./data/*csv')
parser.add_argument('-word-count-path', action='store',
                    dest='word_count_path',
                    default=None,
                    help='Path to word_count.json')
parser.add_argument('-vocabulary-size', action='store',
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
parser.add_argument('-config-path', action='store',
                    dest='config_path',
                    default=None,
                    help='Set config path for training!')
parser.add_argument('-use-preprocessor', action='store_true',
                    dest='use_preprocessor',
                    default=False,
                    help='Should use preprocessor when extract word for building word_count. When training, use config!')
parser.add_argument('-CUDA_VISIBLE_DEVICES', action='store',
                    dest='CUDA_VISIBLE_DEVICES',
                    default="0",
                    help='Set cuda visible device')

results = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = results.CUDA_VISIBLE_DEVICES

seri = JsonClassSerialize()


def build_word_count(save_folder_path, csv_folder_path, use_preprocessor):
    word_count = data_model.build_word_count(csv_folder_path, use_preprocessor)
    seri.save(word_count, os.path.join(save_folder_path, "word_count.json"))
    return word_count


def main():
    if results.is_create_word_count:
        build_word_count(results.save_folder_path, results.csv_folder_path, results.use_preprocessor)
        return

    if results.is_create_mapper:
        print("Creating mapper!")
        if results.csv_folder_path is not None:
            print("Creating word_count.json from csv folder {}".format(results.csv_folder_path))
            word_count = build_word_count(results.save_folder_path, results.csv_folder_path, results.use_preprocessor)
        else:
            assert results.word_count_path is not None
            print("Loading word_count.json from {}".format(results.word_count_path))
            word_count = seri.load(results.word_count_path)
        word_mapper = word_count.get_vocab(int(results.vocabulary_size))
        seri.save(word_mapper, os.path.join(results.save_folder_path, "word_mapper.json"))
        return
    if results.is_create_config:
        print("Creating config!")
        build_config(results.save_folder_path, results.csv_folder_path)
        return

    # save_folder_path = results.save_folder_path
    config_path = results.config_path
    word_mapper_path = results.mapper_path
    train_data = ProgressDataModel()
    train_data_saver = Saver()

    assert utilities.exists(config_path)
    train_data.config = train_data_saver.load_config(config_path)

    assert utilities.exists(word_mapper_path)
    train_data_saver.restore_word_mapper(train_data, word_mapper_path)

    if utilities.exists(train_data_saver.get_progress_path()):
        train_data_saver.restore_progress(train_data, train_data_saver.get_progress_path())
    else:
        train_data_saver.init_progress(train_data, train_data.config.csv_folder_path)

    word2vec = Tf_Word2Vec()
    word2vec.set_train_data(train_data, train_data_saver)
    word2vec.restore_last_training_if_exists()

    if results.is_create_embedding:
        assert (utilities.exists(train_data_saver.get_progress_path()))
        print("Creating word embedding from {}".format(train_data.config.save_folder_path))
        train_data_saver.save_word_embedding(word2vec.final_embeddings,
                                             train_data.word_mapper.reversed_dictionary)
        return

    if results.is_training:
        word2vec.train()
    else:
        word_embedding = word2vec.get_word_embedding()
        print(word_embedding.similar_by("người"))
        print(word_embedding.similar_by("anh"))
        print(word_embedding.similar_by("xã"))


def build_config(save_folder_path, csv_folder_path):
    config = Config()
    config.csv_folder_path = csv_folder_path
    config.save_folder_path = save_folder_path
    seri.save(config, os.path.join(save_folder_path, "config.json"))


if __name__ == "__main__":
    main()
    # build_vocab("./temp/", "./data/longdata/*.csv", 10000)
    # build_config( "./temp/shortdata/", "./data/shortdata/*.csv")
