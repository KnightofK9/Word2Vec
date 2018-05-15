import os

# from gensim_word2vec import Gensim_Word2Vec
from data_model import Config, Saver, build_word_mapper, ProgressDataModel, Progress
from datamodel import FolderDataModel, DataModel
from serializer import JsonClassSerialize
from tf_word2vec import Tf_Word2Vec
import utilities

# start_training = False
start_training = True
seri = JsonClassSerialize()


def main():
    save_folder_path = "./temp/shortdata"
    word_mapper_path = "./temp/word_mapper.json"
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

    if start_training:
        word2vec.train()
    else:
        print(word2vec.similar_by("người"))
        print(word2vec.similar_by("anh"))
        print(word2vec.similar_by("xã"))


def build_vocab(save_folder_path, csv_folder_path, max_vocab_size):
    word_mapper = build_word_mapper(csv_folder_path, max_vocab_size)
    seri.save(word_mapper, os.path.join(save_folder_path, "word_mapper.json"))


def build_config(csv_folder_path, save_folder_path):
    config = Config()
    config.csv_folder_path = csv_folder_path
    config.save_folder_path = save_folder_path
    seri.save(config, os.path.join(save_folder_path, "config.json"))


if __name__ == "__main__":
    main()
    # build_vocab("./temp/", "./data/shortdata/*.csv", 10000)
    # build_config("./data/shortdata/*.csv", "./temp/shortdata/")
