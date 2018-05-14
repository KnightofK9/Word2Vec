import os

# from gensim_word2vec import Gensim_Word2Vec
from datamodel import FolderDataModel, DataModel
from tf_word2vec import Tf_Word2Vec
import utilities

# start_training = False
start_training = True



def main():
    main_save_path = "./temp/shortdata"
    csv_path = "./data/shortdata/*.csv"
    max_vocab_size = 10000
    save_iteration = 50000
    num_steps = 5
    saved_vocabulary_path = main_save_path + "/save_vocab"
    save_data_model_path = main_save_path + "/save_data_model"
    model_path = main_save_path + "/save_model_tf"
    word2vec = Tf_Word2Vec(save_path=model_path, main_path=main_save_path, save_every_iteration=save_iteration,
                           vocabulary_size=max_vocab_size)
    if utilities.exists(save_data_model_path):
        print("Loading saved data model {}".format(save_data_model_path))
        word2vec.load_data(save_data_model_path)
    else:
        print("Init data model")
        word2vec.init_data(csv_path, is_folder_path=True)
        word2vec.save_data(save_data_model_path)

    if utilities.exists(saved_vocabulary_path):
        print("Loading saved vocab {}".format(saved_vocabulary_path))
        word2vec.load_vocab(saved_vocabulary_path)
    else:
        print("Init vocab")
        word2vec.init_vocab()
        word2vec.save_vocab(saved_vocabulary_path)

    word2vec.restore_last_training_if_exists()
    if start_training:
        word2vec.train(num_steps=num_steps)
        word2vec.save_model(model_path)
    else:
        print(word2vec.similar_by("xã"))
        print(word2vec.similar_by("%"))
        print(word2vec.similar_by("kỹ_thuật"))


def build_vocab(save_vocab_file, csv_path, max_vocab_size, is_folder_path=True):
    if is_folder_path:
        data_model = FolderDataModel(csv_path, print_percentage=False)
    else:
        data_model = DataModel(csv_path, print_percentage=False)
    (count, dictionary, reversed_dictionary) = utilities.build_vocab(data_model, max_vocab_size)
    dict_data = {
        "count": count,
        "dictionary": dictionary,
        "reversed_dictionary": reversed_dictionary
    }
    utilities.save_simple_object(dict_data, save_vocab_file)


if __name__ == "__main__":
    main()
    # build_vocab("./temp/save_vocab", "C:/dataset/vocab/*.csv", 10000)
