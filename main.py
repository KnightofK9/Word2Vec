import os

# from gensim_word2vec import Gensim_Word2Vec
from datamodel import FolderDataModel, DataModel
from tf_word2vec import Tf_Word2Vec
import utilities

# start_training = False
start_training = True


def main1():
    # main_save_path = "./temp/test_resume_training"
    main_save_path = "./temp/test_resume_training_folder"
    model_path = main_save_path + "/save_model_tf"
    # csv_path = "./data/shortdata/news10k.csv"
    # is_folder_path = False
    is_folder_path = True
    csv_path = "./data/shortdata_datafolder/*.csv"
    max_vocab_size = 20000
    save_iteration = 50000
    saved_vocabulary = main_save_path + "/save_vocab"
    saved_vocabulary_with_out_text = main_save_path + "/save_vocab_without_text"
    preload = False
    # csv_path = "./data/datafolder/*.csv"
    # word2vec = Gensim_Word2Vec() # Currently not working!!
    word2vec = Tf_Word2Vec(save_path=model_path, main_path=main_save_path, save_every_iteration=save_iteration,
                           vocabulary_size=max_vocab_size)

    if start_training:
        if os.path.exists(saved_vocabulary):
            print("Saved vocab found, loading vocab file {}".format(saved_vocabulary))
            vocab = utilities.load_simple_object(saved_vocabulary)
            word2vec.load_vocab(vocab)
        else:
            print("Loading word2vec data at {}".format(csv_path))
            word2vec.load_data(csv_path, is_folder_path=is_folder_path, preload=preload)
            utilities.save_simple_object(word2vec.train_data, saved_vocabulary)

        # word2vec.load_model("{}-{}".format(model_path, model_iteration))
        # word2vec.load_model_if_exists(iteration=500000)
        word2vec.restore_last_training_if_exists()
        word2vec.train(num_steps=1)
        word2vec.save_model(model_path)
        word2vec.train_data.drop_train_text()
        utilities.save_simple_object(word2vec.train_data, saved_vocabulary_with_out_text)
    else:
        assert (os.path.exists(saved_vocabulary_with_out_text))
        vocab = utilities.load_simple_object(saved_vocabulary_with_out_text)
        word2vec.load_vocab(vocab)
        word2vec.load_model_if_exists()

    # word2vec.draw()
    # print(word2vec.similar_by("người",20))
    # print(word2vec.similar_by("học",20))
    # print(word2vec.similar_by("ngủ",20))


def main():
    main_save_path = "./temp/test_resume_training_folder"
    model_path = main_save_path + "/save_model_tf"
    csv_path = "./data/shortdata_datafolder/*.csv"
    max_vocab_size = 20000
    save_iteration = 10000
    num_steps = 1
    saved_vocabulary_path = main_save_path + "/save_vocab"
    save_data_model_path = main_save_path + "/save_data_model"
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
    # saved_vocabulary = "./temp/save_vocab"
    # saved_vocabulary_with_out_text = "./temp/save_vocab_without_text"
    # build_simple_vocab(saved_vocabulary,saved_vocabulary_with_out_text)
    main()
    # build_vocab("./temp/save_vocab", "./data/shortdata/*.csv", 20000)
