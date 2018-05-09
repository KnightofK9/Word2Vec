import os

# from gensim_word2vec import Gensim_Word2Vec
from tf_word2vec import Tf_Word2Vec
import utilities

start_training = False
# start_training = True


def main():
    main_path = "./temp/datafolder"
    model_path = main_path + "/save_model_tf"
    model_iteration = 3000000
    saved_vocabulary = main_path + "/save_vocab"
    saved_vocabulary_with_out_text = main_path + "/save_vocab_without_text"
    preload = False
    csv_path = "./data/datafolder/*.csv"
    # word2vec = Gensim_Word2Vec() # Currently not working!!
    word2vec = Tf_Word2Vec(save_path=model_path, save_every_iteration=500000)

    if start_training:
        if os.path.exists(saved_vocabulary):
            print("Saved vocab found, loading vocab file {}".format(saved_vocabulary))
            vocab = utilities.load_simple_object(saved_vocabulary)
            word2vec.load_vocab(vocab)
        else:
            print("Loading word2vec data at {}".format(csv_path))
            word2vec.load_data(csv_path,is_folder_path=True, preload=preload)
            utilities.save_simple_object(word2vec.train_data, saved_vocabulary)

        # word2vec.load_model("{}-{}".format(model_path, model_iteration))
        word2vec.load_data_if_exists(iteration=500000)
        word2vec.train(num_steps=1)
        word2vec.save_model(model_path)
        word2vec.train_data.drop_train_text()
        utilities.save_simple_object(word2vec.train_data, saved_vocabulary_with_out_text)
    else:
        assert (os.path.exists(saved_vocabulary_with_out_text))
        vocab = utilities.load_simple_object(saved_vocabulary_with_out_text)
        word2vec.load_vocab(vocab)
        word2vec.load_data_if_exists(iteration=1000000)

    # word2vec.draw()
    print(word2vec.similar_by("người",20))
    print(word2vec.similar_by("học",20))
    print(word2vec.similar_by("ngủ",20))


def build_simple_vocab(saved_vocabulary, saved_vocabulary_with_out_text):
    vocab = utilities.load_simple_object(saved_vocabulary)
    vocab.drop_train_text()
    utilities.save_simple_object(vocab, saved_vocabulary_with_out_text)


if __name__ == "__main__":
    # saved_vocabulary = "./temp/save_vocab"
    # saved_vocabulary_with_out_text = "./temp/save_vocab_without_text"
    # build_simple_vocab(saved_vocabulary,saved_vocabulary_with_out_text)
    main()
