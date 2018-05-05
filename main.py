import os

from gensim_word2vec import Gensim_Word2Vec
from tf_word2vec import Tf_Word2Vec
import utilities

if __name__ == "__main__":
    model_path = "./temp/save_model_tf"
    saved_vocabulary = "./temp/save_vocab"
    # csv_path = "./data/2017-news-1.1-1.9_Process.csv"
    csv_path = "./data/news10k.csv"
    # word2vec = Gensim_Word2Vec() # Currently not working!!
    word2vec = Tf_Word2Vec()
    if not os.path.exists(model_path):
        if os.path.exists(saved_vocabulary):
            print("Saved vocab found, loading vocab file {}".format(saved_vocabulary))
            vocab = utilities.load_simple_object(saved_vocabulary)
            word2vec.load_vocab(vocab)
        else:
            print("Loading word2vec data at {}".format(csv_path))
            word2vec.load_data(csv_path, preload=True)
            utilities.save_simple_object(word2vec.train_data,saved_vocabulary)
        word2vec.train(iteration=1)
        print("Training completed, saving word2vec model at {}".format(model_path))
        word2vec.save_model(model_path)
    else:
        print("Data found! Loading saved model {}".format(model_path))
        word2vec.load_model(model_path)
    word2vec.draw()
    print(word2vec.similar_by("ông"))
