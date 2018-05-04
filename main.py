import os

from gensim_word2vec import Gensim_Word2Vec

if __name__ == "__main__":
    model_path = "./temp/save_model_gensim_with_iter"
    csv_path = "./data/2017-news-1.1-1.9_Process.csv"
    # csv_path = "./data/news10k.csv"
    word2vec = Gensim_Word2Vec()
    if not os.path.exists(model_path):
        print("Loading word2vec data at {}".format(csv_path))
        word2vec.load_iter_data(csv_path)
        word2vec.train(iter=1)
        print("Training completed, saving word2vec model at {}".format(model_path))
        word2vec.save_model(model_path)
    else:
        print("Data found! Loading saved model {}".format(model_path))
        word2vec.load_model(model_path)
    word2vec.draw()
    print(word2vec.similar_by("ông"))
