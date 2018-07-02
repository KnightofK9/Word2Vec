import os

import argparse

import data_model
from data_model import Config, Saver, Progress
from empty_training import EmptyTraining
from serializer import JsonClassSerialize
from tf_word2vec import *
import utilities
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Word2Vec training tool')

parser.add_argument('-train-model', action='store',
                    default="skipgram",
                    dest='train_model',
                    help='Use for creating config, possible model is: cbow, skipgram')
parser.add_argument('-train-mode', action='store',
                    default="word2vec",
                    dest='train_mode',
                    help='Use for creating config, possible mode is: word2vec, doc2vec')
parser.add_argument('-train-type', action='store',
                    default=None,
                    dest='train_type',
                    help='Use for training, possible type: normal, empty')

parser.add_argument('-create-embedding', action='store_true',
                    default=False,
                    dest='is_create_embedding',
                    help='Create embedding from training model')
parser.add_argument('-create-doc-embedding', action='store_true',
                    default=False,
                    dest='is_create_doc_embedding',
                    help='Create doc embedding from training model')
parser.add_argument('-create-doc-mapper', action='store_true',
                    default=False,
                    dest='is_create_doc_mapper',
                    help='Create doc mapper')
parser.add_argument('-create-category-mapper', action='store_true',
                    default=False,
                    dest='is_create_category_mapper',
                    help='Create category mapper')

parser.add_argument('-eval-doc-embedding', action='store_true',
                    default=False,
                    dest='is_eval_doc_embedding',
                    help='Evaluate doc embedding result, require doc_embedding.json and doc_mapper.json')
parser.add_argument('-eval-doc-rele-embedding', action='store_true',
                    default=False,
                    dest='is_eval_doc_rele_embedding',
                    help='Evaluate doc relevant embedding result, require input query')
parser.add_argument('-eval-doc-rele-prediction', action='store_true',
                    default=False,
                    dest='is_eval_doc_rele_prediction',
                    help='Evaluate query prediction result')
parser.add_argument('-eval-query', action='store',
                    default=None,
                    dest='eval_query',
                    help='Query for evaluate doc relevant')

parser.add_argument('-create-word-mapper', action='store_true',
                    default=False,
                    dest='is_create_word_mapper',
                    help='Create word_mapper from list of csv')
parser.add_argument('-create-word-count', action='store_true',
                    default=False,
                    dest='is_create_word_count',
                    help='Create word_count from list of csv')
parser.add_argument('-create-config', action='store_true',
                    default=False,
                    dest='is_create_config',
                    help='Create config file from list of csv folder')
parser.add_argument('-create-cnn-config', action='store_true',
                    default=False,
                    dest='is_create_cnn_config',
                    help='Create cnn config file from list of csv folder')
parser.add_argument('-csv-folder-path', action='store',
                    dest='csv_folder_path',
                    default=None,
                    help='Path to csv folder. Eg: ./data/*csv')
parser.add_argument('-word-count-path', action='store',
                    dest='word_count_path',
                    default=None,
                    help='Path to word_count.json')
parser.add_argument('-min-word-count', action='store',
                    dest='min_word_count',
                    default=None,
                    help='Set minimum of count for building vocabulary, if set, use this instead vocabulary size')
parser.add_argument('-vocabulary-size', action='store',
                    dest='vocabulary_size',
                    default=10000,
                    help='Set vocabulary size for building vocabulary')
parser.add_argument('-save-path', action='store',
                    default=None,
                    dest='save_folder_path',
                    help="Set save path for creating mapper/config or loading training config")
parser.add_argument('-doc-embedding-path', action='store',
                    default=None,
                    dest='doc_embedding_path',
                    help="Set doc embedding path for evaluating")
parser.add_argument('-word-mapper-path', action='store',
                    dest='mapper_path',
                    default=None,
                    help='Set word_mapper path for training!')
parser.add_argument('-doc-mapper-path', action='store',
                    dest='doc_mapper_path',
                    default=None,
                    help='Set doc_mapper path for training!')
parser.add_argument('-category-mapper-path', action='store',
                    dest='category_mapper_path',
                    default=None,
                    help='Set category_mapper_path path for training!')
parser.add_argument('-config-path', action='store',
                    dest='config_path',
                    default=None,
                    help='Set config path for training!')
parser.add_argument('-use-preprocessor', action='store_true',
                    dest='use_preprocessor',
                    default=True,
                    help='Should use preprocessor when extract word for building word_count. When training, use config!')
parser.add_argument('-CUDA_VISIBLE_DEVICES', action='store',
                    dest='CUDA_VISIBLE_DEVICES',
                    default="0",
                    help='Set cuda visible device')
parser.add_argument('-use-cpu', action='store_true',
                    dest='is_use_cpu',
                    default=False,
                    help='Set use CPU instead of GPU')

results = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = results.CUDA_VISIBLE_DEVICES

seri = JsonClassSerialize()


def build_word_count(save_folder_path, csv_folder_path, use_preprocessor):
    assert csv_folder_path is not None
    word_count = data_model.build_word_count(csv_folder_path, use_preprocessor)
    seri.save(word_count, os.path.join(save_folder_path, "word_count.json"))
    return word_count


def build_doc_mapper(save_path, csv_folder_path):
    doc_mapper = data_model.DocMapper()
    doc_mapper.build_mapper(csv_folder_path)
    seri.save(doc_mapper, os.path.join(save_path,"doc_mapper.json"))

def build_category_mapper(save_path, csv_folder_path):
    category_mapper = data_model.CategoryMapper()
    category_mapper.build_mapper(csv_folder_path)
    seri.save(category_mapper, os.path.join(save_path,"category_mapper.json"))

def main():

    train_data_saver = Saver()
    if results.is_create_cnn_config:
        config = data_model.ConfigFactory.generate_cnn_config(results.save_folder_path, results.csv_folder_path)
        seri.save(config,os.path.join(results.save_folder_path,"cnn_config.json"))
        return
    if results.is_create_category_mapper:
        build_category_mapper(results.save_folder_path, results.csv_folder_path)
        return
    if results.is_create_word_count:
        build_word_count(results.save_folder_path, results.csv_folder_path, results.use_preprocessor)
        return
    if results.is_create_doc_mapper:
        build_doc_mapper(results.save_folder_path, results.csv_folder_path)
        return
    if results.is_eval_doc_embedding:
        assert(results.doc_embedding_path)
        assert(results.doc_mapper_path)
        doc_mapper = seri.load(results.doc_mapper_path)
        doc_embedding = train_data_saver.load_doc_embedding(doc_mapper,results.doc_embedding_path)

        top_eval = 10
        for reversed_info in random.sample(list(doc_mapper.reversed_doc_mapper.values()), top_eval):
            print(doc_embedding.similar_by(reversed_info[0]))
        return

    if results.is_create_word_mapper:
        print("Creating mapper!")
        if results.csv_folder_path is not None:
            print("Creating word_count.json from csv folder {}".format(results.csv_folder_path))
            word_count = build_word_count(results.save_folder_path, results.csv_folder_path, results.use_preprocessor)
        else:
            assert results.word_count_path is not None
            print("Loading word_count.json from {}".format(results.word_count_path))
            word_count = seri.load(results.word_count_path)
        if results.min_word_count is not None:
            word_mapper = word_count.get_vocab_by_min_count(int(results.min_word_count))
            print("Successfully create word_mapper length {} with min_word_count {}".format(word_mapper.get_vocabulary_size(),
                                                                                            results.min_word_count))
        else:
            word_mapper = word_count.get_vocab_by_size(int(results.vocabulary_size))
            print("Successfully create word_mapper length {} with vocabulary_size {}".format(word_mapper.get_vocabulary_size(),
                                                                                             results.vocabulary_size))

        seri.save(word_mapper, os.path.join(results.save_folder_path, "word_mapper.json"))
        return
    if results.is_create_config:
        print("Creating config!")
        build_config(results.save_folder_path, results.csv_folder_path,results.train_model, results.train_mode)
        return

    # save_folder_path = results.save_folder_path
    config_path = results.config_path
    word_mapper_path = results.mapper_path
    assert utilities.exists(config_path)
    config = train_data_saver.load_config(config_path)

    train_data = data_model.DataModelFactory.generate_data_model(config)

    assert utilities.exists(word_mapper_path)
    train_data_saver.restore_word_mapper(train_data, word_mapper_path)

    if utilities.exists(train_data_saver.get_progress_path()):
        train_data_saver.restore_progress(train_data, train_data_saver.get_progress_path())
    else:
        train_data_saver.init_progress(train_data, train_data.config)

    train_vec = NetworkFactory.generate_network(config)
    if config.mode == "doc2vec":
        doc_mapper = seri.load(results.doc_mapper_path)
        train_data.set_doc_mapper_data(doc_mapper)
    elif config.mode == "docrelevant":
        category_mapper = seri.load(results.category_mapper_path)
        train_data.set_category_mapper(category_mapper)

    train_vec.use_cpu = results.is_use_cpu

    train_vec.set_train_data(train_data, train_data_saver)
    train_vec.restore_last_training_if_exists()

    if results.is_create_embedding:
        assert (utilities.exists(train_data_saver.get_progress_path()))
        print("Creating word embedding from {}".format(train_data.config.save_folder_path))
        train_data_saver.save_word_embedding(train_vec.final_embeddings,
                                             train_data.word_mapper.reversed_dictionary)
        return

    if results.is_create_doc_embedding:
        assert (utilities.exists(train_data_saver.get_progress_path()))
        print("Creating doc embedding from {}".format(train_data.config.save_folder_path))
        doc_embedding = train_vec.get_doc_embedding()
        train_data_saver.save_doc_embedding(doc_embedding.embedding,
                                            doc_embedding.doc_mapper.reversed_doc_mapper)
        train_data_saver.save_doc_mapper(doc_embedding.doc_mapper)
        print(doc_embedding.similar_by(doc_embedding.doc_mapper.reversed_doc_mapper["0"][0]))
        return
    if results.is_eval_doc_rele_embedding:
        assert (utilities.exists(train_data_saver.get_progress_path()))
        doc_embedding = train_vec.get_doc_embedding()
        if results.eval_query is not None:
            print(train_vec.retrieve_by_query([results.eval_query]))
        else:
            top_eval = 10
            query_list = []
            for reversed_mapper in random.sample(list(doc_embedding.doc_mapper.reversed_doc_mapper.values()), top_eval):
                org_idx = reversed_mapper[0]
                csv_path = reversed_mapper[1]
                line_number = reversed_mapper[2]
                query = utilities.extract_query_from_csv(org_idx, csv_path, line_number)
                query_list.append(query)
            print(train_vec.retrieve_by_query(query_list))

        return
    if results.is_eval_doc_rele_prediction:
        assert (utilities.exists(train_data_saver.get_progress_path()))
        doc_embedding = train_vec.get_doc_embedding()
        category_mapper = train_data.category_mapper
        top_eval = 10
        acc = 0
        for reversed_mapper in random.sample(list(doc_embedding.doc_mapper.reversed_doc_mapper.values()), top_eval):
            org_idx = reversed_mapper[0]
            csv_path = reversed_mapper[1]
            line_number = reversed_mapper[2]
            post_idx, title, tags, content, catId = utilities.extract_info_from_csv(org_idx, csv_path, line_number)
            train_word = preprocessor.get_train_word_from_title_and_tags(title, tags)
            prediction = train_vec.get_query_prediction(train_word)
            prediction_catId = category_mapper.reversed_dictionary[str(prediction)]
            print("Query {}".format(" ".join(train_word)))
            print("Prediction catId {} - true catId".format(prediction_catId, catId))
            if prediction_catId == catId:
                acc += 1
        print("Total accuracy {}".format(acc/top_eval))
        return
    if results.train_type == "empty":
        train_vec.empty_training()
    elif results.train_type is not None:
        train_vec.train()

def build_config(save_folder_path, csv_folder_path, train_model, train_mode):
    config = data_model.ConfigFactory.generate_config(save_folder_path, csv_folder_path, train_model, train_mode)
    seri.save(config, os.path.join(save_folder_path, "config.json"))


if __name__ == "__main__":
    main()
    # build_vocab("./temp/", "./data/longdata/*.csv", 10000)
    # build_config( "./temp/shortdata/", "./data/shortdata/*.csv")
    # word_count = seri.load("./temp/word_count.json")
    # print(word_count.word_count["long_văn"])
    # vocab = word_count.get_vocab(min_count=100)
    # print(len(vocab.dictionary))
    # word_count.draw_histogram()
    # print(word_count.word_count["sửa_ti_vi_tại"])
