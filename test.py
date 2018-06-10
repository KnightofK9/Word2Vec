import numpy as np
import preprocessor
import utilities
from data_model import Saver
from serializer import JsonClassSerialize
from operator import methodcaller

def test_pre_processor():
    test_str = " Kỳ nghỉ năm Nga bắt_đầu ngày 31/12 kéo_dài ngày 8/1 (1992)"
    result = preprocessor.nomalize_uni_string(test_str)
    assert (result == "kỳ nghỉ năm nga bắt_đầu ngày 31/12 kéo_dài ngày 8/1 1992")
    test_str = "Theo An_Bình ( Dân_Trí ) ‹ › × @ # $@!@@!"
    result = preprocessor.nomalize_uni_string(test_str)
    assert (result == "theo an_bình dân_trí ×")


seri = JsonClassSerialize()

def main():
    # array = [0,1,2,3,4,5,6,7,8]
    # print(utilities.sub_array_soft(array,6,2,2))
    # print(utilities.sub_array_hard(array,2,2,2))
    # train_data_saver = Saver()
    # doc_mapper = seri.load("./temp/longdata_cbow_doc2vec/doc_mapper.json")
    # doc_embedding = train_data_saver.load_doc_embedding(doc_mapper, "./temp/longdata_cbow_doc2vec/doc_embedding.vec")
    # print(1)
    # test_str = "Theo An_Bình ( Dân_Trí ) ‹ › × @ # $@!@@!"
    # result = preprocessor.split_row_to_word(test_str)
    # print(result)
    # test_pre_processor()
    print(preprocessor.split_preprocessor_row_to_word_v2("Điện Kremlin tiết_lộ kế_hoạch cá_nhân Tổng_thống Nga Vladimir_Putin kỳ nghỉ lễ năm "))

main()
