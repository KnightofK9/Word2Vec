from datamodel import *
import preprocessor


def test_iter():
    test_str = "A Lazy Fox Jump Over A Crazy Dog"
    test_iter = IterSentences(test_str.split(" "))
    for item in test_iter:
        print(item)


def test_batch():
    train_data = IterBatchDataModel("./data/news10k.csv")
    count = 10
    for word, context in train_data:
        if count == 0:
            break
        print("({},{})".format(word, context))
        count = count - 1


def test_pre_processor():
    test_str = " Kỳ nghỉ năm Nga bắt_đầu ngày 31/12 kéo_dài ngày 8/1 (1992)"
    result = preprocessor.nomalize_uni_string(test_str)
    assert (result == "kỳ nghỉ năm nga bắt_đầu ngày 31/12 kéo_dài ngày 8/1 1992")
    test_str = "Theo An_Bình ( Dân_Trí ) ‹ › × @ # $@!@@!"
    result = preprocessor.nomalize_uni_string(test_str)
    assert (result == "theo an_bình dân_trí ×")


def main():
    test_str = "Theo An_Bình ( Dân_Trí ) ‹ › × @ # $@!@@!"
    result = preprocessor.split_row_to_word(test_str)
    print(result)
    # test_pre_processor()
    # print(preprocessor.nomalize_uni_string("Điện Kremlin tiết_lộ kế_hoạch cá_nhân Tổng_thống Nga Vladimir_Putin kỳ nghỉ lễ năm "))

main()
