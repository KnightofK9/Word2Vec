from data_model import *
import preprocessor


def test_iter():
    test_str = "A Lazy Fox Jump Over A Crazy Dog"
    test_iter = Iter_Sentences(test_str.split(" "))
    for item in test_iter:
        print(item)


def test_batch():
    train_data = Iter_Batch_Data_Model("./data/news10k.csv")
    count = 10
    for word, context in train_data:
        if count == 0:
            break
        print("({},{})".format(word, context))
        count = count - 1


def test_pre_processor():
    test_str = " Kỳ nghỉ năm Nga bắt_đầu ngày 31/12 kéo_dài ngày 8/1"
    result = preprocessor.nomalize_uni_string(test_str)
    assert (result == "kỳ nghỉ năm nga bắt_đầu ngày 31/12 kéo_dài ngày 8/1")
    test_str = "Theo An_Bình ( Dân_Trí ) ‹ › ×"
    result = preprocessor.nomalize_uni_string(test_str)
    assert (result == "theo an_bình dân_trí ×")


def main():
    test_pre_processor()


main()
