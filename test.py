from data_model import *


def main():
    test_str = "A Lazy Fox Jump Over A Crazy Dog"
    test_iter = Iter_Sentences(test_str.split(" "))
    for item in test_iter:
        print(item)


main()
