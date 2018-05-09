import re
from nltk import ngrams

REMOVE_NUMBER = re.compile(r"[0-9\.]+")
REMOVE_LAST_DOT = re.compile(r"[\.,\?]+$")
REMOVE_SPECIAL_CHAR = re.compile(r"[‹›,.;“:”\"\'!?()<>{}\\]")
REMOVE_DUPLICATE_SPACE = re.compile(r'\s+')


def nomalize_uni_string(row):
    row = row.lower()

    row = REMOVE_NUMBER.sub("", row)

    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = REMOVE_LAST_DOT.sub("", row)

    # Xoa cac ky tu dac biet

    row = REMOVE_SPECIAL_CHAR.sub("", row)

    # Xoa cac dau cach lien tuc
    row = REMOVE_DUPLICATE_SPACE.sub(" ", row)

    row = row.strip()
    return row

def split_row_to_word(string, n):
    gram_str = list(ngrams(string.split(), n))
    return [" ".join(gram).lower() for gram in gram_str]
