import re
from nltk import ngrams

KEEP_VN_CHAR = re.compile(u"[_aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz]+")


def nomalize_uni_string(row):
    # row = row.lower()
    #
    # row = REMOVE_NUMBER.sub("", row)
    #
    # # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    # row = REMOVE_LAST_DOT.sub("", row)
    #
    # # Xoa cac ky tu dac biet
    #
    # row = re.sub(u"\p{P}+","", row)
    #
    # # Xoa cac dau cach lien tuc
    # row = REMOVE_DUPLICATE_SPACE.sub(" ", row)
    #
    # row = row.strip()
    return row

def split_row_to_word(string):
    return list(ngrams(string.split(), 1))

def split_preprocessor_row_to_word(string):
    string = string.lower()
    # gram_str = list(ngrams(string.split(), n))
    # return [" ".join(gram).lower() for gram in gram_str]
    return KEEP_VN_CHAR.findall(string)
