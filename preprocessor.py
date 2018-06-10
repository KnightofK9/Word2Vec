import re
from nltk import ngrams

import unicodedata
KEEP_VN_CHAR = re.compile(u"[_aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz]+")
VN_CHAR = "ẮẰẲẴẶĂẤẦẨẪẬÂÁÀÃẢẠĐẾỀỂỄỆÊÉÈẺẼẸÍÌỈĨỊỐỒỔỖỘÔỚỜỞỠỢƠÓÒÕỎỌỨỪỬỮỰƯÚÙỦŨỤÝỲỶỸỴ"
FILTER_FLOAT_NUM = re.compile('([0-9]*[.])?[0-9]+')
FILTER_SPECIAL_CHAR = re.compile('[^A-Za-z._' + VN_CHAR + VN_CHAR.lower() + ']+')
FILTER_LETTER = re.compile(' [a-zA-Z' + VN_CHAR + VN_CHAR.lower() + '] ')
FILTER_EPLISIS = re.compile('\.{2,}')
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
    return string.split(" ")

def split_preprocessor_row_to_word(string):
    string = string.lower()
    # gram_str = list(ngrams(string.split(), n))
    # return [" ".join(gram).lower() for gram in gram_str]
    return KEEP_VN_CHAR.findall(string)

def split_preprocessor_row_to_word_v2(row):
    # row = [row]
    # # filter all empty element in list
    # filter_empty = list(filter(None, row))
    # filter_empty = list(filter(lambda name: name.strip(), filter_empty))
    #
    # # join '.' to seperate sentence
    # string_row = ' '.join([('. ' if c[0].isupper() and c.count(" ") >= 6 else '') + c for c in filter_empty])

    # remove special characters
    filter_float_num = FILTER_FLOAT_NUM.sub('', row)
    filter_special_char = FILTER_SPECIAL_CHAR.sub( ' ',
                                 unicodedata.normalize('NFC', filter_float_num))

    # remove individual letter
    filter_letter = FILTER_LETTER.sub(' ', filter_special_char.lower())

    # remove '..."
    filter_ellipsis = FILTER_EPLISIS.sub( ' ', filter_letter)

    return filter_ellipsis.strip().split(" ")
