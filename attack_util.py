import random
import jamotools
import re
import unicodedata
def insert_long_distractor(sentence,begsl,endsl,ch): #6
    # insert_pool = [str(i) for i in list(range(0,10))]  + ['!','&','%','$','#']
    # k = 10
    # sent_len = len(sentence.split(' '))
    # distractor = random.choice(insert_pool) * sent_len * k
    # distractor =" up" *8
    beg_word = ch+" "
    end_word = " "+ch
    for_distractor =beg_word *begsl
    # distractor =" reuters" *30
    distractor =end_word *endsl
    # distractor =" to" *180
    return for_distractor+sentence + distractor

# def insert_long_distractor(sentence):
#     # insert_pool = [str(i) for i in list(range(0,10))]  + ['!','&','%','$','#']
#     # k = 10
#     # sent_len = len(sentence.split(' '))
#     # distractor = random.choice(insert_pool) * sent_len * k
#     distractor =" up" *8
#     # beg_word = ch+" "
#     # end_word = " "+ch
#     # for_distractor =beg_word *begsl
#     # # distractor =" reuters" *30
#     # distractor =end_word *endsl
#     # distractor =" to" *180
#     return sentence + distractor

def insert_space(word): #1
    word_len = len(word)
    if word_len <= 1:
        return None
    insert_pos = random.choice(list(range(0, word_len-1))) + 1  # word_len=5, 0,1,2,  ->  1,2,3   fooli   f ooli   fo oli foo li  fool i
    new_word = word[: insert_pos] + ' ' + word[insert_pos: ]
    return new_word

def insert_irrelevant(word): #2
    word_len = len(word)
    if word_len <= 1:
        return None
    insert_pos = random.choice(
        list(range(0, word_len - 1))) + 1  # word_len=5, 0,1,2,  ->  1,2,3   fooli   f ooli   fo oli foo li  fool i
    # insert_pool = ['$', '#', '@', '%', '&', '!', '.','<', '^','>','*']
    # insert_pool = ['it']
    insert_pool = ['$', '#', '@', '%', '&', '!','*','1']
    insert_char = random.choice(insert_pool)
    new_word = word[: insert_pos] + insert_char + word[insert_pos:]
    return new_word

def delete_char(word):#3
    word_len = len(word)
    if word_len <= 1:
        return None
    insert_pos = random.choice(list(range(0, word_len - 1))) + 1
    new_word = word[: insert_pos] + word[insert_pos+1:]
    return new_word

def swap_char(word): #4
    # 단어 하나하나를 바꾸는것보다 자음 또는 모음 같은 것을 바꾸는게 나을듯
    word_len = len(word)
    # if word_len < 5:
    #     return None
    insert_pos = random.choice(list(range(0, word_len - 2))) + 1
    i = word[insert_pos]
    j = word[insert_pos+1]
    new_word = word[: insert_pos] + j + i + word[insert_pos+2:]
    return new_word

def sub_char(word): #5
    # 모음을 바꾸는 방법
    # 1. 단어가 들어왔을 때 한 글자 랜덤으로 픽
    word_len = len(word)
    if word_len <1:
        return None
    hangul = re.compile('[^가-힣]') #자음과 모음으로 완성된 한글이 아닌 문자
    pos_list = []
    for id, wo in enumerate(word):
        if hangul.sub('', wo) =='':
            continue
        else:
            pos_list.append(id)
    if len(pos_list) ==0:
        return None
    word_pos = random.choice(pos_list) # 단어를 한 단어씩 분해, '보며'-> '보','며' 
    target_word = word[word_pos] 
    hangul_char = jamotools.split_syllable_char(target_word) # 자음 모음 분해
    word_len = len(word)
    ########### jamotools에서의 모음 != 'ㅏ' 기에, jamotools에서 나오는 것으로 작성
    sub_dict = {
        'ㅏ':'ㅑ',
        'ㅓ':'ㅕ',
        'ㅑ':'ㅏ',
        'ㅕ':'ㅓ',
        'ㅗ':'ㅛ',
        'ㅜ':'ㅠ',
        'ㅛ':'ㅗ',
        'ㅠ':'ㅜ',
        # 'ㅡ':'ㅣ', 
        # 'ㅣ':'ㅡ', 
        'ㅐ':'ㅒ',
        'ㅔ':'ㅖ',
        'ㅒ':'ㅐ',
        'ㅖ':'ㅔ',
        # 'ㅣ': '1',
        # 'ㅈ': 'ㅊ',
        # 'i': '1',
        # 's': '$',
        # 'a': ''
    }
    new_char = []
    for ch in hangul_char:
        ascii_ch = [i for i in sub_dict.keys() if abs(ord(i) - ord(ch)) == 8174]
        if len(ascii_ch) ==1:
            # 아스키 코드로 변경한 값이 8174 크기만큼 차이가 난다면
            new_ch = sub_dict[ascii_ch[0]]
            new_char.append(new_ch)
        else:
            new_char.append(ch)
    new_word = jamotools.join_jamos(new_char)
    new_word = word.replace(target_word, new_word)
    return new_word


def sub_char_sound(word): #5
    # 모음을 바꾸는 방법
    # 1. 단어가 들어왔을 때 한 글자 랜덤으로 픽
    word_len = len(word)
    if word_len <1:
        return None
    hangul = re.compile('[^가-힣]') #자음과 모음으로 완성된 한글이 아닌 문자
    pos_list = []
    for id, wo in enumerate(word):
        if hangul.sub('', wo) =='':
            continue
        else:
            pos_list.append(id)
    if len(pos_list) ==0:
        return None
    word_pos = random.choice(pos_list)
    target_word = word[word_pos]
    hangul_char = jamotools.split_syllable_char(target_word)
    word_len = len(word)
    ########### jamotools에서의 모음 != 'ㅏ' 기에, jamotools에서 나오는 것으로 작성
    sub_dict = {
        'ㅏ':'ㅑ',
        'ㅓ':'ㅕ',
        'ㅑ':'ㅏ',
        'ㅕ':'ㅓ',
        'ㅗ':'ㅛ',
        'ㅜ':'ㅠ',
        'ㅛ':'ㅗ',
        'ㅠ':'ㅜ',
        'ㅡ':'ㅣ',
        'ㅣ':'ㅡ', 
        'ㅐ':'ㅒ',
        'ㅔ':'ㅖ',
        'ㅒ':'ㅐ',
        'ㅖ':'ㅔ',
        #된소리
        'ㄱ':'ㄲ',
        'ㄷ':'ㄸ',
        'ㅂ':'ㅃ',
        'ㅅ':'ㅆ',
        'ㅈ':'ㅉ',
        # 'ㅣ': '1',
        # 'ㅈ': 'ㅊ',
        # 'i': '1',
        # 's': '$',
        # 'a': ''
    }
    new_char = []
    
    for ch in hangul_char:
        ascii_ch = [i for i in sub_dict.keys() if unicodedata.normalize('NFKC',i) == unicodedata.normalize('NFKC',ch)] # 
        # ascii_ch = [i for i in sub_dict.keys() if abs(ord(i) - ord(ch)) == 8174] # 모음 차이 8174, 자음 차이 다 다름
        if len(ascii_ch) ==1:
            # 아스키 코드로 변경한 값이 8174 크기만큼 차이가 난다면
            new_ch = sub_dict[ascii_ch[0]]
            new_char.append(new_ch)
        else:
            new_char.append(ch)
    new_word = jamotools.join_jamos(new_char)
    new_word = word.replace(target_word, new_word)
    return new_word


def insert_period(sentence):
    word_li = sentence.split(' ')
    new_word_li = []
    for word in word_li:
        new_word_li.append(word)
        new_word_li.append('.')
    return ' '.join(new_word_li)

def split_token(word): #6
    # 모음을 바꾸는 방법
    # 1. 단어가 들어왔을 때 한 글자 랜덤으로 픽
    word_len = len(word)
    if word_len <1:
        return None
    hangul = re.compile('[^ 가-힣]') #자음과 모음으로 완성된 한글이 아닌 문자
    pos_list = []
    for id, wo in enumerate(word):
        if hangul.sub('', wo) =='':
            continue
        else:
            pos_list.append(id)
    if len(pos_list) ==0:
        return None
    word_pos = random.choice(pos_list)
    target_word = word[word_pos]
    new_word = jamotools.split_syllables(target_word)
    if len(new_word) != 2:
        return None
    new_word = word.replace(target_word, new_word)
    return new_word



if __name__ == '__main__':
    '''test'''
    test_sentence = ''
    # print(insert_long_distractor(test_sentence))
    print(insert_period(test_sentence))
    word = '여자가'
    print(insert_space(word))
    print(insert_irrelevant(word))
    print(delete_char(word))
    print(swap_char(word))
    print(sub_char(word))
    print(split_token(word))







