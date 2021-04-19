from __future__ import absolute_import, division, print_function

import os
import glob
import hgtk
import utils
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


HAN_FIRST = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅃㅉㄸㄲㅆ'
HAN_MIDDLE = 'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ'
HAN_LAST = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㅆㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ'


def read_label_eng(data_path):
    label_file = []
    for x in glob.glob(os.path.join(data_path, '*/lex/*/gt*')):
        # if x.split('_')[-1].split('/')[-1] in ['gt.txt', 'gt2.txt']:
        label_file.append(x)

    labels = []
    for file in label_file:
        text_file = open(file, 'r')
        text_line = text_file.read().splitlines()
        labels.extend(text_line)
        text_file.close()

    dict = {'former': [], 'latter': []}

    for label in labels:
        for idx, char in enumerate(label[:-1]):
            dict['former'].append(char)
            dict['latter'].append(label[idx+1])

    df = pd.DataFrame(dict, columns=['former', 'latter'])
    confusion_matrix = pd.crosstab(df['former'], df['latter'], rownames=['Former letter'], colnames=['Latter letter'])

    sns.heatmap(confusion_matrix, annot=False, cmap='Reds')
    plt.savefig("english_nonlex_confusion_matrix.png")
    plt.show()


def read_label_kor(data_path):
    label_file = []
    for x in glob.glob(os.path.join(data_path, '*/nonlex/*/gt.txt')):
        label_file.append(x)
    labels = []
    for file in label_file:
        text_file = open(file, 'rb')
        text_line = text_file.readlines()
        labels.extend(text_line)
        text_file.close()

    dict_fm = {'first': [], 'middle': []}
    dict_ml = {'middle': [], 'last': []}

    for label in labels:
        label = label[:-2].decode('cp949')
        decomp_label = hgtk.text.decompose(label)
        split_label = decomp_label.split("ᴥ")
        for char in split_label[:-1]:
            dict_fm['first'].append(char[0])
            dict_fm['middle'].append(char[1])
            if len(char) == 3:
                dict_ml['middle'].append(char[1])
                dict_ml['last'].append(char[2])

    df_fm = pd.DataFrame(dict_fm, columns=['first', 'middle'])
    df_ml = pd.DataFrame(dict_ml, columns=['middle', 'last'])

    confusion_matrix_fm = pd.crosstab(df_fm['first'], df_fm['middle'], rownames=['First letter'], colnames=['Middle letter'])
    confusion_matrix_ml = pd.crosstab(df_ml['middle'], df_ml['last'], rownames=['Middle letter'],
                                      colnames=['Last letter'])

    sns.heatmap(confusion_matrix_fm, annot=False, cmap='Blues')
    plt.savefig("korean_nonlex_fm_confusion_matrix.png")
    plt.show()
    sns.heatmap(confusion_matrix_ml, annot=False, cmap='Blues')
    plt.savefig("korean_nonlex_ml_confusion_matrix.png")
    plt.show()


def load_label(path, lang='english', type='lex'):
    LEX = 'lex/*/gt.txt' if type is 'lex' else 'nonlex/*/gt.txt'
    label = []
    for x in glob.glob(os.path.join(path, LEX)):
        label.append(x)

    labels = []
    for i, file in enumerate(label):
        if lang == 'english':
            text_file = open(file, 'r')
        elif lang == 'korean':
            text_file = open(file, 'rb')
        else:
            pass
        text_line = text_file.readlines()
        labels.extend(text_line)
        text_file.close()

    return labels


def count_letter(labels, lang='english'):
    if lang == 'english':
        dict = {char: 0 for char in utils.ALPHABET}
        for label in labels:
            for letter in label[:-1]:
                dict[letter] += 1
        return dict
    elif lang == 'korean':
        first_dict = {char: 0 for char in HAN_FIRST}
        middle_dict = {char: 0 for char in HAN_MIDDLE}
        last_dict = {char: 0 for char in HAN_LAST}
        for label in labels:
            label = label[:-2].decode('cp949')
            decomp_label = hgtk.text.decompose(label)
            split_label = decomp_label.split("ᴥ")
            for char in split_label[:-1]:
                first_dict[char[0]] += 1
                middle_dict[char[1]] += 1
                if len(char) == 3:
                    last_dict[char[2]] += 1
        return first_dict, middle_dict, last_dict
    else:
        # not implemented for kor_eng_nonlex data yet
        pass


def stack_bar(train_path, val_path, test_path, lang='english', type='lex'):
    train_label = load_label(train_path, lang, type)
    val_label = load_label(val_path, lang, type)
    test_label = load_label(test_path, lang, type)
    width = 0.5
    if lang == 'english':
        N = len(utils.ALPHABET)
        train_dict = count_letter(train_label, lang=lang)
        val_dict = count_letter(val_label, lang=lang)
        test_dict = count_letter(test_label, lang=lang)
        ind = np.arange(N)

        p1 = plt.bar(ind, tuple(train_dict.values()), width)
        p2 = plt.bar(ind, tuple(val_dict.values()), width, bottom=tuple(train_dict.values()))
        p3 = plt.bar(ind, tuple(test_dict.values()), width,
                     bottom=tuple(map(operator.add, tuple(train_dict.values()), tuple(val_dict.values()))))
        plt.ylabel('Number of appearance')
        plt.title('Number of appearance of each letter by dataset')
        plt.xticks(ind, train_dict.keys())
        plt.legend((p1[0], p2[0], p3[0]), ('Train', 'Val', 'Test'))
        plt.savefig("Distribution of each alphabet in non-lexical datasets.png")
        plt.show()
    elif lang == 'korean':
        N_f = len(HAN_FIRST)
        N_m = len(HAN_MIDDLE)
        N_l = len(HAN_LAST)
        train_dict_f, train_dict_m, train_dict_l = count_letter(train_label, lang=lang)
        val_dict_f, val_dict_m, val_dict_l = count_letter(val_label, lang=lang)
        test_dict_f, test_dict_m, test_dict_l = count_letter(test_label, lang=lang)
        ind_f = np.arange(N_f)
        ind_m = np.arange(N_m)
        ind_l = np.arange(N_l)

        p1_f = plt.bar(ind_f, tuple(train_dict_f.values()), width)
        p2_f = plt.bar(ind_f, tuple(val_dict_f.values()), width, bottom=tuple(train_dict_f.values()))
        p3_f = plt.bar(ind_f, tuple(test_dict_f.values()), width,
                     bottom=tuple(map(operator.add, tuple(train_dict_f.values()), tuple(val_dict_f.values()))))
        plt.ylabel('Number of appearance')
        plt.title('Number of appearance of first letters by dataset')
        plt.xticks(ind_f, train_dict_f.keys())
        plt.legend((p1_f[0], p2_f[0], p3_f[0]), ('Train', 'Val', 'Test'))
        plt.show()

        p1_m = plt.bar(ind_m, tuple(train_dict_m.values()), width)
        p2_m = plt.bar(ind_m, tuple(val_dict_m.values()), width, bottom=tuple(train_dict_m.values()))
        p3_m = plt.bar(ind_m, tuple(test_dict_m.values()), width,
                     bottom=tuple(map(operator.add, tuple(train_dict_m.values()), tuple(val_dict_m.values()))))
        plt.ylabel('Number of appearance')
        plt.title('Number of appearance of middle letters by dataset')
        plt.xticks(ind_m, train_dict_m.keys())
        plt.legend((p1_m[0], p2_m[0], p3_m[0]), ('Train', 'Val', 'Test'))
        plt.show()

        p1_l = plt.bar(ind_l, tuple(train_dict_l.values()), width)
        p2_l = plt.bar(ind_l, tuple(val_dict_l.values()), width, bottom=tuple(train_dict_l.values()))
        p3_l = plt.bar(ind_l, tuple(test_dict_l.values()), width,
                     bottom=tuple(map(operator.add, tuple(train_dict_l.values()), tuple(val_dict_l.values()))))
        plt.ylabel('Number of appearance')
        plt.title('Number of appearance of first letters by dataset')
        plt.xticks(ind_l, train_dict_l.keys())
        plt.legend((p1_l[0], p2_l[0], p3_l[0]), ('Train', 'Val', 'Test'))
    else:
        # not implemented for kor_eng_nonlex data yet
        pass


if __name__ == "__main__":
    from options import AirTypingOptions

    options = AirTypingOptions()
    opts = options.parse()

    read_label_eng('/root/dataset/wita/RawData/data/english/')
    read_label_kor('/root/dataset/wita/RawData/data/korean/')

    train_eng = '/root/dataset/wita/RawData/data/english/train'
    val_eng = '/root/dataset/wita/RawData/data/english/val'
    test_eng = '/root/dataset/wita/RawData/data/english/test'
    train_kor = '/root/dataset/wita/RawData/data/korean/train'
    val_kor = '/root/dataset/wita/RawData/data/korean/val'
    test_kor = '/root/dataset/wita/RawData/data/korean/test'
    stack_bar(train_eng, val_eng, test_eng, lang='english', type='nonlex')
    stack_bar(train_eng, val_eng, test_eng, lang='english', type='lex')
    stack_bar(train_kor, val_kor, test_kor, lang='korean', type='nonlex')
    stack_bar(train_kor, val_kor, test_kor, lang='korean', type='lex')
