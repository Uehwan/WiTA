import pandas as pd
import numpy as np


def kor_non_lexical(csv_path, num_word, num_data):
    csv_data = pd.read_csv(csv_path)
    syllable = csv_data['syllable']
    syl = []
    sample_len = [1, 2, 3]
    for s in syllable:
        syl.append(s)
    syl = np.asarray(syl)
    for i in range(num_data):
        for j in range(num_word):
            length = np.random.choice(sample_len, 1)
            non_lexical = np.random.choice(syl, length[0])
            with open('kor_non_lex_' + str(i) + ".txt", "a") as f:
                f.write("".join(map(str, non_lexical)))
                f.write("\n")


def freq_kor_word(csv_path, num_word, num_data):
    csv_data = pd.read_csv(csv_path)
    word = csv_data['word']
    freq_word = []
    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for w in word:
        freq_word.append(w)
    freq_word = np.asarray(freq_word)
    for x in range(len(freq_word)):
        for y in freq_word[x]:
            if y in num:
                freq_word[x] = freq_word[x].replace(y, '')
    for i in range(num_data):
        for j in range(num_word):
            kor_word = np.random.choice(freq_word, 1)
            with open('kor_word_' + str(i) + '.txt', "a") as f:
                f.write("".join(map(str, kor_word)))
                f.write("\n")


def eng_non_lexical(num_word, num_data):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    sample_len = [3, 4, 5, 6, 7]
    for i in range(num_data):
        for j in range(num_word):
            length = np.random.choice(sample_len, 1)
            eng_non_lex = np.random.choice(alphabet, length)
            with open('eng_non_lex_' + str(i) + '.txt', "a") as f:
                f.write("".join(map(str, eng_non_lex)))
                f.write("\n")


def eng_freq_word(txt_path, num_word, num_data):
    f = open(txt_path, 'r')
    eng_word_list = f.readlines()
    for i in range(num_data):
        for j in range(num_word):
            eng_word = np.random.choice(eng_word_list, 1)
            with open('eng_freq_word_' + str(i) + '.txt', "a") as f:
                f.write("".join(map(str, eng_word)))


def kor_eng_non_lex(kor_path, num_word, num_data):
    kor_data = pd.read_csv(kor_path)
    syllable = kor_data['syllable']
    syl = []
    kor_sample_len = [1, 2, 3]
    eng_sample_len = [2, 3, 4]
    for s in syllable:
        syl.append(s)
    syl = np.asarray(syl)
    order = [0, 1]
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    for i in range(num_data):
        for j in range(num_word):
            korean_first = np.random.choice(order, 1)
            kor_length = np.random.choice(kor_sample_len, 1)
            kor_non_lex = np.random.choice(syl, kor_length[0])
            eng_length = np.random.choice(eng_sample_len, 1)
            eng_non_lex = np.random.choice(alphabet, eng_length)
            if korean_first == 1:
                with open('kor_eng_nonlex_' + str(i) + '.txt', "a") as f:
                    f.write("".join(map(str, kor_non_lex)))
                    f.write("".join(map(str, eng_non_lex)))
                    f.write("\n")
            else:
                with open('kor_eng_nonlex_' + str(i) + '.txt', "a") as f:
                    f.write("".join(map(str, eng_non_lex)))
                    f.write("".join(map(str, kor_non_lex)))
                    f.write("\n")


def tutorial(kor_path, eng_path, num_word):
    kor_data = pd.read_csv(kor_path)
    korean_word = kor_data['word']
    kor = []
    for w in korean_word:
        kor.append(w)
    kor_word_arr = np.asarray(kor)
    f = open(eng_path, 'r')
    eng_word_list = f.readlines()
    for j in range(num_word//2 + 1):
        eng_word = np.random.choice(eng_word_list, 1)
        kor_word = np.random.choice(kor_word_arr, 1)
        with open('tutorial.txt', "a") as f:
            f.write("".join(map(str, eng_word)))
            f.write("".join(map(str, kor_word)))
            f.write("\n")


if __name__ == "__main__":
    kor_non_lexical('./most_freq_kor_syllable.csv', 15, 2)
    freq_kor_word('./most_freq_kor_word.csv', 75, 2)
    eng_non_lexical(15, 2)
    eng_freq_word('./google-10000-english-no-swears.txt', 75, 2)
    kor_eng_non_lex('./most_freq_kor_syllable.csv', 15, 2)
    tutorial('./most_freq_kor_word.csv', './google-10000-english-no-swears.txt', 15)
