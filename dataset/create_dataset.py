import os
import shutil


curr_path = os.getcwd()


def create_dir():
    path = os.path.join(curr_path, "dataset/")
    for i in range(2):
        os.mkdir(path+str(i))


def move_file():
    for i in range(2):
        eng_word_from = os.path.join(curr_path, 'eng_freq_word_' + str(i) + '.txt')
        eng_nonlex_from = os.path.join(curr_path, 'eng_non_lex_' + str(i) + '.txt')
        kor_word_from = os.path.join(curr_path, "kor_word_" + str(i) + '.txt')
        kor_nonlex_from = os.path.join(curr_path, "kor_non_lex_" + str(i) + '.txt')
        kor_eng_nonlex_from = os.path.join(curr_path, "kor_eng_nonlex_" + str(i) + '.txt')

        eng_word_to = os.path.join(curr_path, "dataset/" + str(i) + '/eng_freq_word_' + str(i) + '.txt')
        eng_nonlex_to = os.path.join(curr_path, "dataset/" + str(i) + '/eng_non_lex_' + str(i) + '.txt')
        kor_word_to = os.path.join(curr_path, "dataset/" + str(i) + '/kor_word_' + str(i) + '.txt')
        kor_nonlex_to = os.path.join(curr_path, "dataset/" + str(i) + '/kor_non_lex_' + str(i) + '.txt')
        kor_eng_nonlex_to = os.path.join(curr_path, "dataset/" + str(i) + '/kor_eng_nonlex_' + str(i) + '.txt')

        shutil.move(eng_word_from, eng_word_to)
        shutil.move(eng_nonlex_from, eng_nonlex_to)
        shutil.move(kor_word_from, kor_word_to)
        shutil.move(kor_nonlex_from, kor_nonlex_to)
        shutil.move(kor_eng_nonlex_from, kor_eng_nonlex_to)


if __name__ == "__main__":
    # create_dir()
    move_file()