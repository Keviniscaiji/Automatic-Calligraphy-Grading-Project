from PIL import Image
# from pdf2image import convert_from_path, convert_from_bytes  # conda install -c conda-forge poppler
import os
import matplotlib.pyplot as plt
import numpy as np
from datastruct import paper, line, word, letter
from cut_img import *


def save_img(img, img_name):
    img = Image.fromarray(img)
    img.save(img_name)


def get_img_from_pdf(filename, dpi=256):
    images = convert_from_path(filename, dpi=dpi)
    image = images[0]
    image = np.array(image)
    return image


def preprocess(srcimg):
    # ？
    dstimg = srcimg
    return dstimg


def get_clear_handwriting_cut(srcimg):
    # 将手写部分提取出
    try:
        re, coord = cutX(srcimg, threshold01=3, space01=15, threshold10=3, space10=15)
        example_region = re[3]
        handwriting_img = re[4]
        re, coord2 = cutX(handwriting_img, threshold01=3, space01=5, threshold10=3, space10=5)
        handwriting_region = re[0]
        re, coord3 = cutY(handwriting_region, threshold01=3, space01=15, threshold10=3, space10=15)
        handwriting_region = re[0]
        return handwriting_region
    except:
        return []


def region_cut(srcimg):
    re, coord = cutX(srcimg, threshold01=6, space01=30, threshold10=3, space10=30)
    example_region = re[3]
    handwriting_img = re[4]

    re, coord2 = cutX(handwriting_img, threshold01=6, space01=10, threshold10=3, space10=10)
    handwriting_region = re[0]
    handwriting_coord = coord2[0]

    re, coord3 = cutY(example_region, threshold01=6, space01=30, threshold10=3, space10=30)
    example_region = re[0]
    example_coord = coord[0]

    img = Image.fromarray(example_region)
    img.save('output/example.jpg')
    img = Image.fromarray(handwriting_region)
    img.save('output/handwriting.jpg')
    return example_region, handwriting_region, example_coord, handwriting_coord


def example_lines_get(example_region):
    re, coord = cutX(example_region, threshold01=10, space01=4, threshold10=8, space10=4)
    example_lines = re
    example_lines_coord = coord
    for i in range(len(re)):
        img = Image.fromarray(re[i])
        img.save('output/example_line' + str(i) + '.jpg')
    return example_lines, example_lines_coord


def handwriting_lines_get(handwriting_region):
    re, coord = cutX(handwriting_region, threshold01=10, space01=2, threshold10=10, space10=10)
    handwriting_lines = re[1:]
    handwriting_lines_coord = coord[1:]
    for i in range(len(re)):
        if (i == 0):
            continue
        img = Image.fromarray(re[i])
        img.save('output/handwriting_line' + str(i) + '.jpg')
    return handwriting_lines, handwriting_lines_coord


def example_words_get(example_line):
    re, coord = cutY(example_line, threshold01=2, space01=10, threshold10=2, space10=10)
    for i in range(len(re)):
        img = Image.fromarray(re[i])
        img.save('output/example_line_word' + str(i) + '.jpg')
    example_words = re
    example_words_coord = coord
    return example_words, example_words_coord


def handwriting_words_get(handwriting_line):
    re, coord = cutY(handwriting_line, threshold01=5, space01=10, threshold10=8, space10=10)
    for i in range(len(re)):
        img = Image.fromarray(re[i])
        img.save('output/handwriting_line_word' + str(i) + '.jpg')
    handwriting_words = re
    handwriting_words_coord = coord
    return handwriting_words, handwriting_words_coord


# 分割单词中的字母，就算是 example 即印刷体效果也很差，所以放弃了
def example_letters_get(example_words):
    re, coord = cutY(example_words, threshold01=0, space01=0, threshold10=2, space10=0)
    for i in range(len(re)):
        img = Image.fromarray(re[i])
        img.save('output/example_line_word_letter' + str(i) + '.jpg')
    example_letters = re
    example_letters_coord = coord
    return example_letters, example_letters_coord


# 手写体的字母分割，效果更差
def handwriting_letters_get(handwriting_words):
    # plt.imshow(handwriting_words)
    re, coord = cutY(handwriting_words, threshold01=5, space01=1, threshold10=8, space10=1)
    for i in range(len(re)):
        img = Image.fromarray(re[i])
        img.save('output/handwriting_line_word_letter' + str(i) + '.jpg')
    handwriting_letters = re
    handwriting_letters_coord = coord
    return handwriting_letters, handwriting_letters_coord


def calculate_word_distance(handwriting_words_coord):
    pass  # return word_distancs


def calculate_letter_size(handwriting_letter_coord):
    pass  # return letter_size


def calculate_letter_distance(handwriting_letter_coord):
    pass  # return letter_distance


def calculate_word_angle(handwriting_words):
    pass  # return handwriting_word_angle


def handwriting_content_recognition(handwriting_words):
    pass  # return handwriting_content


def example_content_recongintion(example_words):
    pass  # return example_content


def handwriting_style(handwriting_words):
    pass  # return handwriting_word_style


def handwriting_beauty(handwriting_words):
    pass  # return handwriting_beauty_result


def calculate_letter_size_score(letter_size):
    pass  # return letter_size_score, letter_size_error_index


def calculate_letter_distance_score(letter_distance):
    pass  # return letter_distance_score, letter_distance_error_index


def calculate_word_distance_score(word_distance):
    pass  # return word_distance_score, word_distance_error_index


def calculate_angle_score(handwriting_word_angle):
    pass  # return word_angle_score


def check_content(handwriting_content, example_content):
    pass  # return content_score


def see_handwriting_all(handwriting_region):
    pass  # return whole_score


def calculate_total_score():
    pass  # return final_score


def get_paper(filename):
    img = get_img_from_pdf(filename)
    img = preprocess(img)

    example_region, handwriting_region, example_coord, handwriting_coord = region_cut(img)
    example_paper = paper(example_region, example_coord)
    handwriting_paper = paper(handwriting_region, handwriting_coord)

    handwriting_lines, handwriting_lines_coords = handwriting_lines_get(handwriting_paper.get_img())
    for handwriting_line, handwriting_line_coord in zip(handwriting_lines, handwriting_lines_coords):
        line_temp = line(handwriting_line, handwriting_line_coord)
        handwriting_paper.add_line(line_temp)

    for handwriting_line in handwriting_paper.get_lines():
        handwriting_words, handwriting_words_coords = handwriting_words_get(handwriting_line.get_img())
        for handwriting_word, handwriting_word_coord in zip(handwriting_words, handwriting_words_coords):
            word_temp = word(handwriting_word, handwriting_word_coord)
            handwriting_line.add_word(word_temp)

    for handwriting_line in handwriting_paper.get_lines():
        for handwriting_word in handwriting_line.get_words():
            handwriting_letters, handwriting_letters_coords = handwriting_letters_get(handwriting_word.get_img())
            for handwriting_letter, handwriting_letter_coord in zip(handwriting_letters, handwriting_letters_coords):
                letter_temp = letter(handwriting_letter, handwriting_letter_coord)
                handwriting_word.add_letter(letter_temp)

    example_lines, example_lines_coords = example_lines_get(example_paper.get_img())
    for example_line, example_line_coord in zip(example_lines, example_lines_coords):
        line_temp = line(example_line, example_line_coord)
        example_paper.add_line(line_temp)

    for example_line in example_paper.get_lines():
        example_words, example_words_coords = example_words_get(example_line.get_img())
        for example_word, example_word_coord in zip(example_words, example_words_coords):
            word_temp = word(example_word, example_word_coord)
            example_line.add_word(word_temp)

    for example_line in example_paper.get_lines():
        for example_word in example_line.get_words():
            example_letters, example_letters_coords = example_letters_get(example_word.get_img())
            for example_letter, example_letter_coord in zip(example_letters, example_letters_coords):
                letter_temp = letter(example_letter, example_letter_coord)
                example_word.add_letter(letter_temp)
    return example_paper, handwriting_paper
