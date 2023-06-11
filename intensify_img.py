import os

import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

base_path = os.path.join("dataset")
intensify_img_path = os.path.join(base_path, "intensify_img")
imgs_path = os.path.join(base_path, "train_cutimg")

def detect_horizontal_lines(img):
    '''
    :param img: cv2读取后的图片
    :return: 横线的横坐标
    '''

    #输入识别水平的直线并返回一个列表
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 50,150 为二值化时的阈值 apertureSize为Sobel滤波器的大小
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=300, maxLineGap=10)
    lis = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_exist = True
        try:
            if abs(y1 - y2) < 1:
                for item in lis:
                    if abs(item - y1) < 20 and abs(item - y2) < 20:
                        line_exist = False
                if line_exist:
                    lis.append(y2)
        except:
            pass
    return lis

def cut_img(num_of_lines):
    '''

    :param num_of_lines:需要截取的行数
    :return: 生成要求数量的文档并编号
    '''

    img_names = os.listdir(imgs_path)
    for img_name in img_names:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        width = img.shape[1]
        y_list = detect_horizontal_lines(img)
        y_list.sort()
        file_num = 1
        # print(y_list)
        # img = intensify_img(img) #将图像按照论文进行增强
        for y in y_list[num_of_lines:]:
            y_index = y_list.index(y)
            new_img= img[y_list[y_index - num_of_lines]:y + 30, 0:width]
            img_name_list = img_name.split(".")
            new_img_name = img_name_list[0]+"-"+str(file_num)+"."+img_name_list[-1]
            new_img_path = os.path.join(intensify_img_path, new_img_name)
            cv2.imwrite(new_img_path, new_img)
            file_num += 1
            # region.save(os.path.join(intensify_img_path, img_name.split(".")[0]+file_num+"."+img_name.split(".")[-1]))


def intensify_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 1) 将彩色照片转化为灰度图
    img_blur = cv2.blur(gray, (3, 3))# 2) 使用均值滤波，滤波器大小为 3*3
    ret, thresh = cv2.threshold(img_blur, 127, 255, 0, cv2.THRESH_BINARY) # 3) 将滤波后的图像进行二值化，二值化阈值设为127
    # cv2.imshow(thresh)
    return thresh
if __name__ == '__main__':
    # print(os.listdir())
    cut_img(6)
