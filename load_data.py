import os
from filecmp import cmp
from typing import Optional, Callable, Any, List, Tuple, Dict
from xml.etree.ElementInclude import default_loader

import cv2
import numpy as np
import torchvision
import tqdm as tqdm
from PIL import Image
import torch
from PIL.ImagePath import Path
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import xlrd
from torchvision.datasets import ImageFolder

import preprocess

base_path = os.path.join("dataset")  # dataset的根目录名
train_labels_path = os.path.join(base_path, "train_labels")
test_labels_path = os.path.join(base_path, "test_labels")
labels_path = os.path.join(base_path, "labels")
pdf_path = os.path.join(base_path, "pdf")
img_path = os.path.join(base_path, "img")
train_cutimg_path = os.path.join(base_path, "train_cutimg")
test_cutimg_path = os.path.join(base_path, "test_cutimg")
xlsx_name = '81-105.xlsx'  # xlsx的名字
# xlsx_name = '标注文件（得分计算） 2022.6.30.xlsx'  # xlsx的名字
sheet_name = "Sheet1"  # xlsx表格的名字
xlsx_path = os.path.join(base_path, xlsx_name)
train_loader_folders = ['cutimg']  # train_loader 加载的文件夹

transform = transforms.Compose([
    transforms.Resize([640, 640]),  # 转化为 640x640
    transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
])


def is_number(s):  # 检查文件名是否是数字
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def load_xlsx():  # 打开excel

    wb = xlrd.open_workbook(xlsx_path)
    sh = wb.sheet_by_name(sheet_name)
    # nrows = sh.nrows  # 有效数据行数
    # ncols = sh.ncols  # 有效数据列数
    dict = {}
    index = 0
    for i in range(1, sh.nrows):
        row = sh.row_values(i)
        # print(row)
        if is_number(row[0]):
            index = row[0]
        if row[10] == '最终得分：':
            dict[index] = [row[11]]
            dict[index].append(row[13])
            dict[index] += row[15:21]
    return dict


def generate_txt():  # 生成txt文件
    dict = load_xlsx()
    # 依照xlsx的信息生成txt文件
    for value in dict:
        array = dict[value]
        file_name = str(int(value)) + ".txt"
        f = open(os.path.join(labels_path, file_name), "w")
        i = 0
        while i < len(array):
            if i == len(array) - 1:
                f.write(str(array[i] / 100))
            else:
                f.write(str(array[i] / 10))
            if i < len(array) - 1:
                f.write("\n")
            i += 1
        f.close()


def load_img(p):
    file_names = os.listdir(p)
    pics = []
    for file_name in file_names:
        if (file_name.endswith("jpg") or file_name.endswith("JPG")) and is_number(file_name.split(".")[-2]):
            pics.append(os.path.join(p, file_name))
    return pics


def load_txt(p):  # 加载txt文件,返回一个tensor
    file_names = os.listdir(p)
    array = []
    for file_name in file_names:
        if is_number(file_name.split(".")[-2]):
            # 检查文件是否以数字命名
            f = open(os.path.join(p, file_name), "r")
            doc = f.read()
            sub_array = []
            for item in doc.split("\n"):
                sub_array.append(float(item))
            # array.append(sub_array[:5])  # 只需要前5个指标
            array.append(sub_array)
            # txt_dict[file_name.split(".")[-2]] = doc.split("\n")
            # print(doc.split("\n"))
            f.close()
    # print(array)
    return array


def load_txt_dic(p):  # 加载txt文件,返回一个tensor
    file_names = os.listdir(p)
    dict = {}
    for file_name in file_names:
        if is_number(file_name.split(".")[-2]):
            # 检查文件是否以数字命名
            f = open(os.path.join(p, file_name), "r")
            doc = f.read()
            sub_array = []
            for item in doc.split("\n"):
                sub_array.append(float(item))
            dict[f.name] = sub_array[:5]  # 只需要前5个指标
            f.close()
    # print(array)
    return dict


class MyDataset(Dataset):
    def __init__(self, img_p, labels_p, transform=None):
        super(MyDataset, self).__init__()
        self.label_root = labels_p
        self.img = load_img(img_p)
        self.label = load_txt(labels_p)

        # for line in data:
        #     line = line.rstrip()
        #     word = line.split()
        #     imgs.append(os.path.join(self.root, word[1], word[0]))
        #
        #     labels.append(word[1])
        # self.img = imgs
        # self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        # print(img)
        label = self.label[item]
        img = Image.open(img).convert('RGB')
        # 此时img是PIL.Image类型   label是str类型

        if transforms is not None:
            img = self.transform(img)
        label = np.array(label).astype(np.float)
        label = torch.from_numpy(label)

        return img, label


def generate_dataset(cutimg_path, label_path):
    dataset = MyDataset(cutimg_path, label_path, transform=transform)
    return dataset


def generate_loader(dataset):  # 生成train_loader

    # from torchvision import transforms
    #
    # transform = transforms.Compose([
    #     transforms.Resize(640),  # 将图片短边缩放至640，长宽比保持不变：
    #     transforms.CenterCrop(640),  # 将图片从中心切剪成3*640*640大小的图片
    #     transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
    # ])

    # dataset = MyDataset(cutimg_path, labels_path, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    return data_loader


def generate_img():
    """由pdf生成img"""
    preprocess.pdf2img_prcs(pdf_path, img_path)


def generate_cutimg():
    """由img生产手写部分cutimg"""
    preprocess.img2cutimg_prcs(img_path, cutimg_path)


# def

def load_data():
    generate_txt()
    # tensor = load_txt()
    label_array = load_txt()
    # print(label_array)
    train_loader = generate_loader()


def generate_ranking_txt():
    label_dict = load_txt_dic(labels_path)
    sorted_label_dict = sorted(label_dict.items(), key=lambda d: sum(d[1]), reverse=True)

    file_name = "score_ranking" + ".txt"
    f = open(os.path.join(base_path, file_name), "w")
    for i in sorted_label_dict:
        f.write(str(i[0].split("/")[-1]) + "\n")
    f.close()


def generate_bar(arr, form_name):

    dataset_size = len(arr)
    avg = 0
    for item in arr:
        avg += item
    avg /= dataset_size
    print("average score: "+str(avg)+ " "+ form_name)
    print("standard deviation",end=":")
    print(np.std(arr,ddof=1))
    # print("standard deviation "+str(np.std(arr,ddof=1)+ " "+ form_name))
    dict = {}
    for item in arr:
        if "% .2f" % item in dict:
            dict["% .2f" % item] += 1
        else:
            dict["% .2f" % item] = 1
    sorted_dict = sorted(dict.items(), key=lambda d: d[0], reverse=True)

    x_data = []
    y_data = []

    for item in sorted_dict:
        rag = item[0]  # range of score
        freq = item[1]
        x_data.append(str(rag))
        y_data.append(freq)

    # 正确显示中文和负号

    # 画图，plt.bar()可以画柱状图
    plt.figure(figsize=(11,5))
    for i in range(len(x_data)):
        plt.bar(x_data[i], y_data[i])

    # 设置图片名称
    plt.title(form_name)
    # 设置x轴标签名
    plt.xlabel("Range")
    # 设置y轴标签名
    plt.ylabel("Frequency")
    # 显示
    plt.savefig(os.path.join("figures", "bar_chart", '{}.png'.format(form_name)))
    plt.show()
    # print(sorted_dict)


def generate_scatter(arr, form_name, x_name, y_name):
    avg = 0
    dataset_size = len(arr)
    for item in arr:
        avg += item
    avg /= dataset_size
    arr_L1 = sum([abs(i - avg) for i in arr]) / dataset_size
    print("L1 score for {}:".format(y_name) + str(arr_L1))
    print("{} average:".format(y_name) + str(avg))
    x = range(1, dataset_size + 1)
    plt.title(form_name, fontsize=24)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel("{} score".format(y_name), fontsize=14)
    plt.plot([0, dataset_size], [avg, avg])
    plt.scatter(x, arr, 20)
    plt.savefig(os.path.join("figures","scatter_chart", '{}.png'.format(y_name)))
    plt.show()


def generate_figures():
    label_array = load_txt(labels_path)

    # 总分在66以上
    # 64 - 66
    # 62 - 64
    # 60 - 62
    # 已经60以下的内容一致性扣分情况
    accuracy_of_content_array_above_66  = []
    accuracy_of_content_array_64_66 = []
    accuracy_of_content_array_62_64 = []
    accuracy_of_content_array_60_62 = []
    accuracy_of_content_array_below_60 = []




    for item in label_array:
        final_score = item[7] *100
        if final_score>66:
            accuracy_of_content_array_above_66.append(item[5])
        elif final_score>64:
            accuracy_of_content_array_64_66.append(item[5])
        elif final_score>62:
            accuracy_of_content_array_62_64.append(item[5])
        elif final_score>60:
            accuracy_of_content_array_60_62.append(item[5])
        else :
            accuracy_of_content_array_below_60.append(item[5])

    generate_bar(accuracy_of_content_array_above_66, "Accuracy of content >66")
    generate_bar(accuracy_of_content_array_64_66, "Accuracy of content 64-66")
    generate_bar(accuracy_of_content_array_62_64, "Accuracy of content 62-64")
    generate_bar(accuracy_of_content_array_60_62, "Accuracy of content 60-62")
    generate_bar(accuracy_of_content_array_below_60, "Accuracy of content <60")
        # letter_size_array.append(item[0])
        # letter_spacing_array.append(item[1])
        # word_spacing_array.append(item[2])
        # inclination_array.append(item[3])
        # horizontal_angle_array.append(item[4])
        # total_array.append(sum(item))

    # generate_scatter(letter_size_array,"Score","File num","Letter size")
    # generate_scatter(letter_spacing_array, "Score", "File num", "Letter spacing")
    # generate_scatter(word_spacing_array, "Score", "File num", "Word spacing")
    # generate_scatter(inclination_array, "Score", "File num", "Inclination")
    # generate_scatter(horizontal_angle_array, "Score", "File num", "Horizontal angle")
    # generate_scatter(total_array, "Score", "File num", "Total")
    #
    # generate_bar(letter_size_array, "Letter size")
    # generate_bar(letter_spacing_array,  "Letter spacing")
    # generate_bar(word_spacing_array,  "Word spacing")
    # generate_bar(inclination_array,  "Inclination")
    # generate_bar(horizontal_angle_array, "Horizontal angle")
    # generate_bar(total_array, "Total")

def split_labels():
    path_names = os.listdir(labels_path)
    for path_name in path_names:
        label_path = os.path.join(labels_path, path_name)
        f_1 = open(label_path,"r")
        doc = f_1.read()
        line_count = 1
        for item in doc.split("\n"):
            dir_name = "labels_line_{}".format(str(line_count))
            new_label_path = os.path.join(dir_name, path_name)
            f_2 = open(os.path.join(base_path,new_label_path), "w")
            f_2.write(item)
            f_2.close()
            line_count += 1
        f_1.close()
if __name__ == '__main__':
    split_labels()
    # load_data()
    # generate_txt()
    # generate_cutimg()
    # generate_figures()
    # generate_ranking_txt()