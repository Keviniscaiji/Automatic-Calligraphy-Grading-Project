import os
import re
import shutil
from shutil import copyfile

import pandas as pd
import xlrd
from matplotlib import pyplot as plt


def remove_extra_files(img_path, cut_labels_path, new_cut_labels_path):
    '''
    :param img_path 图片的路径 img_new_path label的路径 label_origin_path 想要保存的新的label的路径
    :return: 依据img_path里面的图片将cut_labels_path里面的文件部分复制到new_cut_labels_path。
    '''
    base_path = os.path.join("dataset")

    img_path = os.path.join(base_path, img_path)

    cut_labels_path = os.path.join(base_path, cut_labels_path)

    new_cut_labels_path = os.path.join(base_path, new_cut_labels_path)
    #
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if not os.path.exists(cut_labels_path):
        os.mkdir(cut_labels_path)
    if not os.path.exists(new_cut_labels_path):
        os.mkdir(new_cut_labels_path)
    for path in os.listdir(img_path):
        try:
            new_filename = path.split("/")[-1].split(".")[0] + ".jpg"
            copyfile(os.path.join(cut_labels_path, new_filename), os.path.join(new_cut_labels_path, new_filename))
        except:
            print(path)
    print(len(os.listdir(new_cut_labels_path)))
    print(len(os.listdir(img_path)))


def normalize_txt(txt_path):
    '''
    :param txt_path 图片的路径 img_new_path label的路径 label_origin_path 想要保存的新的label的路径
    :return: 依据img_path里面的图片将cut_labels_path里面的文件部分复制到new_cut_labels_path。
    '''
    base_path = os.path.join("dataset", "2022_11_2dataset")

    txt_path = os.path.join(base_path, txt_path)

    #

    for path in os.listdir(txt_path):
        print(path)
        f = open(os.path.join(txt_path, path), 'r')
        score_list = f.read().split("\n")
        new_score_list = []
        for item in score_list:
            new_score_list.append(float(item) / 10);
        f.close()

        f = open(os.path.join(txt_path, path),
                 'w')
        f.write('\n'.join(list(map(str, new_score_list))))
        f.close()


def cal_file_list_score_reward(file_list):
    # 计算当前输入的列表的图片的得分
    '''

    :param file_list: 包含txt路径的列表
    :return: 是否分数超过要求的得分
    '''
    base_path = os.path.join("dataset", "2022_11_2dataset")
    file_list_len = len(file_list)
    score = 0
    for file in file_list:
        f = open(os.path.join(file), 'r')
        score_list = f.read().split("\n")
        for score_item in score_list:
            score += float(score_item)
    if score / file_list_len < 4:
        return False
    else:
        return True


def combine_files_by_pic(txt_path):
    '''

    :param file_list: txt的路径
    :return: 一个包含txt路径的字典
    '''
    base_path = os.path.join("dataset", "2022_11_2dataset")
    txt_path = os.path.join(base_path,txt_path)
    dic ={}
    for path in os.listdir(txt_path):
        path_list = path.split("-")
        if path_list[0] in dic:
            dic[path_list[0]].append(os.path.join(txt_path,path))
        else:
            dic[path_list[0]] = [os.path.join(txt_path,path)]
    return dic

def print_failed_data(dict):
    for key in dict:
        if not cal_file_list_score_reward(dict[key]):
            for item in dict[key]:
                os.remove(item)
                new_item = item.replace("new_test_labels","new_test_imgs")
                new_item = new_item.replace(".txt",'.jpg')
                os.remove(new_item)
                print(new_item)

def main():
    # 移除所有分数不达标的文件
    dic_1 = combine_files_by_pic("labels_1line_patch")
    dic_test = combine_files_by_pic(os.path.join("labels", 'train_labels'))
    dic_train = combine_files_by_pic(os.path.join("labels", 'test_labels'))
    for key in dic_1:
        if not cal_file_list_score_reward(dic_1[key]):
            if key in dic_test:
                for item in dic_test[key]:
                    # os.remove(item)
                    print(item)
            elif key in dic_train:
                for item in dic_train[key]:
                    # os.remove(item)
                    print(item)

def move_selected_files(csv_path, org_pic_path, dst_pic_path):
    wb = xlrd.open_workbook(csv_path)
    a = []
    for sh in wb.sheets():
        k = 1
        for i in range(1, sh.nrows):
            row = sh.row_values(i)
            a.append(row[1])
            print(k)
            # k是预测的排名
            k = k + 1
            txtName = str(row[1]) + ".txt"

    for i in a:
        txtName = str(int(i))+".txt"
        shutil.copyfile(os.path.join(org_pic_path, txtName), os.path.join(dst_pic_path, txtName))


    # # score_list = f.read().split("\n")
    # if not os.path.exists(dst_pic_path):
    #     os.mkdir(dst_pic_path)
    # print(score_list)
    # for item in score_list:
    #     pic_name = item+".jpg"
    #     try:
    #         shutil.move(os.path.join(org_pic_path, pic_name), os.path.join(dst_pic_path, pic_name))
    #     except:
    #         pass
    # print(dic_test)
    # cal_file_list_score_reward(dic["5"])
def rankSelectFiles():

    dic = {}
    txt_path = os.path.join("dataset","selected_labels")
    for path in os.listdir(txt_path):
        f = open(os.path.join(txt_path,path),"r")
        path = path.split('.')[0]
        score = float(f.read())
        if score in dic.keys():
            dic[score] = dic[score].append(path)
        else:
            dic[score] = [path]
        dicOrder = sorted(dic.items(), key=lambda x: x[0], reverse=True)
        f.close()
    # print(dicOrder)

    rankDic = {}
    index = 1
    for item in dicOrder:
        val = int(item[1][0])
        rankDic[val] = index
        index = index + 1
    # print(rankDic)

    csv_path = "dataset/test_score.xlsx"
    wb = xlrd.open_workbook(csv_path)
    a = []
    realRankDic = {}
    for sh in wb.sheets():
        k = 1
        for i in range(1, sh.nrows):
            row = sh.row_values(i)
            a.append(row[1])
            realRankDic[int(row[1])] = k
            # print(k)
            # k是预测的排名
            k = k + 1

    newModelDic = {}
    txt_path = os.path.join("dataset","score")
    for path in os.listdir(txt_path):
        f = open(os.path.join(txt_path,path),"r")
        path = path.split('.')[0]
        score = 0
        for line in f.readlines():
            score = score + float(line)
        newModelDic[score] = path
        f.close()
    newModelDicOrdered = sorted(newModelDic.items(), key=lambda x: x[0], reverse=True)

    newModelRankDic = {}
    index = 1
    for item in newModelDicOrdered:
        val = int(item[1])
        newModelRankDic[val] = index
        index = index + 1

    # print(newModelRankDic)

    # # print(realRankDic)
    import numpy as np

    realRankList = []
    rankList = []
    for i in realRankDic.keys():
        realRankList.append(realRankDic[i])
        rankList.append(rankDic[i])


    x1 = np.array(realRankList)
    y1 = np.array(rankList)
    r1 = np.corrcoef(x1, y1)
    print(r1)

    # sum = 0
    # for i in realRankDic.keys():
    #     # print(abs(realRankDic[i] - rankDic[i]))
    #     sum += abs(realRankDic[i] - rankDic[i])
    #     if(abs(realRankDic[i] - rankDic[i]) > 20):
    #         print(i,end=" ")
    # print()
    # print(sum)


    realRankList = []
    newModelRankList = []
    for i in realRankDic.keys():
        realRankList.append(realRankDic[i])
        newModelRankList.append(newModelRankDic[i])

    x2 = np.array(realRankList)
    y2 = np.array(newModelRankList)
    r2 = np.corrcoef(x2, y2)
    print(r2)

    # newModelSum = 0
    # for i in realRankDic.keys():
    #
    #     newModelSum += abs(realRankDic[i] - newModelRankDic[i])
    #     if(abs(realRankDic[i] - newModelRankDic[i]) > 20):
    #         print(i,end=" ")
    # print()
    # print(newModelSum)
    # print(newModelSum)

def readFileLineNums(path):
    pathList = re.split('[.-]',path)
    return pathList[-2]

def calLineScoreSum(orgPath):
    ScoreList = []
    ScoreListAbove40 = []
    ScoreListBelow40 = []
    # lineScoreSumDic = {}
    # lineNumDic = {}
    baseDic = 'dataset/labelsLine'
    for path in os.listdir(orgPath):
        scoreSum = 0
        # lineNum = readFileLineNums(path)
        f = open(os.path.join(baseDic,path),'r')
        for i in f.readlines()[:5]:
            scoreSum += float(i)
        ScoreList.append(scoreSum)
        if scoreSum >= 40:
            ScoreListAbove40.append(scoreSum)
        else:
            ScoreListBelow40.append(scoreSum)
        # if lineNum in lineScoreSumDic.keys():
        #     lineScoreSumDic[lineNum] += float(scoreSum)
        #     lineNumDic[lineNum] +=1
        # else:
        #     lineScoreSumDic[lineNum] = float(scoreSum)
        #     lineNumDic[lineNum] = 1
    # for key in lineScoreSumDic.keys():
    #     lineScoreSumDic[key] /= lineNumDic[key]
    # print(len(ScoreList))
    # sortedDic = sorted(lineScoreSumDic.items(),key=lambda x:int(x[0]),reverse = False)
    # lineNumList = []
    # lineScoreList = []
    # for item in sortedDic:
    #     lineNumList.append(item[0])
    #     lineScoreList.append(item[1])
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score between 0-50')
    plt.hist(ScoreList)
    plt.show()

    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score between 40-50')
    plt.hist(ScoreListAbove40,[40,42,44,46,48,50])
    plt.show()

    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score between 0-40')
    plt.hist(ScoreListBelow40)
    plt.show()

    # plt.xlabel("Num")
    # plt.ylabel("Average Score")
    # plt.bar(ScoreList)
    # plt.show()
    # fig, ax = plt.subplots()
    # bar_container = ax.bar(lineNumList, lineScoreList)
    # ax.set(label = "Average Line Score",ylim=(55, 66))
    # ax.bar_label(bar_container, fmt='%.1f')
    # plt.show()
    # print(lineScoreSumDic)
    # return lineScoreSumDic


if __name__ == '__main__':
    # calLineScoreSum('dataset/labelsLine')
    # main()
    # dic_1 = combine_files_by_pic("new_test_labels")
    # print_failed_data(dic_1)
    rankSelectFiles()

    # move_selected_files("dataset/test_score.xlsx", "dataset/cut_labels", "dataset/selected_labels")
    # remove_extra_files("test_cutimg",os.path.join("cut_to_one"),"new_test_cutimg")
    # remove_extra_files("train_cutimg",os.path.join("cut_to_one"),  "new_train_cutimgs")
    # normalize_txt(os.path.join("2022_11_2/new_train_labels"))
