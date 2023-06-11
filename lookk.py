# import cv2
# path = "C:/Users/24511/PycharmProjects/pythonProject/ehw/look/1-5.jpg"
# pic = cv2.imread(path)
# cv2.imshow('imshow',pic)
# cv2.waitKey(0)
# pic2 = cv2.resize(pic,(640,640))
# cv2.imshow('imshow',pic2)
# cv2.waitKey(0)
# path2 = "C:/Users/24511/PycharmProjects/pythonProject/ehw/look/1.jpg"
# pic3 = cv2.imread(path2)
# cv2.imshow('imshow',pic3)
# cv2.waitKey(0)
# pic4 = cv2.resize(pic3,(640,640))
# cv2.imshow('imshow',pic4)
# cv2.waitKey(0)
import os
import shutil
path = "/home/pc/ehw/dataset/train_labels/"
pics = os.listdir(path)
# print(len(pics))
i=0
list=[]
for pic in pics:
    pic2=pic.split('-')
    # pic.split()

    if int(pic2[1])>=10:
        list.append(pics[i])
        # print(data)
    i=i+1
print(len(pics))
for pic3 in list:
    path2 = "/home/pc/ehw/dataset/train_labels/"+pic3
    shutil.copy(path2, "/home/pc/ehw/train_labels/")
print(list)

    # print(i)
# print(i)
# print(pics)
