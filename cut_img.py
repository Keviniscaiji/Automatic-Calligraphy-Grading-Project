# 一些工具函数 好像主要是用来垂直/水平投影后分割行/单词的
def cutX(arr, threshold01 = 6, space01 = 5, threshold10 = 3, space10 = 5):
    length = len(arr[0])
    re = []
    coord = []
    count = countX(arr)
    point = findPoint(count, threshold01 = threshold01, space01 = space01, threshold10 = threshold10, space10 = space10)
    lengthP=len(point)
    if lengthP%2==1:
        lengthP=lengthP-1
    for i in range(0,lengthP,2):
        re.append(arr[point[i]:point[i+1],0:length])
        coord.append([point[i], 0, point[i+1], length])
    return re, coord

def cutY(arr, threshold01 = 6, space01 = 5, threshold10 = 3, space10 = 5):
    length=len(arr)
    re=[]
    count =countY(arr)
    coord = []
    point =findPoint(count, threshold01 = threshold01, space01 = space01, threshold10 = threshold10, space10 = space10)
    #print(point)
    for i in range(0,len(point) // 2 * 2,2):
        re.append(arr[0:length,point[i]:point[i+1]])
        coord.append([0, point[i], length, point[i+1]])
    return re, coord

def countX(arr, threshold = 192):
    count=[0]*len(arr)
    for h in range(len(arr)):
        count2=0
        for l in range(len(arr[h])):
            if arr[h,l,0] < threshold:
                count2=count2+1
                arr[h,l]=[1]
        count[h]=count2
    return count

def countY(arr, threshold = 192):
    count=[0]*len(arr[0])
    for l in range(len(arr[0])):
        count2=0
        for h in range(len(arr)):
            if arr[h,l,0] < threshold:
                count2=count2+1
                arr[h,l]=[1]
        count[l]=count2
    return count

def findPoint(arr, threshold01 = 6, space01 = 5, threshold10 = 3, space10 = 5):
    index=-1
    re=[]
    zeroToMore(arr, index, re, threshold01 = threshold01, space01 = space01, threshold10 = threshold10, space10 = space10)
    return re

def zeroToMore(arr, index, re, threshold01 = 6, space01 = 5, threshold10 = 3, space10 = 5):
    if index<len(arr)-1:
        count = 0
        for i in range(index+1,len(arr)):
            if arr[i]>threshold01:
                count = count + 1
            else:
                count = 0
            if count > space01:
                index=i - space01
                re.append(index)
                break
            if i==len(arr)-1:
                index=i
        moreToZero(arr, index, re, threshold01 = threshold01, space01 = space01, threshold10 = threshold10, space10 = space10)
#同上  记录第一个零下标，表示一行的结束
def moreToZero(arr, index, re, threshold01 = 6, space01 = 5, threshold10 = 3, space10 = 5):
    if index<len(arr)-1:
        count = 0
        for i in range(index+1,len(arr)):
            if arr[i]<=threshold10:
                count = count + 1
            else:
                count = 0
            if count > space10:
                index=i
                re.append(index)
                break
            if i==len(arr)-1:
                index=i
        zeroToMore(arr, index, re, threshold01 = threshold01, space01 = space01, threshold10 = threshold10, space10 = space10)
