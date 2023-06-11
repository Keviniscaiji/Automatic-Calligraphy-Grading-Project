import math
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

base_path = os.path.join("dataset")
intensify_img_path = os.path.join(base_path, "intensify_txt")
imgs_path = os.path.join(base_path, "test_cutimg")
black_thr = 127

def generate_line_info(img,max_break):
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #转为二值图
    ret, binary = cv2.threshold(gray, black_thr, 255, cv2.THRESH_BINARY)

    # 膨胀算法的色块大小
    h, w = binary.shape
    hors_k = int(math.sqrt(w)*1.2)


    # 白底黑字，膨胀白色横向色块，抹去文字和竖线，保留横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k,1))
    hors = ~cv2.dilate(binary, kernel, iterations = 1) # 迭代两次，尽量抹去文本横线，变反为黑底白线

    lines_mid_points =calc_line_info(hors,max_break)
    return lines_mid_points
'''
    计算线条的完整信息，包括线条的轴心位置、线条起止pos、内部线段的信息
    原理：通过坐标轴投影算法，获取垂直于该坐标轴的线段在该轴上的中心位置
    前提：输入的lines_matrix必须是i,j二维数组，且i为要投影的坐标轴（相当于cv图像的y轴，若要投影x轴请先转置后传入）
    输入：lines_matrix-线段的二值变反矩阵；max_break-最大支持线段中间出现的断裂（像素数量）
    输出：线段信息list

    line_info = {
        'axis': 0, # 线条轴心
        'wide': 0, # 线条粗细
        'len': 0,  # 线条总长度（线段长度之和）
        'segment': [], # 线条内部的线段信息 [[线段长度, 线段start, 线段end],...] 支持一根线条被分割为多条线段（中间跨域一个或多个合并单元格）
    }

    输入示例：lines_matrix
    i0-------------------------> j
    i1-------------------------> j
    i2-------------------------> j
    i3-------------------------> j
'''


def calc_line_info(lines_matrix, max_break, debug=False):
    # 计算i轴每个位置的投影是否有像素值
    project_i = [any(x) for x in lines_matrix]  # 对每个i对应的list进行any操作，求出i轴上该位置是否出现像素点
    lines_mid_points = []
    # 取出有像素值的i轴pos
    pos_i = [i for i, x in enumerate(project_i) if x == True]

    # 异常检测：若只检测到一条线或没有线 则返回空
    if len(pos_i) <= 1:
        return []

    # 将连续的pos分组（支持连续pos出现10个像素的断裂）
    pos_group_i = []
    temp_group = [pos_i[0]]  # 第一个pos默认满足要求，放入临时结果中

    # 可调参数 线段断层截面像素点
    for i in range(1, len(pos_i)):  # 从第二个pos开始计算
        if pos_i[i] - pos_i[i - 1] <= 5:  # 连续像素计为一组，支持截面断层
            temp_group.append(pos_i[i])
        else:
            pos_group_i.append(temp_group)  # 上一组结束，放入结果中
            temp_group = [pos_i[i]]  # 新一组第一个pos默认满足要求，放入临时结果中

    # 最后一组pos放入结果中
    pos_group_i.append(temp_group)

    '''
        线条信息 数据结构
    '''
    line_info = {
        'axis': 0,  # 线条轴心
        'wide': 0,  # 线条粗细
        'len': 0,  # 线条总长度（线段长度之和）
        'segment': [],  # 线条内部的线段信息 [[线段长度, 线段start, 线段end],...] 支持一根线条被分割为多条线段（中间跨域一个或多个合并单元格）
    }

    lines_info = []
    line_start_end = []

    for (i, poses) in enumerate(pos_group_i):
        info = line_info.copy()
        axis = int(np.median(poses))
        info['axis'] = axis

        # same_line = False
        # for item in lines_mid_points:
        #     if abs(item - axis)<120:
        #         same_line = True
        # if not same_line:
            # lines_mid_points.append(axis)

        info['wide'] = poses[-1] - poses[0] + 1

        '''
            计算图像中线段的长度（即在j轴像素点的个数）
            注意有可能一条线段被分割成了多个分段（中间出现了合并单元格），因此该算法需要返回线段长度list[]
            同时要支持干扰造成的线段中间出现的断裂
        '''

        # 获取该条线所在的矩阵 并转置
        area = np.transpose(lines_matrix[poses[0]:poses[-1] + 1])

        # 取得每个投影点的像素情况
        mask = [str(any(x) + 0) for x in area]  # any(x)+0可以将True False转为1 0

        # 将mask中连续的1分割出来，每段连续的1即为一条线段
        s = ''.join(mask).split('0')
        segs = [[i, len(v)] for (i, v) in enumerate(s) if len(v) > 0]

        # 调整每条线段在原始list中正确的pos（segs中的i只是s数组中的位置，并非mask中的位置）
        for i in range(1, len(segs)):
            # 每条线段的pos = segs中的i + 之前线段的总长度
            segs[i][0] = segs[i][0] + sum([x[1] - 1 for (j, x) in enumerate(segs) if j < i])

        segments = [segs[0]]  # 初始化为第一条线段

        # 若有多条线段，进行智能线段分析：因干扰造成的10像素内的断裂自动连在一起
        MAX_LEN_BREAK = max_break  # 最大支持线段断裂长度
        if len(segs) > 1:
            # 从第二条线段开始判断与上一条线段之间是断裂还是间隔
            for i in range(1, len(segs)):
                delta = segs[i][0] - segs[i - 1][0] - segs[i - 1][1]
                if delta < MAX_LEN_BREAK:
                    # 小于断裂长度，应连接线段
                    segments[-1] = [segments[-1][0], segs[i][0] - segments[-1][0] + segs[i][1]]
                else:
                    # 为间隔，是不同的线段
                    segments.append(segs[i])

        # 线段数据重组为：[线段长度, 线段start, 线段end]
        segments = [[x[1], x[0], x[0] + x[1]] for x in segments]



        info['segment'] = segments
        info['len'] = sum([x[0] for x in segments])

        if info["len"]>1800:
            lines_mid_points.append(info['axis'])

        lines_info.append(info)

    if debug:
        print('\n\n----------lines info------------')
        x = [print(v) for v in lines_info]

    # return lines_info
    return lines_mid_points
    # return line_start_end

if __name__ == '__main__':
    img = cv2.imread(os.path.join(imgs_path,"11.jpg"))
    max_break = 5
    print(generate_line_info(img,max_break))