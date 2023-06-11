import os
import PIL.Image as Image
base_path = os.path.join("dataset")


black_thr = 127


def detect_horizontal_lines(img):
    '''
    :param img: cv2读取后的图片
    :return: 横线的横坐标
    '''

    # 输入识别水平的直线并返回一个列表
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


def cut_img_overlap(num_of_lines, imgs_path, intensify_img_path):
    imgs_path = os.path.join(base_path,imgs_path)

    intensify_img_path = os.path.join(base_path, intensify_img_path)
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)
    if not os.path.exists(intensify_img_path):
        os.mkdir(intensify_img_path)

    '''
    :param num_of_lines:需要截取的行数
    :return: 生成要求数量的文档并编号
    '''
    nol = num_of_lines
    img_names = os.listdir(imgs_path)
    for img_name in img_names:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        y_list = generate_line_info(img, max_break=2)
        y_list.sort()
        file_num = 1
        # print(y_list)
        img = intensify_img(img)  # 将图像按照论文进行增强
        # 对图像进行分割
        for y in y_list[num_of_lines:]:
            y_index = y_list.index(y)
            # 等比例缩放图片
            # new_width = int(width*((y-y_list[y_index - num_of_lines])/height))
            block_width = y-y_list[y_index - num_of_lines]
            # block_width = int (block_width)
            new_img = img[y_list[y_index - num_of_lines]:min((block_width+y_list[y_index - num_of_lines]),height), 0:width]
            img_name_list = img_name.split(".")
            new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
            new_img_path = os.path.join(intensify_img_path, new_img_name)
            cv2.imwrite(new_img_path, new_img)
            file_num += 1
            # region.save(os.path.join(new_img_path, img_name.split(".")[0]+file_num+"."+img_name.split(".")[-1]))


def cut_img_overlap_patch(num_of_lines,imgs_path,intensify_img_path):
    imgs_path = os.path.join(base_path,imgs_path)
    intensify_img_path = os.path.join(base_path, intensify_img_path)
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)
    if not os.path.exists(intensify_img_path):
        os.mkdir(intensify_img_path)

    '''
    :param num_of_lines:需要截取的行数
    :return: 生成要求数量的文档并编号
    '''
    nol = num_of_lines
    img_names = os.listdir(imgs_path)
    for img_name in img_names:
        try:
            img_path = os.path.join(imgs_path, img_name)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            y_list = generate_line_info(img, max_break=2)
            y_list.sort()
            file_num = 1
            # print(y_list)
            img = intensify_img(img)  # 将图像按照论文进行增强
            # 对图像进行分割
            for y in y_list[num_of_lines:]:
                y_index = y_list.index(y)
                # 等比例缩放图片
                # new_width = int(width*((y-y_list[y_index - num_of_lines])/height))
                block_width = y-y_list[y_index - num_of_lines]
                # block_width = int (block_width)
                new_img = img[y_list[y_index - num_of_lines]:min((block_width+y_list[y_index - num_of_lines]),height), 0:width]
                img_name_list = img_name.split(".")
                new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
                new_img_path = os.path.join(intensify_img_path, new_img_name)
                cv2.imwrite(new_img_path, new_img)
                file_num += 1
                # region.save(os.path.join(new_img_path, img_name.split(".")[0]+file_num+"."+img_name.split(".")[-1]))

            new_img1 = img[y_list[-3]:y_list[-1],
                      0:width]
            new_img2 =img[y_list[0]:y_list[1],
                      0:width]
            im1 = np.concatenate((new_img1, new_img2), axis=0)
            img_name_list = img_name.split(".")
            new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
            new_img_path = os.path.join(intensify_img_path, new_img_name)
            cv2.imwrite(new_img_path, im1)
            file_num += 1

            new_img1 = img[y_list[-2]:y_list[-1],
                      0:width]
            new_img2 =img[y_list[0]:y_list[2],
                      0:width]
            im1 = np.concatenate((new_img1, new_img2), axis=0)
            img_name_list = img_name.split(".")
            new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
            new_img_path = os.path.join(intensify_img_path, new_img_name)
            cv2.imwrite(new_img_path, im1)
            print(img_name + " cut {} lines".format(str(num_of_lines)))
        except:
            pass


def cut_img_without_overlap(interval,imgs_path, intensify_img_path):

    '''
    :param num_of_lines:需要生成的图片数目
    :return: 生成要求数量的文档并编号
    '''

    imgs_path = os.path.join(base_path,imgs_path)
    intensify_img_path = os.path.join(base_path, intensify_img_path)
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)
    if not os.path.exists(intensify_img_path):
        os.mkdir(intensify_img_path)

    nol = interval
    img_names = os.listdir(imgs_path)
    for img_name in img_names:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        # 获取图片的宽及高
        y_list = generate_line_info(img, max_break=2)
        # 获取横线的纵坐标
        y_list.sort()
        file_num = 1
        # print(y_list)
        img = intensify_img(img)
        # 将图像按照论文进行增强
        num_of_line = len(y_list)
        # interval = int(num_of_line/num_of_imgs)
        i = interval
        while i < num_of_line:
            block_width = y_list[i]-y_list[i - interval]
            block_width = int (block_width)
            new_img = img[y_list[i - interval]:min((block_width+y_list[i - interval]),height), 0:width]
            i += interval
            img_name_list = img_name.split(".")
            new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
            new_img_path = os.path.join(intensify_img_path, new_img_name)
            cv2.imwrite(new_img_path, new_img)
            file_num += 1
        print(img_name + " cropped to {} lines pics".format(str(interval)))

def cut_img_cover_black(interval, imgs_path, intensify_img_path):
    imgs_path = os.path.join(base_path,imgs_path)
    intensify_img_path = os.path.join(base_path, intensify_img_path)
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)
    if not os.path.exists(intensify_img_path):
        os.mkdir(intensify_img_path)
    '''
    :param num_of_lines:需要生成的图片数目
    :return: 生成要求数量的文档并编号
    '''
    nol = interval
    img_names = os.listdir(imgs_path)
    for img_name in img_names:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        img_PIL = Image.open(img_path)
        height, width = img.shape[:2]


        # img_PIL = Image.open(img_origin_path)
        # img_array = img.load()
        # for x in range(0, width):
        #     for y in range(0, height):
        #         rgb = img_array[x, y]
        #         r = rgb[0]
        #         g = rgb[1]
        #         b = rgb[2]
        #         if b > 130 and r < 120:
        #             img_array[x, y] = (255, 0, 0)
        # img.save(img_origin_path)

        # 获取图片的宽及高
        y_list = generate_line_info(img, max_break=2)
        # 获取横线的纵坐标
        y_list.sort()
        file_num = 1
        # print(y_list)
        img = intensify_img(img)
        # 将图像按照论文进行增强
        num_of_line = len(y_list)
        # interval = int(num_of_line/num_of_imgs)
        i = interval

        while i < num_of_line:
        # img_PIL = Image.open(os.path.join("dataset", "test", "1.jpg"))
            width_PIL, height_PIL = img_PIL.size
            img_array = img_PIL.load()
            img1 = np.zeros(((y_list[i-interval]-0), width_PIL), dtype=np.uint8)  # 生成一个黑底背景
            img_black1 = Image.fromarray(img1)
            img2 = np.zeros(((height_PIL-y_list[i]), width_PIL), dtype=np.uint8)  # 生成一个黑底背景
            img_black2 = Image.fromarray(img2)
            img_PIL.paste(img_black1, (0, 0))
            img_PIL.paste(img_black2,(0,y_list[i]))

            i += interval
            img_name_list = img_name.split(".")
            new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
            new_img_path = os.path.join(intensify_img_path, new_img_name)
            img_PIL.save(os.path.join(new_img_path))
            file_num += 1
            img_PIL = Image.open(img_path)
        print(new_img_path)

        # for x in range(0, width_PIL):
        #         for y in range(0, height_PIL):
        #             rgb = img_array[x, y]
        #             r = rgb[0]
        #             g = rgb[1]
        #             b = rgb[2]
        #             if y > y_list[i] or y < y_list[i - interval]:
        #                 # print(y_list[i],y_list[i - interval])
        #                 img_array[x, y] = (0, 0, 0)
        #     i += interval
        #
        #     img_name_list = img_name.split(".")
        #     new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
        #     new_img_path = os.path.join(new_img_path, new_img_name)
        #     img_PIL.save(os.path.join(new_img_path))
        #     file_num += 1
        #     print(new_img_path)

        # while i < num_of_line:
        #     img[0:height,0,width]
        #     new_img = img[y_list[i - interval]:y_list[i], 0:width]
        #     i += interval
        #     img_name_list = img_name.split(".")
        #     new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
        #     new_img_path = os.path.join(new_img_path, new_img_name)
        #     cv2.imwrite(new_img_path, new_img)
        #     file_num += 1
        # print(img_name + " cropped to {} lines pics".format(str(interval)))
        # for y in y_list[num_of_lines:]:
        #     y_index = y_list.index(y)
        #     # 等比例缩放图片
        #     # new_width = int(width*((y-y_list[y_index - num_of_lines])/height))
        #     new_img = img[y_list[y_index - num_of_lines]:y, 0:width]
        #     img_name_list = img_name.split(".")
        #     new_img_name = img_name_list[0]+ "-" + str(nol)  + "-" + str(file_num) + "." + img_name_list[-1]
        #     new_img_path = os.path.join(new_img_path, new_img_name)
        #     cv2.imwrite(new_img_path, new_img)
        #     file_num += 1
        # region.save(os.path.join(new_img_path, img_name.split(".")[0]+file_num+"."+img_name.split(".")[-1]))
        # print(img_name+" cut {} lines".format(str(num_of_lines)))


def intensify_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 1) 将彩色照片转化为灰度图
    img_blur = cv2.blur(gray, (3, 3))  # 2) 使用均值滤波，滤波器大小为 3*3
    ret, thresh = cv2.threshold(img_blur, 127, 255, 0, cv2.THRESH_BINARY)  # 3) 将滤波后的图像进行二值化，二值化阈值设为127
    return thresh


import math
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# base_path = os.path.join("dataset")
# new_img_path = os.path.join(base_path, "intensify_txt")
# origin_imgs_path = os.path.join(base_path, "test_cutimg")


def generate_line_info(img, max_break):
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 转为二值图
    ret, binary = cv2.threshold(gray, black_thr, 255, cv2.THRESH_BINARY)

    # 膨胀算法的色块大小
    h, w = binary.shape
    hors_k = int(math.sqrt(w) * 1.2)

    # 白底黑字，膨胀白色横向色块，抹去文字和竖线，保留横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
    hors = ~cv2.dilate(binary, kernel, iterations=1)  # 迭代两次，尽量抹去文本横线，变反为黑底白线

    lines_mid_points = calc_line_info(hors, max_break)
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

        if info["len"] > 1000:
            lines_mid_points.append(info['axis'])

        lines_info.append(info)

    if debug:
        print('\n\n----------lines info------------')
        x = [print(v) for v in lines_info]

    # return lines_info
    return lines_mid_points
    # return line_start_end


# if __name__ == '__main__':
#     img = cv2.imread(os.path.join(origin_imgs_path, "48.jpg"))
#     max_break = 5
#     print(generate_line_info(img, max_break))

if __name__ == '__main__':
    # print("")
    # cut_img_overlap(1, "test_cutimg", "test_cutimg_1_overlap_lines");
    # cut_img_overlap(1, "train_cutimg", "train_cutimg_1_overlap_lines");
    # cut_img_without_overlap(1,"test_cutimg", "test_cutimg_1_lines");
    # cut_img_without_overlap(1, "train_cutimg", "train_cutimg_1_lines");

    cut_img_overlap_patch(3, "test_imgs", "test_cutimg");
    cut_img_overlap_patch(3, "train_imgs", "train_cutimg");

    # cut_img_without_overlap(2,"test_cutimg", "test_cutimg_2_lines");
    # cut_img_without_overlap(2, "train_cutimg", "train_cutimg_2_lines");
    # cut_img_overlap(3, "test_cutimg", "test_cutimg_3_overlap_lines");
    # cut_img_overlap(3, "train_cutimg", "train_cutimg_3_overlap_lines");
    # cut_img_overlap(4, "new_test_cutimg", "test_cutimg_4_overlap_lines");
    # cut_img_overlap(4, "new_train_cutimg", "train_cutimg_4_overlap_lines");
    # cut_img_overlap(5, "new_test_cutimg", "test_cutimg_5_overlap_lines");
    # cut_img_overlap(5, "new_train_cutimg", "train_cutimg_5_overlap_lines");
    # cut_img_without_overlap(3,"test_cutimg", "test_cutimg_3_lines");
    # cut_img_without_overlap(3, "train_cutimg", "train_cutimg_3_lines");
    # cut_img_without_overlap(1, "test_cutimg", "test_cutimg_1_line");
    # cut_img_without_overlap(1, "test_cutimg", "test_cutimg_1_line");
    # cut_img_cover_black(3)
    # cut_img_without_overlap(2)
    # cut_img_without_overlap(3)

    # width = 640
    # height = 640
    #
    # img = cv2.imread(os.path.join(new_img_path,"1-10-1.jpg"))
    # # 例如cv.imread("test/1.jpg")
    #
    # img = cv2.resize(img, (width, height))
    # # 默认使用双线性插值法
    #
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

