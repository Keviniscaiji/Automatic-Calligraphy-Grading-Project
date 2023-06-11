

import random
import os
import cv2
import numpy as np

from cut_imgs_to_lines import generate_line_info, intensify_img

base_path = os.path.join("dataset")


def cut_img_ramdon_patch(num_of_lines, origin_imgs_path, new_img_path, origin_label_path, new_label_path,
                         per_pic_num):

    origin_imgs_path = os.path.join(base_path, origin_imgs_path)
    new_img_path = os.path.join(base_path, new_img_path)
    origin_label_path = os.path.join(base_path, origin_label_path)
    new_label_path = os.path.join(base_path, new_label_path)

    if not os.path.exists(new_img_path):
        os.mkdir(new_img_path)
    if not os.path.exists(new_label_path):
        os.mkdir(new_label_path)

    '''
    :param num_of_lines:需要截取的行数
    :return: 生成要求数量的文档并编号
    '''

    nol = num_of_lines
    img_names = os.listdir(origin_imgs_path)

    for img_name in img_names:

        img_path = os.path.join(origin_imgs_path, img_name)
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
            block_width_1 = y - y_list[y_index - num_of_lines]
            # block_width_1 = int (block_width_1)
            new_img = img[y_list[y_index - num_of_lines]:min((block_width_1 + y_list[y_index - num_of_lines]), height),
                      0:width]
            img_name_list = img_name.split(".")
            new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
            new_current_img_path = os.path.join(new_img_path, new_img_name)
            cv2.imwrite(new_current_img_path, new_img)


            # 将图片对应的label合并
            composite_score_list = []
            for label_index in range(y_index + 1 - num_of_lines, y_index + 1):
                origin_label_path_tem = img_name_list[0] + "-" + "1" + "-" + str(label_index) + "." + "txt"
                label_file = open(os.path.join(origin_label_path, origin_label_path_tem), 'r')
                data_list = label_file.read().split("\n")
                composite_score_list.append(data_list)
                label_file.close()

            new_label_file_list = []
            i = 0
            while i < len(composite_score_list[0]):
                tem_list = []
                for item in composite_score_list:
                    tem_list.append(float(item[i]))
                # print(tem_list)
                new_label_file_list.append(np.mean(tem_list))
                i += 1
            # new_label_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + ".txt"
            f = open(os.path.join(new_label_path, '{}-{}-{}.txt'.format(img_name_list[0], str(nol), str(file_num))),
                     'w')
            f.write('\n'.join(list(map(str, new_label_file_list))))
            f.close()
            file_num += 1

        for y in y_list[len(y_list)-num_of_lines+1:]:
            composite_score_list = []
            y_index = y_list.index(y)
            y_below = num_of_lines-(len(y_list) - y_index)

            new_img1 = img[y_list[y_index-1]:y_list[-1],
                      0:width]
            new_img2 = img[y_list[0]:y_list[y_below],
                      0:width]
            new_img = np.concatenate((new_img1, new_img2), axis=0)

            img_name_list = img_name.split(".")
            new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
            new_current_img_path = os.path.join(new_img_path, new_img_name)
            cv2.imwrite(new_current_img_path, new_img)

            label_index_list1 = [i for i in range(len(y_list)-(num_of_lines-y_below), len(y_list))]
            label_index_list2 = [i for i in range(1,y_below+1)]

            label_index_list = label_index_list1+label_index_list2
            print(label_index_list)

            # 将图片对应的label合并
            for label_index in label_index_list:
                origin_label_path_tem = img_name_list[0] + "-" + "1" + "-" + str(label_index) + "." + "txt"
                label_file = open(os.path.join(origin_label_path, origin_label_path_tem), 'r')
                data_list = label_file.read().split("\n")
                composite_score_list.append(data_list)
                label_file.close()
                # print(origin_label_path_tem)

            # for label_index in range(1, y_below):
            #     origin_label_path_tem = img_name_list[0] + "-" + "1" + "-" + str(label_index) + "." + "txt"
            #     label_file = open(os.path.join(origin_label_path, origin_label_path_tem), 'r')
            #     data_list = label_file.read().split("\n")
            #     composite_score_list.append(data_list)
            #     label_file.close()
                # print(origin_label_path_tem)

            new_label_file_list = []

            i = 0
            while i < len(composite_score_list[0]):
                tem_list = []
                for item in composite_score_list:
                    tem_list.append(float(item[i]))
                # print(tem_list)
                new_label_file_list.append(np.mean(tem_list))
                i += 1
            # new_label_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + ".txt"
            f = open(os.path.join(new_label_path, '{}-{}-{}.txt'.format(img_name_list[0], str(nol), str(file_num))),
                     'w')
            f.write('\n'.join(list(map(str, new_label_file_list))))
            f.close()
            file_num += 1

        location_index_list = [i for i in range(len(y_list)-1)]



        # 在正常剪裁之后随机生成文件

        while file_num <= per_pic_num:
            composite_score_list = []
            index_list = random.sample(location_index_list, num_of_lines)
            for label_index in index_list:
                label_index = label_index+1
                origin_label_path_tem = img_name_list[0] + "-" + "1" + "-" + str(label_index) + "." + "txt"
                label_file = open(os.path.join(origin_label_path, origin_label_path_tem), 'r')
                data_list = label_file.read().split("\n")
                composite_score_list.append(data_list)
                label_file.close()


            new_label_file_list = []
            i = 0

            while i < len(composite_score_list[0]):
                tem_list = []
                for item in composite_score_list:
                    tem_list.append(float(item[i]))
                new_label_file_list.append(np.mean(tem_list))
                i += 1

            f = open(os.path.join(new_label_path, '{}-{}-{}.txt'.format(img_name_list[0], str(nol), str(file_num))),
                     'w')
            f.write('\n'.join(list(map(str, new_label_file_list))))
            f.close()

            whole_img = img[y_list[index_list[0]]:y_list[index_list[0]+1],
                       0:width]
            for i in range(1, num_of_lines):
                new_img_part = img[y_list[index_list[i]]:y_list[index_list[i]+1],
                       0:width]
                whole_img = np.concatenate((whole_img, new_img_part), axis=0)
            new_img_name = img_name_list[0] + "-" + str(nol) + "-" + str(file_num) + "." + img_name_list[-1]
            new_current_img_path = os.path.join(new_img_path, new_img_name)
            cv2.imwrite(new_current_img_path, whole_img)

            file_num += 1

        print(img_name)


if __name__ == '__main__':
    # cut_img_ramdon_patch(3, os.path.join("2022_11_2dataset", "train_imgs", ),
    #                      os.path.join("2022_11_2dataset", "new_train_imgs"),
    #                      os.path.join("2022_11_2dataset", "labels_1line_patch"),
    #                      os.path.join("2022_11_2dataset", "new_train_labels")
    #                      , 50);
    cut_img_ramdon_patch(3, os.path.join("2022_11_7dataset", "train_imgs", ),
                         os.path.join("2022_11_7dataset", "new_train_imgs"),
                         os.path.join("2022_11_7dataset", "labels_1line_patch"),
                         os.path.join("2022_11_7dataset", "new_train_labels")
                         , 30);
    cut_img_ramdon_patch(3, os.path.join("2022_11_7dataset", "test_imgs", ),
                         os.path.join("2022_11_7dataset", "new_test_imgs"),
                         os.path.join("2022_11_7dataset", "labels_1line_patch"),
                         os.path.join("2022_11_7dataset", "new_test_labels")
                         , 12);
    # generate_random_dataset("labels_7parameters_3_line_overlap_patch","train_labels_7parameters_3_line_overlap_patch_1",
    #                         "cut_img_patch","train_cut_img_patch_1")
    # a = ["a","b","c"]
    # b = random.sample(a, 2)
    # print(b)

    # list = [i for i in range(1,5)]
    # print(list)


