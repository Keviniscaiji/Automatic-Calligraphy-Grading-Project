import os
from shutil import copyfile

def remove_extra_files(img_path,cut_labels_path,new_cut_labels_path):
    '''
    :param img_path 图片的路径 img_new_path label的路径 label_origin_path 想要保存的新的label的路径
    :return: 依据img_path里面的图片将cut_labels_path里面的文件部分复制到new_cut_labels_path。
    '''
    base_path = os.path.join("dataset")

    img_path = os.path.join(base_path,img_path)

    cut_labels_path = os.path.join(base_path,cut_labels_path)

    new_cut_labels_path = os.path.join(base_path,new_cut_labels_path)

    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if not os.path.exists(cut_labels_path):
        os.mkdir(cut_labels_path)
    if not os.path.exists(new_cut_labels_path):
        os.mkdir(new_cut_labels_path)
    for path in os.listdir(img_path):
        try:
            new_filename = path.split("/")[-1].split(".")[0]+".txt"
            copyfile(os.path.join(cut_labels_path,new_filename), os.path.join(new_cut_labels_path,new_filename))
        except:
            print(path)
    print(len(os.listdir(new_cut_labels_path)))
    print(len(os.listdir(img_path)))


if __name__ == '__main__':
    remove_extra_files("train_cutimg_3_overlap_lines_patch", "labels_7parameters_3_line_overlap_patch",
                       "train_labels_7parameters_3_line_overlap_patch")
    remove_extra_files("test_cutimg_3_overlap_lines_patch", "labels_7parameters_3_line_overlap_patch",
                       "test_labels_7parameters_3_line_overlap_patch")

    remove_extra_files("train_cutimg_3_overlap_lines_patch", "labels_5parameters_3_line_overlap_patch",
                       "train_labels_5parameters_3_line_overlap_patch")
    remove_extra_files("test_cutimg_3_overlap_lines_patch", "labels_5parameters_3_line_overlap_patch",
                       "test_labels_5parameters_3_line_overlap_patch")

    # generate_ramdon_dataset("test_cutimg_5_overlap_lines", "labels_5parameters_5_lines_overlap",
    #                    "test_labels_5parameters_5_lines_overlap")
    # generate_ramdon_dataset("train_cutimg_5_overlap_lines", "labels_5parameters_5_lines_overlap",
    #                    "train_labels_5parameters_5_lines_overlap")

    # generate_ramdon_dataset("test_cutimg_5_overlap_lines", "labels_7parameters_5_lines_overlap",
    #                    "test_labels_7parameters_5_lines_overlap")
    # generate_ramdon_dataset("train_cutimg_5_overlap_lines", "labels_7parameters_5_lines_overlap",
    #                    "train_labels_7parameters_5_lines_overlap")




    # os.remove(os.path.join(img_origin_path,".DS_Store"))
    # print(len(os.listdir(img_new_path)))