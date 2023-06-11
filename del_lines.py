import os

base_path = os.path.join("dataset")


def del_lines(base_file_path):
    for file_path in os.listdir(base_file_path):
        file = open(os.path.join(base_file_path, file_path))
        lines = file.readlines()
        print(lines)
        # del lines[-3:-1]  # 删除最后一行
        # del lines[0:16]  # 删除第1行到第17行
        file.close()
        file_new = open(os.path.join(base_file_path, file_path), 'w')
        file_new.writelines(lines[0:5])  # 将删除行后的数据写入文件
        file_new.close()


if __name__ == '__main__':
    del_lines(os.path.join(base_path, "new_train_labels_1"))
    del_lines(os.path.join(base_path, "new_train_labels_2"))
    del_lines(os.path.join(base_path, "new_train_labels_3"))
    del_lines(os.path.join(base_path, "new_test_labels_1"))
    del_lines(os.path.join(base_path, "new_test_labels_2"))
    del_lines(os.path.join(base_path, "new_test_labels_3"))
