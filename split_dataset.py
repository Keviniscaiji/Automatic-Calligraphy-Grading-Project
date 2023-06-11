import random
import shutil

total_img_num = 239
train_size = int(0.8 * total_img_num)
test_size = total_img_num - train_size
test_id_list = random.sample(range(1, total_img_num + 1), test_size)

path = 'dataset/'
for test_id in test_id_list:
    imgname = '{}.jpg'.format(test_id)
    labelname = '{}.txt'.format(test_id)
    shutil.move(path + 'cutimg/' + imgname, path + 'test_cutimg/' + imgname)
    shutil.move(path + 'labels/' + labelname, path + 'test_labels/' + labelname)
