import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

import load_data
import pandas as pd

'''定义超参数'''
batch_size = 32  # 批的大小
learning_rate = 1e-2  # 学习率
num_epoches = 100000  # 遍历训练集的次数

'''训练集'''
# 标签
# load_data.generate_txt()
# load_data.generate_img()  # 转pdf为img
# load_data.generate_cutimg()  # 裁剪手写区域

train_loader_folders = ['cutimg']  # train_loader 加载的文件夹
test_loader_folders = ['cutimg']  # test_loader 加载的文件夹

train_dataset = load_data.generate_dataset(load_data.train_cutimg_path, load_data.train_labels_path)
test_dataset = load_data.generate_dataset(load_data.test_cutimg_path, load_data.test_labels_path)
train_loader = load_data.generate_loader(train_dataset)
test_loader = load_data.generate_loader(test_dataset)

info_df = pd.DataFrame(columns=['loss',  'test_loss'])

'''定义网络模型'''


class English_net(nn.Module):
    def __init__(self):
        super(English_net, self).__init__()
        block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm2d(num_features=64),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  ##160*160
            nn.Dropout(0.2)
        )

        block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  ####80*80
            nn.Dropout(0.2)
        )

        block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  ###40*40
            nn.Dropout(0.2)
        )

        block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )  ###20*20
        self.features = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
        )
        self.subnet1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10*10
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.subnet2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10*10
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.subnet3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10*10
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.subnet4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10*10
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.subnet5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10*10
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  ##5*5
        )

        self.classifier1 = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(53248, 128),
            nn.Linear(12800, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, 1),
        )

        self.classifier2 = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(53248, 128),
            nn.Linear(12800, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, 1),
        )

        self.classifier3 = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(53248, 128),
            nn.Linear(12800, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, 1),
        )

        self.classifier4 = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(53248, 128),
            nn.Linear(12800, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, 1),
        )

        self.classifier5 = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(53248, 128),
            nn.Linear(12800, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x1 = self.features[0](x)
        x2 = self.features[1](x1)
        x3 = self.features[2](x2)
        x4 = self.features[3](x3)
        x5 = self.subnet1(x4)
        x6 = self.subnet2(x4)
        x7 = self.subnet3(x4)
        x8 = self.subnet4(x4)
        x9 = self.subnet5(x4)
        score1 = x5.view(x5.size(0), -1)
        score1_ = self.classifier1(score1)
        score2 = x6.view(x6.size(0), -1)
        score2_ = self.classifier2(score2)
        score3 = x7.view(x7.size(0), -1)
        score3_ = self.classifier3(score3)
        score4 = x8.view(x8.size(0), -1)
        score4_ = self.classifier4(score4)
        score5 = x9.view(x9.size(0), -1)
        score5_ = self.classifier5(score5)
        return torch.cat([score1_, score2_, score3_, score4_, score5_], axis=1)


'''创建model实例对象，并检测是否支持使用GPU'''
model = English_net()
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()

'''定义loss和optimizer'''
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

'''训练模型'''

for epoch in range(num_epoches):
    print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)  # .format为输出格式，formet括号里的即为左边花括号的输出
    running_loss = 0.0
    # running_acc = 0.0
    for i, data in tqdm(enumerate(train_loader, 1)):
        img, label = data
        # print(img.size())
        # print(label.size())
        # cuda
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # 向前传播
        out = model(img)
        # print(out.size())
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        # _, pred = torch.max(out, 1)  # 预测最大值所在的位置标签
        # num_correct = (pred == label).sum()
        # accuracy = (pred == label).float().mean()
        # running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += 1
    print('Finish {} epoch, Loss: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset))))
    info_df.loc[epoch + 1, 'loss'] = running_loss / (len(train_dataset))

    model.eval()  # 模型评估
    eval_loss = 0
    # eval_acc = 0
    for data in test_loader:  # 测试模型
        img, label = data
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        print(out)
        print(label)
        eval_loss += loss.item() * label.size(0)
        # _, pred = torch.max(out, 1)
        # num_correct = (pred == label).sum()
        # eval_acc += num_correct.item()
    print('Test Loss: {:.6f}'.format(eval_loss / (len(test_dataset))))
    info_df.loc[epoch + 1, 'test_loss'] = eval_loss / (len(test_dataset))
    info_df.to_csv('info.csv')
    print()

# 保存模型
torch.save(model.state_dict(), './cnn.pth')
