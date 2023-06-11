import torch
import torch.nn as nn
# x= torch.randn(4,5)
# y= torch.randn(4,5)
# print(x)
def get_level(x):
    level0 = []
    sum = abs(torch.sum(x))
    # print(sum)
    # # for item in sum:
    #     if item >= 4.4:
    #         level = 1
    #     elif item >=4.0:
    #         level = 2
    #     else :
    #         level = 4
    #     return level
    if sum >= 4.4:
        level = 1
    elif sum >= 4.0:
        level = 2
    else:
        level = 4
    return level

def lossfunction(predict, label):
    loss = 0.0
    for item in range(predict.size(0)):
        pre_level = get_level(predict[item])
        # print(pre_level)
        label_level = get_level(label[item])
        # print(label_level)
        sub_level = abs(pre_level-label_level)
        # print(sub_level)
        if sub_level == 0 :
            loss += 0.5*torch.sum(abs(predict[item]-label[item]))
        if sub_level == 1 :
            loss += torch.sum(abs(predict[item]-label[item]))
        if sub_level >= 2:
            loss += torch.sum(torch.exp(abs(predict[item]-label[item]))-1)
        loss = loss/predict.size(0)
        return loss


# loss = lossfunction(x,y)
# print(loss)