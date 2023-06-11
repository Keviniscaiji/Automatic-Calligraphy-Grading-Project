import os
from shutil import copyfile
basePath = os.path.join("dataset")
testTxtPath = os.path.join(basePath,"test_list.txt")
trainTxtPath = os.path.join(basePath,"train_list.txt")
imgsPath = os.path.join(basePath,"imgs")
trainImgsPath = os.path.join(basePath,"train_imgs")
testImgsPath = os.path.join(basePath, "test_imgs")

trainCutImgsPath = os.path.join(basePath,"train_cutimg")
testCutImgsPath = os.path.join(basePath, "test_cutimg")
labelsPath = os.path.join(basePath,"cut_labels")
trainLabelsPath = os.path.join(basePath,"train_labels")
testLabelsPath = os.path.join(basePath,"test_labels")

def moveFiles():
    with open(testTxtPath, 'r', encoding='utf-8') as infile:
        fileNames = infile.readline().split(',');
        for fileName in fileNames:
            copyfile(os.path.join(imgsPath,fileName+".jpg"), os.path.join(trainImgsPath,fileName+".jpg"))
    infile.close()
    with open(trainTxtPath, 'r', encoding='utf-8') as infile:
        fileNames = infile.readline().split(',');
        for fileName in fileNames:
            copyfile(os.path.join(imgsPath,fileName+".jpg"), os.path.join(testImgsPath, fileName + ".jpg"))
    infile.close()

def moveTxtFiles():
    img_names = os.listdir(trainCutImgsPath)
    for imgName in img_names:
        txtName = imgName.replace(".jpg",".txt")
        copyfile(os.path.join(labelsPath,txtName), os.path.join(trainLabelsPath,txtName))

    img_names = os.listdir(testCutImgsPath)
    for imgName in img_names:
        txtName = imgName.replace(".jpg",".txt")
        copyfile(os.path.join(labelsPath,txtName), os.path.join(testLabelsPath,txtName))


if __name__ == '__main__':
    moveTxtFiles();