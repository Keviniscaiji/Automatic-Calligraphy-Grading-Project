# 转换pdf为img与裁剪手写区域
from func import *
import cv2
import os


def get_hr_from_img(srcimg):
    """get handwriting part from img"""
    img_gray = cv2.cvtColor(srcimg, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.blur(img_gray, (3, 3))
    img_canny = cv2.Canny(img_blur, 10, 60)
    # img_contour, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 用于新版 opencv 只有两个返回值
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        rectPoint = cv2.minAreaRect(contour)
        # print(rectPoint)
        area = rectPoint[1][0] * rectPoint[1][1]
        if (area > 1500 * 1900):
            print(rectPoint)
            print(area)
            break
    if (abs(rectPoint[2]) > 80):
        x1 = int(rectPoint[0][1] - rectPoint[1][0] / 2)
        x2 = int(rectPoint[0][1] + rectPoint[1][0] / 2)
        y1 = int(rectPoint[0][0] - rectPoint[1][1] / 2)
        y2 = int(rectPoint[0][0] + rectPoint[1][1] / 2)
    else:
        x1 = int(rectPoint[0][1] - rectPoint[1][1] / 2)
        x2 = int(rectPoint[0][1] + rectPoint[1][1] / 2)
        y1 = int(rectPoint[0][0] - rectPoint[1][0] / 2)
        y2 = int(rectPoint[0][0] + rectPoint[1][0] / 2)
    img = srcimg[x1:x2, y1:y2]
    return img


def pdf2img_prcs(pdfdir, imgdir):
    """pdf 转 img 批处理"""
    for pdf_name in os.listdir(pdfdir):
        print(pdf_name)
        index_name = pdf_name[:-4]
        img_name = index_name + '.jpg'
        img = get_img_from_pdf(os.path.join(pdfdir, pdf_name), dpi=300)
        cv2.imwrite(os.path.join(imgdir, img_name), img)


def img2cutimg_prcs(imgdir, cutimgdir):
    """img 转手写部分 cut_img_overlap 批处理"""
    for img_name in os.listdir(imgdir):
        img = cv2.imread(os.path.join(imgdir, img_name))
        img_processed = get_hr_from_img(img)
        cv2.imwrite(os.path.join(cutimgdir, img_name), img_processed)
