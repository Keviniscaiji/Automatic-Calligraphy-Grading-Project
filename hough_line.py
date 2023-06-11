import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

img_dir_path = os.path.join("dataset","train_cutimg")
img_save_dir = os.path.join("dataset","intensify_img")


for img_name in os.listdir(img_dir_path):
    img_file_path = os.path.join(img_dir_path,img_name)
    print(img_file_path)
    img = cv2.imread(img_file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))
    # low_threshold = 50
    # high_threshold = 200
    # edges = cv2.Canny(gray_blur,low_threshold,high_threshold)
    ret, binary = cv2.threshold(gray_blur, 127, 255, cv2.THRESH_BINARY)
    binary = 255 - binary
    rho = 1
    theta = np.pi / 90
    threshold = 50
    min_line_length = 10
    max_line_gap = 1

    lines = cv2.HoughLinesP(binary, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    line_image = img
    if lines is None:
        continue
    xx1 = 0
    yy1 = 0
    xx2 = 0
    yy2 = 0
    slope = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (x1 != x2) and (y1 != y2) and (abs((y2 - y1) / (x1 - x2)))<1:
                # if (True):
                print(x1, y1, x2, y2)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                xx1 = xx1 + x1
                yy1 = yy1 + y1
                xx2 = xx2 + x2
                yy2 = yy2 + y2
                slope.append((y2 - y1) / (x1 - x2))
            # else:
            #     pass
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # if (yy1 != yy2) and (xx1 != xx2):
        # print((yy2 - yy1) / (xx1 - xx2))
    # print(np.median(np.array(slope)))
    # print(np.average(np.array(slope)))

    img_save_path = os.path.join(img_save_dir, img_name)
    os.makedirs(img_save_path, exist_ok=True)
    img_save_file_path = os.path.join(img_save_path, img_name)
    cv2.imwrite(img_save_file_path, line_image)
    # plt.imshow(line_image)
    # plt.show()
    break