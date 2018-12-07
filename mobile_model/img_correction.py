import numpy as np
import cv2
import os
from PIL import Image
import imutils


img_path='/Users/wywy/Desktop/crop_img1'
save_path='/Users/wywy/Desktop/vv'
c=0
for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.remove(img_path+'/'+file)
    else:
        img=cv2.imread(img_path+'/'+file)
        # tret2,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #高斯滤波
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        #自适应二值化方法
        # blurred=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,2)
        _, erosion = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite('/Users/wywy/Desktop/bb/{}.jpg'.format(c),erosion)
        c+=1
print(c)
        # edged = cv2.Canny(blurred, 10, 100)
        # cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        # docCnt = []
        # # 确保至少有一个轮廓被找到
        # if len(cnts) > 0:
        #     # 将轮廓按大小降序排序
        #     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        #     # 对排序后的轮廓循环处理
        #     for c in cnts:
        #         # 获取近似的轮廓
        #         peri = cv2.arcLength(c, True)
        #         approx = cv2.approxPolyDP(c, 0.12*peri, True)
        #         # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
        #         if len(approx) == 4:
        #             docCnt.append(approx)
        # #
        # print(docCnt)
        # # newimage=img.copy()
        # for i in docCnt:
        #     for j in i:
        # #
        # # #     #circle函数为在图像上作图，新建了一个图像用来演示四角选取
        #         cv2.circle(img, (j[0][0],j[0][1]), 10, (255, 0, 0), -1)



