import numpy as np
import cv2
from PIL import Image


def CrossPoint(line1, line2):
    x0, y0, x1, y1 = line1[0]
    x2, y2, x3, y3 = line2[0]

    dx1 = x1 - x0
    print(dx1)
    dy1 = y1 - y0

    dx2 = x3 - x2
    dy2 = y3 - y2


    D1 = x1 * y0 - x0 * y1
    D2 = x3 * y2 - x2 * y3

    y = float(dy1 * D2 - D1 * dy2) / (dy1 * dx2 - dx1 * dy2)
    x = float(y * dx1 - D1) / dy1

    return (int(x), int(y))


def SortPoint(points):
    sp = sorted(points, key=lambda x: (int(x[1]), int(x[0])))
    if sp[0][0] > sp[1][0]:
        sp[0], sp[1] = sp[1], sp[0]

    if sp[2][0] > sp[3][0]:
        sp[2], sp[3] = sp[3], sp[2]

    return sp


def imgcorr(src):
    rgbsrc = src.copy()   #复制一张原始图片
    # graysrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)   #图像转换成灰度图
    # blurimg = cv2.GaussianBlur(src, (3, 3), 0)  #对图像进行高斯滤波
    # ret, thresh = cv2.threshold(blurimg, 127, 255, cv2.THRESH_BINARY)  #二值化 （减少其他线条的干扰）
    # Cannyimg = cv2.Canny(thresh, 35, 189)   #边缘检测
    # cv2.imshow('x', Cannyimg)
    gray_img = cv2.cvtColor(rgbsrc, cv2.COLOR_BGR2GRAY)
    # th2 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    ret1, th1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    Cannyimg = cv2.Canny(th1, 35, 189)

    #霍夫变换参数根据实际情况调整。特别是minLineLength和maxLineGap的参数。
    lines = cv2.HoughLinesP(Cannyimg, 1, np.pi / 180, threshold=200, minLineLength=80, maxLineGap=300)   #对图像进行霍夫变换得到边缘直线

    for i in range(lines.shape[0]):   #循环得到的线条
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(rgbsrc, (x1, y1), (x2, y2), (255, 255, 0), 1)   #可在原图上画线条看看是否检测准确
    cv2.imwrite("/Users/wywy/Desktop/output.jpg", rgbsrc)
    cv2.imshow('x', rgbsrc)
    cv2.waitKey(0)

    points = np.zeros((4, 2), dtype="float32")   #创建一个空矩阵
    points[0] = CrossPoint(lines[0], lines[2])
    points[1] = CrossPoint(lines[0], lines[3])
    points[2] = CrossPoint(lines[1], lines[2])
    points[3] = CrossPoint(lines[1], lines[3])

    sp = SortPoint(points)
    print(sp)

    width = int(np.sqrt(((sp[0][0] - sp[1][0]) ** 2) + (sp[0][1] - sp[1][1]) ** 2))
    height = int(np.sqrt(((sp[0][0] - sp[2][0]) ** 2) + (sp[0][1] - sp[2][1]) ** 2))
    dstrect = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]], dtype="float32")
    transform = cv2.getPerspectiveTransform(np.array(sp), dstrect)   #图像仿射变换   参数1；要变化的图像在原始图像上的位置，参数二，变化后图片的大小
    warpedimg = cv2.warpPerspective(src, transform, (width, height))   #透视变换,原始图片，放射变化后的参数，变换后图像的大小

    return warpedimg


if __name__ == '__main__':

    src = cv2.imread("/Users/wywy/Desktop/test.jpg")
    dst = imgcorr(src)
    # cv2.imshow("Image", dst)
    # cv2.waitKey(0)
    # cv2.imwrite("output.jpg", dst)


