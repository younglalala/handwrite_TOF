import cv2
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import  os

# img1_path='/Users/wywy/Desktop/test/33/7.jpg'
# save_path='/Users/wywy/Desktop/one_choice'
#
#
# image=cv2.imread(img1_path)
# # blur_image=cv2.medianBlur(image,3)
# # # hsv_image=cv2.cvtColor(blur_image,cv2.COLOR_BGR2HSV)
# gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#
# th2 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)  # ok
# # print(th2.shape)
# # # ret2,th2 = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# # cv2.imshow('test',th2)
# # cv2.waitKey(0)
#
#
# def crop_img(binary_img):
#     h,w=binary_img.shape
#     img=Image.fromarray(np.uint8(binary_img))
#     crop_x1=0
#     crop_y1=0
#
#     crop_x2=0
#     crop_y2=16
#     c=63
#     for i in range(7):
#         crop_x2=int(w/7*(i+1))
#         crop_x1=int(w/7*i)
#
#         crop_out=img.crop((crop_x1,crop_y1,crop_x2,crop_y2))
#         crop_out.save(save_path+'/'+str(c)+'.jpg')
#         c+=1
#     print(c)
#
#
#
#
#
#
#
#
#
#
#
# crop_img(th2)

#test
#
img_path='/Users/wywy/Desktop/test/33/1.jpg'
image=cv2.imread(img_path)
gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# th2 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
ret1, th1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
Cannyimg = cv2.Canny(th1, 35, 189)
# cv2.imshow('drawimg',Cannyimg)
# cv2.waitKey()

lines = cv2.HoughLinesP(Cannyimg, 1, np.pi / 180, threshold=5, minLineLength=4, maxLineGap=4)   #对图像进行霍夫变换得到边缘直线


for i in range(lines.shape[0]):   #循环得到的线条
    for x1, y1, x2, y2 in lines[i]:
        print(x1, y1, x2, y2,'-------')
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 1)

cv2.imshow('x', image)
cv2.waitKey(0)
image, cts, hierarchy = cv2.findContours( Cannyimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image,cnts, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(np.size(cnts))
print(cnts[0])
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

cv2.drawContours(image, cts, -1, (0,0,255), 1)
list=sorted(cts,key=cv2.contourArea,reverse=True)

print("寻找轮廓的个数：",len(cts))
cv2.imshow("draw_contours",image)

cv2.imshow('drawimg',image)










#
# img_path='/Users/wywy/Desktop/test/5'
# save_path='/Users/wywy/Desktop/test/55'
# c=0
# for file in os.listdir(img_path):
#     if file =='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         bg=Image.new('RGB',(168,16),'white')
#         img=Image.open(img_path+'/'+file)
#         img=img.resize((int(168/7*5),16))
#         bg.paste(img,(0,0))
#         bg.save(save_path+'/'+str(c)+'.jpg')
#         c+=1
# print(c)

# img_path='/Users/wywy/Desktop/test11.jpg'
# img = cv2.imread(img_path,0)
# img = np.array(img)
# mean = np.mean(img)
# img = img - mean
#
# img = img*1.5 + mean*1.5 #修对比度和亮度
# img = img/255.   #非常关键，没有会白屏
# cv2.imshow('pic',img)
# cv2.waitKey()






