import cv2
import numpy as np
import imutils
import os
from PIL import Image
from imutils.perspective import  four_point_transform
from skimage import data, exposure
#
# img_path='/Users/wywy/Desktop/crop_img'
# save_path='/Users/wywy/Desktop/crop_img1'
# cc=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         image = cv2.imread(img_path+'/'+'l16.jpg')
#         #转换为灰度图像
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         #高斯滤波
#         # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#         #自适应二值化方法
#         _, blurred = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         edged = cv2.Canny(blurred, 20, 100)
#         cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#         cnts = cnts[0] if imutils.is_cv2() else cnts[1]
#         docCnt = None
#         # 确保至少有一个轮廓被找到
#         all_area=[]
#         if len(cnts) > 0:
#             # 将轮廓按大小降序排序
#             cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#             # 对排序后的轮廓循环处理
#             for c in cnts:
#                 x, y, w, h = cv2.boundingRect(c)
#                 all_area.append(w*h)
#         index=int(np.where(np.array(all_area)==max(all_area))[0])
#
#         x1, y1, w1, h1=cv2.boundingRect(cnts[index])
#         img = cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
#         one_area=image[:,x1:x1 + w1][y1:y1 + h1,:]
#         cv2.imwrite(save_path+'/'+str(cc)+'.jpg',one_area)
#         cc+=1
# print(cc)

# img = cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)

# cv2.imwrite('/Users/wywy/Desktop/0.jpg',image)


img_path='/Users/wywy/Desktop/crop_img1'
save_path='/Users/wywy/Desktop/test3/4'
c=0
for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.remove(img_path+'/'+file)
    else:
        img=Image.open(img_path+'/'+file).convert('RGB')
        w,h=img.size
        h1=h/5
        for i in range(5):
            crop=img.crop((0,int(h1*i),int(w),int(h1*(i+1))))
            crop=crop.resize((168,32))
            crop.save(save_path+'/'+str(c)+'.jpg')
            c+=1
print(c)
