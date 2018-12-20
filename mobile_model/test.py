import os
import urllib.request
import cv2
import numpy as np
import random
from PIL import Image
from skimage import data, exposure, img_as_float
import math
import imutils


#获取学生拍照数据
txt_path='/Users/wywy/Desktop/链接.txt'
def geturlimage(txt_path):

    with open(txt_path) as f:
        lines=f.readlines()
        x=0
        for line in lines:
            urllib.request.urlretrieve(line.rsplit()[0], '/Users/wywy/Desktop/手机拍照数据/{}.jpg'.format(x))
            x += 1


#
# #直方图均衡化
# def hisEqulColor(img):
#     ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#     channels = cv2.split(ycrcb)
#     cv2.equalizeHist(channels[0], channels[0]) #equalizeHist(in,out)
#     cv2.merge(channels, ycrcb)
#     img_eq=cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
#     return img_eq
#
# # a=cv2.equalizeHist(img)
#
#
#单尺度计算

def singleScaleRetinex(img,sigma):
    temp=cv2.GaussianBlur(img,(0,0),sigma)
    gaussian=np.where(temp ==0, 0.01, temp)
    tetinex=np.log10(img+0.01) - np.log10(gaussian)

    return tetinex

#多尺度计算
def multiScaleRetinex(img,sigma_list):
    retinex=np.zeros_like(img*1.0)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img,sigma)
    retinex=retinex/len(sigma_list)

    return retinex


#颜色恢复

def colorRestoration(img,alpha,beta):
    img_sum=np.sum(img,axis=2,keepdims=True)
    color_restoration=beta*(np.log10(alpha*img)-np.log10(img_sum))

    return color_restoration


# MSRCR
low_clip=0.01
high_clip=0.99

def simplestColorBalance(img,low_clip,high_clip):
    total=img.shape[0]*img.shape[1]
    for i in range(img.shape[2]):
        unique,counts=np.unique(img[:,:,i],return_counts=True)
        current=0
        for u, c in zip(unique,counts):
            if float(current) /total<low_clip:
                low_val=u
            if float(current) /total<high_clip:
                high_val=u
            current+=c
        img[:,:,i]=np.maximum(np.minimum(img[:,:,i],high_val),low_val)

    return img

def MSRCR(img,sigma_list,G,b,alpha,beta,low_clip,high_clip):
    img=np.float64(img)+1.0

    img_retinex=multiScaleRetinex(img,sigma_list)
    img_color=colorRestoration(img,alpha,beta)
    img_msrcr=G*(img_retinex*img_color+b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:,:,i]=(img_msrcr[:,:,i]-np.min(img_msrcr[:,:,i]))/(np.max(img_msrcr[:,:,i]) - np.min(img_msrcr[:,:,i]))*255

    img_msrcr=np.uint8(np.minimum(np.maximum(img_msrcr,0),255))
    img_msrcr=simplestColorBalance(img_msrcr,low_clip,high_clip)

    return img_msrcr


def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    avgB = np.average(nimg[0])
    avgG = np.average(nimg[1])
    avgR = np.average(nimg[2])

    avg = (avgB + avgG + avgR) / 3

    nimg[0] = np.minimum(nimg[0] * (avg / avgB), 255)
    nimg[1] = np.minimum(nimg[1] * (avg / avgG), 255)
    nimg[2] = np.minimum(nimg[2] * (avg / avgR), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)





def ssr(img,sigma):

    #按公式进行计算
    _temp=cv2.GaussianBlur(img,(0,0),sigma)
    gaussian=np.where(_temp==0,0.01,_temp)
    img_ssr=np.log10(img+0.01) - np.log10(gaussian)

    #量化到0-255
    for i in range(img_ssr.shape[2]):
        img_ssr[:, :, i] = (img_ssr[:, :, i] - np.min(img_ssr[:, :, i])) / (
                    np.max(img_ssr[:, :, i]) - np.min(img_ssr[:, :, i])) * 255
        img_ssr=np.uint8(np.minimum(np.maximum(img_ssr,0),255))

        return img_ssr




# G=5
# b=25
# beta=46
# alpha=125
# sigma_list=[int(15),int(80),int(250)]
# sigma=1
#
#
# img_path='/Users/wywy/Desktop/test_data/4_save'
# save_path='/Users/wywy/Desktop/test_out'
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=cv2.imread(img_path+'/'+file)
#         # img=cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_AREA)
#         # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
#         # img_t = cv2.convertScaleAbs(img)
#         # gam2 = exposure.adjust_gamma(img_t, 0.3)
#         img=MSRCR(img,sigma_list,G,b,alpha,beta,low_clip,high_clip)
#         #
#         # # img=ssr(img, sigma)
#         img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#         # print(img_gray.shape)
#         #
#         k_size = int(img.shape[0] * 0.3), int(img.shape[1] / 4 * 0.4)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
#         closed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
#         _,threshold=cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         # closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, (int(img.shape[0] / 1), int(img.shape[1] / 1)))
#
#         # cnts = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         # cnts = cnts[1]
#         # contours_set = []
#         # choice_area = [(image.shape[1] / choice_num) * (i) for i in range(choice_num + 1)]
#         # if len(cnts) > 0:
#         #     for c in cnts:
#         #         rect = cv2.minAreaRect(c)  # 寻找外接矩形，返回（（中心点x，y），（w，h），矩形角度）
#         #         left_top_x, left_top_y, right_x, rigth_y = math.ceil(rect[0][0] - rect[1][0] / 2), math.ceil(
#         #             rect[0][1] - rect[1][1] / 2), math.ceil(rect[0][0] + rect[1][0] / 2), \
#         #                                                    math.ceil(rect[0][1] + rect[1][1] / 2)
#         #         w, h = rect[1][0], rect[1][1]
#         #         # 塞选条件为：宽度大于原始宽度的1/2高度小于原始图片的1/4的图片过滤掉
#         #         # if w > image.shape[1] / choice_num * 1.2 or h >= image.shape[0] or h <= image.shape[0] / 4:
#         #         #     pass
#         #         # else:
#         #         contours_set.append([left_top_x, left_top_y, right_x, rigth_y])
#         #
#         #         image = cv2.rectangle(img, (left_top_x, left_top_y), (right_x, rigth_y), (0, 255, 0), 2)
#
#         cv2.imwrite(save_path+'/'+file,threshold)



# img_path='/Users/wywy/Desktop/test_data/4'
# save_path='/Users/wywy/Desktop/test_data1/4'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(os.path.join(img_path,file))
#     else:
#         img=Image.open(os.path.join(img_path,file))
#         bg=Image.new('RGB',(img.size[0]+5,img.size[1]+5),'white')
#         bg.paste(img,(3,3))
#         bg.save(img_path+'/'+file)



# img_path='/Users/wywy/Desktop/add'
# save_path='/Users/wywy/Desktop/train_cls'
# c=606
# for i in range(50):
#     for file in os.listdir(img_path):
#         if file=='.DS_Store':
#             os.remove(img_path+'/'+file)
#         else:
#             name=file.split('.')[0].split('_')[-1]
#             img=Image.open(img_path+'/'+file)
#             img.save(save_path+'/aug'+str(c)+'_'+name+'.jpg')
#             # print(save_path+'/aug'+str(c)+'_'+name+'.jpg')
#             c+=1
#
# print(c)














