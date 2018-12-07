import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance,ImageStat,ImageFilter
import random
import cv2
from numpy import fft
import scipy.misc
from skimage import exposure

#解析XML文件

# def parse_rec(filename):
#   """ Parse a PASCAL VOC xml file """
    # tree = ET.parse(filename)
    # objects = []
    # for obj in tree.findall('object'):
    #     obj_struct = {}
    #     obj_struct['name'] = obj.find('name').text
    #     obj_struct['pose'] = obj.find('pose').text
    #     obj_struct['truncated'] = int(obj.find('truncated').text)
    #     obj_struct['difficult'] = int(obj.find('difficult').text)
    #     bbox = obj.find('bndbox')
    #     obj_struct['bbox'] = [int(bbox.find('xmin').text),
    #                           int(bbox.find('ymin').text),
    #                           int(bbox.find('xmax').text),
    #                           int(bbox.find('ymax').text)]
    #     objects.append(obj_struct)
    #
    # return objects






#
# #ap为评价模型的一种方式。（单个类别平均精度）
# def voc_ap(rec, prec, use_07_metric=False):
#
#   """ rec:recall值，召回率
#       prec:精度
#   把准确率在recall值为Recall = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}时（总共11个rank水平上），求平均值：
#
# AP = 1/11 ∑ recall∈{0,0.1,…,1} Precision(Recall)
#
# 这样，在不同的recall水平上，平均的准确率给了模型更好的评估。
#
#   ap = voc_ap(rec, prec, [use_07_metric])
#   Compute VOC AP given precision and recall.
#   If use_07_metric is true, uses the
#   VOC 07 11 point method (default:False).
#   """
#   if use_07_metric:
#     # 11 point metric
#     ap = 0.
#     for t in np.arange(0., 1.1, 0.1):
#       if np.sum(rec >= t) == 0:
#         p = 0
#       else:
#         p = np.max(prec[rec >= t])
#       ap = ap + p / 11.
#   else:
#     # correct AP calculation
#     # first append sentinel values at the end
#     mrec = np.concatenate(([0.], rec, [1.]))
#     mpre = np.concatenate(([0.], prec, [0.]))
#
#     # compute the precision envelope
#     for i in range(mpre.size - 1, 0, -1):
#       mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
#
#     # to calculate area under PR curve, look for points
#     # where X axis (recall) changes value
#     i = np.where(mrec[1:] != mrec[:-1])[0]
#
#     # and sum (\Delta recall) * prec
#     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#   return ap



#把反数据矫正并保存


def rotating_img(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(os.path.join(img_path,file))
            out = img.transpose(Image.ROTATE_180)
            out=img.transpose(Image.ROTATE_180)
            out.save(os.path.join(save_path,file))




#把三分类数据变成二分类
def distribut_data(img_path,save_path):
    c=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(os.path.join(img_path,file))
        else:
            name=file.split('.')[0].split('_')[-1]
            img = Image.open(img_path + '/' + file)
            if name=='1':
                img.save(save_path+'/'+str(c)+'_1.jpg')
            else:
                img.save(save_path+'/'+str(c)+'_0.jpg')
            c+=1





# img_path='/Users/wywy/Desktop/train_cls'
# all_1=[]
# all_0=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='1':
#             all_1.append(file)
#         else:
#             all_0.append(file)
# random.shuffle(all_0)
# c=0
# for f in all_0:
#     c += 1
#     if c>len(all_1):
#
#         os.remove(img_path+'/'+f)
#
#         print(c)


# import skimage.data
# from skimage import io
# import selectivesearch
# from scipy import misc
# import cv2
#
# # img = skimage.data.astronaut()
# img=Image.open('/Users/wywy/Desktop/dog.jpg')
# img=np.array(img)
# # misc.imsave('/Users/wywy/Desktop/1.jpg',img)
#
#
#
# img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
# rect=regions[:15]
# # all_rect=[]
# for r in rect:
#     set=r['rect']
#     cv2.rectangle(img, (set[0],set[1]), (set[0]+set[2],set[1]+set[3]), (0,255,0), 1)
# misc.imsave ('/Users/wywy/Desktop/dog1.jpg',img)

def img_aug(img_path,save_path):
    count=0
    for i in range(1):
        all_file=[]

        for file in os.listdir(img_path):
            if file=='.DS_Store':
                os.remove(img_path+'/'+file)
            else:
                all_file.append(file)
        random.shuffle(all_file)
        for f in all_file:
            # if count<5000:

                # img=cv2.imread(img_path+'/'+f)
                img=Image.open(img_path+'/'+f)
                img=img.resize((64,64))
                img=np.array(img)
                img1=img
                aa=[0.68,0.65,0.67,0.7,0.69,0.66,0.64,0.63,0.61,0.62,0.6,0.8,0.5,0.53,0.51,0.52,0.55,0.58,0.54]
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        index=random.randint(0,len(aa)-1)
                        img1[i, j] =img1[i,j][0]*aa[index],img1[i,j][1]*aa[index],img1[i,j][2]*aa[index]
                scipy.misc.imsave(save_path+'/m'+str(count)+f, img1)
                # print(save_path+'/aug'+f)

                count+=1
    print(count)

# img_path='/Users/wywy/Desktop/data-collection/outputs/3'
# save_path='/Users/wywy/Desktop/data-collection/outputs/33'
# img_aug(img_path,save_path)

#删除错误数据
# img_path='/Users/wywy/Desktop/删除数据'
# save_path='/Users/wywy/Desktop/all_m'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[1:]
#         if os.path.isfile(save_path+'/'+name[0]+'_'+name[1]+'.jpg'):
#             os.remove(save_path+'/'+name[0]+'_'+name[1]+'.jpg')
#             print(save_path+'/'+name[0]+'_'+name[1]+'.jpg')


# img_path='/Users/wywy/Desktop/all_choice'
# save_path='/Users/wywy/Desktop/未增强数据'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img_size=img.size
#         if img_size[0]==64 :
#             img.save(save_path+'/'+file)


#给截取出来的数据命名
# img_path='/Users/wywy/Desktop/2'
# save_path='/Users/wywy/Desktop/手机测试数据2'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img=img.convert('RGB')
#         img.save(save_path+'/'+str(c)+'_2.jpg')
#         c+=1

# img_path='/Users/wywy/Desktop/all_m'
# save_path='/Users/wywy/Desktop/2'
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='2':
#             img=Image.open(img_path+'/'+file)
#             img.save(save_path+'/'+file)

#图片亮度增强
# img_path='/Users/wywy/Desktop/all_m'
# save_path='/Users/wywy/Desktop/aug3'
# all_file=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         all_file.append(file)
# random.shuffle(all_file)
#
# c=0
# for f in all_file:
#     if c<15000:
#         name=f.split('.')[0].split('_')[-1]
#         if name=='3':
#             pass
#         else:
#             img=Image.open(img_path+'/'+f)
#             enh_bri = ImageEnhance.Brightness(img)
#             brightness = 1.5
#             image_brightened = enh_bri.enhance(brightness)
#             image_brightened.save(save_path+'/'+f)
#             c+=1
# print(c)

# img_path='/Users/wywy/Desktop/aug3'
# save_path='/Users/wywy/Desktop/all_m'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/save'+str(c)+'_'+name+'.jpg')
#         # print(save_path+'/save'+str(c)+'_'+name+'.jpg')
#         c+=1

# img_path='/Users/wywy/Desktop/手机测试数据'
# save_path='/Users/wywy/Desktop/手机识别数据3'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img=img.convert('RGB').resize((64,64))
#         img.save(save_path+'/'+str(c)+'_4.jpg')
#         c+=1
# img_path='/Users/wywy/Desktop/train_e1'
# save_path='/Users/wywy/Desktop/all_m'
# c=8952
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/d'+str(c)+'_'+name+'.jpg')
#         # print(save_path + '/d' + str(c) + '_' + name + '.jpg')
#         c+=1
# print(c)

# img_path='/Users/wywy/Desktop/增加数据'
# # save_path='/Users/wywy/Desktop/all_m'
# # c=524
# # for i in range(60):
# #     for file in os.listdir(img_path):
# #         if file=='.DS_Store':
# #             os.remove(img_path+'/'+file)
# #         else:
# #             img=Image.open(img_path+'/'+file)
# #             img=img.convert('RGB')
# #             img=img.resize((64,64))
# #             img.save(save_path+'/add'+str(c)+'_3.jpg')
# #             # print(save_path+'/add'+str(c)+'_3.jpg')
# #             c+=1
# # print(c)





# img_path='/Users/wywy/Desktop/all_m'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         c+=1
# print(c)

# img_path='/Users/wywy/Desktop/增加数据'
# save_path='/Users/wywy/Desktop/all_m2'
# c=0
# for i in range(60):
#     for file in os.listdir(img_path):
#         if file=='.DS_Store':
#             os.remove(img_path+'/'+file)
#         else:
#             img=Image.open(img_path+'/'+file)
#             img=img.convert('RGB').resize((64,64))
#             img.save(save_path+'/'+str(c)+'_3.jpg')
#             c+=1
#
# print(c)
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         c+=1
#         # if c>42000:
#         #     os.remove(img_path+'/'+file)
#
#
# print(c)

# img=Image.open('/Users/wywy/Desktop/11.jpg').convert('RGB')
# img=img.resize((500, 375))
# img.save('/Users/wywy/Desktop/22.jpg')

# img_path='/Users/wywy/Downloads/tf-faster-rcnn-master/data/demo'
# for file in os.listdir(img_path):
#     print(file)
    # if file=='.DS_Store':
    #     os.remove(img_path+'/'+file)

# img_path='/Users/wywy/Desktop/all_m'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='3':
#             os.remove(img_path+'/'+file)
#
# print(c)
#
# img_path='/Users/wywy/Desktop/图片文件/dog.jpg'
# img=Image.open(img_path)
# img=img.resize(())
# img_path='/Users/wywy/Desktop/all_m2'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='2':
#             c+=1
# print(c)

import math
def twist(img): #图像扭曲填坑
    src_img=img.copy() #将原始图像复制一个副本，这是关键。开始的时候忘记将原始像素复制出来一份，下边的代码里在原始像素里覆盖来覆盖去,总是没有满意的效果，我还以为公式写错了。
    cycle=2 #sin函数2*PI一个周期，这里设置要整几个周期
    y_amp=4 #y跟随正弦曲线变化的幅度,4就不少了。在本例中5像素以上数字会碎。
    height=64
    char_width=64
    for y in range(0,height):
        for x in range(0,char_width):
            new_y=y+round(math.sin(x*2*math.pi*cycle/char_width)*y_amp)
            #三角函数是高中学的,(x/char_width*2*pi)代表(x/char_width)个周期,这样就把像素坐标(x,y)的变化与正弦曲线扯上了关系.
            if new_y < height and new_y > 0:
                img.putpixel((x,new_y),src_img.getpixel((x,y)))
    return img
# img_path='/Users/wywy/Desktop/all_m'
# save_path='/Users/wywy/Desktop/扭曲数据'
# all_img=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         all_img.append(file)
# random.shuffle(all_img)
# c=0
# for f in all_img:
#     name=f.split('.')[0].split('_')[-1]
#     if name=='3':
#         pass
#     else:
#         c+=1
#         if c<3000:
#
#             img=twist(Image.open(img_path+'/'+f))
#             img.save(save_path+'/'+f)

#
# img_path='/Users/wywy/Desktop/all_mmm'
# save_path='/Users/wywy/Desktop/cls_test'
# c=1008
# for i in range(10):
#     for file in os.listdir(img_path):
#         if file=='.DS_Store':
#             os.remove(img_path+'/'+file)
#         else:
#             name=file.split('.')[0].split('_')[-1]
#             if name=='3':
#                 img = Image.open(img_path + '/' + file)
#                 img.save(save_path + '/' + str(c) + '_0.jpg')
#                 c += 1
#
#             else:
#                 pass
#
#
# print(c)

# img_path = '/Users/wywy/Desktop/xx/5'
# save_path = '/Users/wywy/Desktop/xx/55'
# c=0
# for file in os.listdir(img_path):
#     if file == '.DS_Store':
#         os.remove(img_path + '/' + file)
#     else:
#         img=Image.open(img_path+'/'+file).convert('RGB').resize((210, 32))
#         img.save(save_path+'/'+str(c)+'.jpg')
#         c+=1
# print(c)


# 二值判断,如果确认是噪声,用改点的上面一个点的灰度进行替换
# 该函数也可以改成RGB判断的,具体看需求如何


                    # 打开图片



#otsu算法

# img_path='/Users/wywy/Desktop/xx'
# # save_path=''
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=cv2.imread(img_path+'/'+file,0)
#         tret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         cv2.imshow('test',img)
# cv2.waitKey(0)






# img_path='/Users/wywy/Desktop/error'
# save_path='/Users/wywy/Desktop/e_cls/7'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         for f in os.listdir(img_path+'/'+file):
#             name=f.split('.')[0].split('_')[-1]
#             img = Image.open(img_path + '/' + file + '/' + f)
#             if img.size[0]>170:
#                 img.save(save_path+'/'+str(c)+'_'+name+'.jpg')
#                 c+=1
# print(c)

# img_path='/Users/wywy/Desktop/e_cls/5'
# save_path='/Users/wywy/Desktop/e_cls/5'
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file).resize((210,32))
#         img.save(save_path+'/'+file)



#生成数据---------------------------------------
# img_path='/Users/wywy/Desktop/e_cls/5'
# save_path='/Users/wywy/Desktop/data2'
# choice_num=5
# choice_set = [(0, 0, 42, 32), (42, 0, 84, 32), (84, 0, 126, 32), (126, 0, 168, 32)
#                     ,(168,0,210,32),(210,0,252,32),(252,0,294,32)]
#
# chioce_dict=dict(zip(list('ABCDEFG'),[0,1,2,3,4,5,6]))
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         gray_img=cv2.imread(img_path+'/'+file,0)
#         kernel = np.ones((3, 3), np.uint8)
#         erosion = cv2.erode(gray_img, kernel, iterations=1)
#         # gray_img=cv2.cvtColor(erosion,cv2.COLOR_BGR2GRAY)
#         # cv2.imshow('test',gray_img)
#         # cv2.waitKey(0)
#         name=file.split('.')[0].split('_')[-1]
#         if len(list(name))>1:
#             pass
#         else:
#             for ii in range(choice_num):
#                 crop_img = erosion[choice_set[ii][1]:choice_set[ii][3], choice_set[ii][0]:choice_set[ii][2]]
#                 tret2, crop_img1 = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 if name=='X' or ii !=int(chioce_dict.get(name)):
#                     cv2.imwrite(save_path+'/0/a'+str(c)+'_0.jpg',crop_img1)
#                 else:
#                     cv2.imwrite(save_path+'/1/a'+str(c)+'_1.jpg',crop_img1)
#                 c+=1


# img_path='/Users/wywy/Desktop/all_one/00'
# save_path='/Users/wywy/Desktop/00/11'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=cv2.imread(img_path+'/'+file)
#         count=0
#         flat=img.reshape([-1])
#         for ii in flat:
#             if ii==0:
#                 count+=1
#         if count>600:
#             img_=Image.open(img_path+'/'+file)
#             img_.save(save_path+'/'+file)

#
# img_path='/Users/wywy/Desktop/00/11'
# save_path='/Users/wywy/Desktop/all_one/00'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         os.remove(save_path+'/'+file)


# img_path='/Users/wywy/Desktop/00'
# save_path='/Users/wywy/Desktop/f_0'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=cv2.imread(img_path+'/'+file,0)
#         tret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         flat_img=img.reshape([-1])
#         count=0
#         for f in flat_img:
#             if f==0:
#                 count+=1
#         if count>250:
#             img=Image.open(img_path+'/'+file)
#             img.save(save_path+'/'+file)




# def brightness( im_file ):
#    im = Image.open(im_file)
#    stat = ImageStat.Stat(im)
#    r,g,b = stat.mean
#    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
#
# img_path='/Users/wywy/Desktop/mohu/7'
# save_path='/Users/wywy/Desktop/mohu/77'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file).convert('RGB').resize((294,32))
#         img.save(save_path+'/'+str(c)+'.jpg')
#         c+=1
# print(c)

# import requests
# from PIL import Image
# from io import BytesIO
# # #
# # # # 网络上图片的地址
# img_src ='http://static.lixueweb.com/android/student/homework/0c0c159c-2578-ed99-1fad-03b4e23cd529/15388886455400.jpg'
#
#
#
# response = requests.get(img_src)
# image = Image.open(BytesIO(response.content))
# image.save('/Users/wywy/Desktop/手机拍照数据/{}'.format(img_src.split('/')[-1]))


# img_path='/Users/wywy/Desktop/反光/7'
# save_path='/Users/wywy/Desktop/反光/all_1'
# c=912
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/c'+str(c)+'_1.jpg')
#         c+=1
# print(c)



# img_path='/Users/wywy/Desktop/all_train'
# # save_path=''
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='1':
#             c+=1
# print(c)

# img_path='/Users/wywy/Desktop/ok_data1'
# save_path='/Users/wywy/Desktop/all_train'
# c=129
# for i in range(10):
#     for file in os.listdir(img_path):
#         if file=='.DS_Store':
#             os.remove(img_path+'/'+file)
#         else:
#             name=file.split('.')[0].split('_')[-1]
#             if name=='1':
#                 c+=1
#                 if c<10000:
#                     img = Image.open(img_path + '/' + file)
#                     img.save(save_path + '/d' + str(c) + '_1.jpg')
# print(c)

import sys
def integral(img):
    '''
    计算图像的积分和平方积分
    :param img:Mat--- 输入待处理图像
    :return:integral_sum, integral_sqrt_sum：Mat--- 积分图和平方积分图
    '''
    integral_sum = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    integral_sqrt_sum = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)

    rows, cols = img.shape
    for r in range(rows):
        sum = 0
        sqrt_sum = 0
        for c in range(cols):
            sum += img[r][c]
            sqrt_sum += math.sqrt(img[r][c])

            if r == 0:
                integral_sum[r][c] = sum
                integral_sqrt_sum[r][c] = sqrt_sum
            else:
                integral_sum[r][c] = sum + integral_sum[r - 1][c]
                integral_sqrt_sum[r][c] = sqrt_sum + integral_sqrt_sum[r - 1][c]

    return integral_sum, integral_sqrt_sum


def sauvola(img, k=0.1, kernerl=(51, 51)):
    '''
    sauvola阈值法。
    根据当前像素点邻域内的灰度均值与标准方差来动态计算该像素点的阈值
    :param img:Mat--- 输入待处理图像
    :param k:float---修正参数,一般0<k<1
    :param kernerl:set---窗口大小
    :return:img:Mat---阈值处理后的图像
    '''
    if kernerl[0] % 2 != 1 or kernerl[1] % 2 != 1:
        raise ValueError('kernerl元组中的值必须为奇数,'
                         '请检查kernerl[0] or kernerl[1]是否为奇数!!!')

    # 计算积分图和积分平方和图
    integral_sum, integral_sqrt_sum = integral(img)
    # integral_sum, integral_sqrt_sum = cv2.integral2(img)
    # integral_sum=integral_sum[1:integral_sum.shape[0],1:integral_sum.shape[1]]
    # integral_sqrt_sum=integral_sqrt_sum[1:integral_sqrt_sum.shape[0],1:integral_sqrt_sum.shape[1]]

    # 创建图像
    rows, cols = img.shape
    diff = np.zeros((rows, cols), np.float32)
    sqrt_diff = np.zeros((rows, cols), np.float32)
    mean = np.zeros((rows, cols), np.float32)
    threshold = np.zeros((rows, cols), np.float32)
    std = np.zeros((rows, cols), np.float32)

    whalf = kernerl[0] >> 1  # 计算领域类半径的一半

    for row in range(rows):
        print('第{}行处理中...'.format(row))
        for col in range(cols):
            xmin = max(0, row - whalf)
            ymin = max(0, col - whalf)
            xmax = min(rows - 1, row + whalf)
            ymax = min(cols - 1, col + whalf)

            area = (xmax - xmin + 1) * (ymax - ymin + 1)
            if area <= 0:
                sys.exit(1)

            if xmin == 0 and ymin == 0:
                diff[row, col] = integral_sum[xmax, ymax]
                sqrt_diff[row, col] = integral_sqrt_sum[xmax, ymax]
            elif xmin > 0 and ymin == 0:
                diff[row, col] = integral_sum[xmax, ymax] - integral_sum[xmin - 1, ymax]
                sqrt_diff[row, col] = integral_sqrt_sum[xmax, ymax] - integral_sqrt_sum[xmin - 1, ymax]
            elif xmin == 0 and ymin > 0:
                diff[row, col] = integral_sum[xmax, ymax] - integral_sum[xmax, ymax - 1]
                sqrt_diff[row, col] = integral_sqrt_sum[xmax, ymax] - integral_sqrt_sum[xmax, ymax - 1]
            else:
                diagsum = integral_sum[xmax, ymax] + integral_sum[xmin - 1, ymin - 1]
                idiagsum = integral_sum[xmax, ymin - 1] + integral_sum[xmin - 1, ymax]
                diff[row, col] = diagsum - idiagsum

                sqdiagsum = integral_sqrt_sum[xmax, ymax] + integral_sqrt_sum[xmin - 1, ymin - 1]
                sqidiagsum = integral_sqrt_sum[xmax, ymin - 1] + integral_sqrt_sum[xmin - 1, ymax]
                sqrt_diff[row, col] = sqdiagsum - sqidiagsum

            mean[row, col] = diff[row, col] / area
            std[row, col] = math.sqrt((sqrt_diff[row, col] - math.sqrt(diff[row, col]) / area) / (area - 1))
            threshold[row, col] = mean[row, col] * (1 + k * ((std[row, col] / 128) - 1))

            if img[row, col] < threshold[row, col]:
                img[row, col] = 0
            else:
                img[row, col] = 255

    return img

#


# import matplotlib.pyplot as graph
# from numpy import fft
#
#
#
# # 仿真运动模糊
# def motion_process(image_size, motion_angle):
#     PSF = np.zeros(image_size)
#     print(image_size)
#     center_position = (image_size[0] - 1) / 2
#     print(center_position)
#
#     slope_tan = math.tan(motion_angle * math.pi / 180)
#     slope_cot = 1 / slope_tan
#     if slope_tan <= 1:
#         for i in range(15):
#             offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
#             PSF[int(center_position + offset), int(center_position - offset)] = 1
#         return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
#     else:
#         for i in range(15):
#             offset = round(i * slope_cot)
#             PSF[int(center_position - offset), int(center_position + offset)] = 1
#         return PSF / PSF.sum()
#
#
# # 对图片进行运动模糊
# def make_blurred(input, PSF, eps):
#     input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
#     PSF_fft = fft.fft2(PSF) + eps
#     blurred = fft.ifft2(input_fft * PSF_fft)
#     blurred = np.abs(fft.fftshift(blurred))
#     return blurred
#
#
# def inverse(input, PSF, eps):  # 逆滤波
#     input_fft = fft.fft2(input)
#     PSF_fft = fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
#     result = fft.ifft2(input_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
#     result = np.abs(fft.fftshift(result))
#     return result
#
#
# def wiener(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
#     input_fft = fft.fft2(input)
#     PSF_fft = fft.fft2(PSF) + eps
#     PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
#     result = fft.ifft2(input_fft * PSF_fft_1)
#     result = np.abs(fft.fftshift(result))
#     return result
#
# img_path='/Users/wywy/Desktop/识别错误/44'
# cc=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         image = cv2.imread(img_path+'/'+file)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         img_h = image.shape[0]
#         img_w = image.shape[1]
#         graph.figure(1)
#         graph.xlabel("Original Image")
#         graph.gray()
#         graph.imshow(image)  # 显示原图像
#
#         graph.figure(2)
#         graph.gray()
#         # 进行运动模糊处理
#         PSF = motion_process((img_h, img_w), 60)
#         blurred = np.abs(make_blurred(image, PSF, 1e-3))
#
#         graph.subplot(231)
#         graph.xlabel("Motion blurred")
#         graph.imshow(blurred)
#
#         result = inverse(blurred, PSF, 1e-3)  # 逆滤波
#
#
#         graph.subplot(232)
#         graph.xlabel("inverse deblurred1")
#         graph.imshow(result)
#
#         result = wiener(blurred, PSF, 1e-3)  # 维纳滤波
#         graph.subplot(233)
#         graph.xlabel("wiener deblurred(k=0.01)")
#         graph.imshow(result)
#
#         blurred_noisy = blurred + 0.1 * blurred.std() * \
#                         np.random.standard_normal(blurred.shape)  # 添加噪声,standard_normal产生随机的函数
#
#         graph.subplot(234)
#         graph.xlabel("motion & noisy blurred")
#         graph.imshow(blurred_noisy)  # 显示添加噪声且运动模糊的图像
#
#         result = inverse(blurred_noisy, PSF, 0.1 + 1e-3)  # 对添加噪声的图像进行逆滤波
#         graph.subplot(235)
#         graph.xlabel("inverse deblurred2")
#         graph.imshow(result)
#
#         result = wiener(blurred_noisy, PSF, 0.1 + 1e-3)  # 对添加噪声的图像进行维纳滤波
#         graph.subplot(236)
#         graph.xlabel("wiener deblurred(k=0.01)1")
#         graph.imshow(result)
#
#         graph.show()
#

# im02 = Image.open("/Users/wywy/Desktop/识别错误/44/1815_1.jpg")
#
# im= im02.filter(ImageFilter.EDGE_ENHANCE_MORE)    #ImageFilter.SHARPEN,ImageFilter.EDGE_ENHANCE_MORE
# img=np.array(im)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # tret2, crop_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# choice_set = [(0, 0, 42, 32), (42, 0, 84, 32), (84, 0, 126, 32), (126, 0, 168, 32)
#         , (168, 0, 210, 32), (210, 0, 252, 32), (252, 0, 294, 32)]
#
# for i in range(4):
#     crop_img= img[choice_set[i][1]:choice_set[i][3], choice_set[i][0]:choice_set[i][2]]
#     tret2, crop_img1 = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     cv2.imshow('test{}'.format(i),crop_img1)
# cv2.waitKey(0)

# im.save('/Users/wywy/Desktop/1.jpg')