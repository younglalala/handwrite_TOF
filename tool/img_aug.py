import cv2
import numpy as np
import os
import random
import Augmentor
from PIL import Image,ImageEnhance

import scipy.misc


#针对图片旋转部分：：

#方法1：
# # 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 0.75)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))

#
# img_path='/Users/wywy/Desktop/判断题数据/train_choice2'
# save_path='/Users/wywy/Desktop/判断题数据/aug'
#
# x=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         angle=np.random.randint(30)
#         img = cv2.imread(img_path + '/' + file)
#         index=random.sample([-1,1],1)
#         # print(file)
#         name=file.split('.')[0].split('_')[-1]
#         imgRotation = rotate_bound_white_bg(img, angle*int(index[0]))
#         h,w,c=np.shape(imgRotation)
#         # h_setoff,w_setoff=round( h/240,1),round( w/340,1)
#         # print(h_setoff,w_setoff)
#         # a=scipy.misc.imresize(imgRotation,(240,340))
#         # print('/Users/wywy/Desktop/rotate/{}_{}_{}_.jpg'.format(x,h_setoff,w_setoff))
#         scipy.misc.imsave('/Users/wywy/Desktop/判断题数据/aug/aug{}_{}.jpg'.format(x,name), imgRotation)
#         x+=1
# print(x)


#方法2：

def img_distortion(img_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            pass
    p=Augmentor.Pipeline(img_path)
    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    # p.zoom(probability=0.1, min_factor=0.5, max_factor=1.0)
    p.random_distortion(probability=1, grid_width=2, grid_height=2, magnitude=2)
    # p.flip_left_right(probability=0.2)
    # p.flip_top_bottom(probability=0.2)
    p.sample(2000)

# img_path='/Users/wywy/Desktop/判断题数据/1'
# img_distortion(img_path)


#图片尺寸缩放以及剪裁部份


#缩放尺寸
def img_zoom(img_path,save_path):
    scale_list = []
    for j in range(1000):
        scale = random.random()
        if scale < 0.5 or scale > 0.95:
            pass
        else:
            scale_list.append(scale)

    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:

            img=Image.open(img_path+'/'+file)
            bg = Image.new('RGB', img.size,'white')
            index=random.sample(scale_list,1)
            img=img.resize((int(img.size[0]*index[0]),int(img.size[1]*index[0])),Image.ANTIALIAS)
            paste_x=np.random.randint(0,img.size[0]-int(img.size[0]*index[0]))
            paste_y=np.random.randint(0,img.size[1]-int(img.size[1]*index[0]))
            bg.paste(img,(paste_x,paste_y))
            bg.save(save_path+'/aug1'+file)

# img_path='/Users/wywy/Desktop/判断题数据/train_choice2'
# save_path='/Users/wywy/Desktop/判断题数据/aug1'
# img_zoom(img_path,save_path)

# img_path='/Users/wywy/Desktop/判断题数据/aug1'
# save_path='/Users/wywy/Desktop/all_choice'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)


#针对手机拍照数据图片进行适当模糊处理
def img_GaussianBlur(img_path,save_path):
    kernel_size = (5, 5)
    sigma = 1.5

    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            enh_col = ImageEnhance.Color(img)
            color = 2
            image_colored = enh_col.enhance(color)
            img=cv2.GaussianBlur(np.array(image_colored), kernel_size, sigma)
            scipy.misc.imsave(save_path+'/g'+file,img)
# img_path='/Users/wywy/Desktop/判断题数据/mobile_train'
# save_path='/Users/wywy/Desktop/all_m'
# img_GaussianBlur(img_path,save_path)


#生成手机拍照数据并改变图片大小
def mobile_img(img_path,save_path):
    c=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            if img.size[0]==64:
                pass
            else:
                img=img.resize((64,64))
                img=np.array(img)
                # img = np.expand_dims(img, axis=2)
                # img = np.concatenate((img, img, img), axis=-1)
                # print(img.shape)
                if len(img.shape)==3:
                    img1 = img
                    aa = [0.68, 0.65, 0.67, 0.7, 0.69, 0.66, 0.64, 0.63, 0.61, 0.62, 0.8,0.6,0.5,0.55,0.57,0.56,0.53,0.52]
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            index=random.sample(aa,1)
                            img1[i, j] = img1[i, j][0] * index[0], img1[i, j][1] * index[0], img1[i, j][2] * index[0]
                    scipy.misc.imsave(save_path + '/' + file, img1)
                    # print(file)


img_path='/Users/wywy/Desktop/all_choice-1'
save_path='/Users/wywy/Desktop/all_m2'
mobile_img(img_path,save_path)


