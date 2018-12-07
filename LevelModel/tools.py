import cv2
import os
import numpy as np
import operator
import logging


import imutils
from imutils.perspective import  four_point_transform
from imutils import auto_canny, contours
from skimage import data, exposure
from PIL import Image,ImageStat
import Augmentor

from skimage import img_as_float
import matplotlib.pyplot as plt
from skimage import io
import math
import numpy.matlib
import random



#等级打分区域数据收集部分------------------------------


def pasre(img_path,save_path):
    '''
    截取并保存图片
    :param
    :return:
    '''
    cc = 0
    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)
    _, contours, _ = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_crop=image.copy()
    cv2.drawContours(img_crop, contours, -1, (0, 255, 0), 3)
    squares = []
    epsilon = 0.1
    min_area = 100*310
    max_area = 350*310
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon * cnt_len, True)
        if len(cnt) != 4:
            continue

        area = cv2.contourArea(cnt)

        if not min_area<area:
            continue
        squares.append(cnt.reshape(-1, 2))
    # if len(squares) !=35:
    #     print(len(squares))
    #     raise NotImplementedError   #函数不可用异常
    points = []

    for square in squares:
        #对坐标进行排序
        points.append([square[0],square[1], square[2],square[3]])
        x_set = np.array([square[0],square[1], square[2],square[3]])[:, 0]
        y_set = np.array([square[0],square[1], square[2],square[3]])[:, 1]
        np.array(x_set.sort())
        np.array(y_set.sort())
        x1, y1, x2, y2 = x_set[0], y_set[0], x_set[-1], y_set[-1]

        crop_img = image[y1:y2, x1:x2]
        cv2.imwrite(save_path+'/{}_{}.jpg'.format(img_path.split('/')[-1].split('.')[0],cc), crop_img)
        cc += 1
    print(len(points))


def discover(folder_path,save_path):
    for file in os.listdir(folder_path):
        if file=='.DS_Store':
            os.remove(folder_path+'/'+file)
        else:
            file_path=os.path.join(folder_path,file)
            pasre(file_path,save_path)


def main():
    img_path='/Users/wywy/Desktop/cc'
    save_path='/Users/wywy/Desktop/cc_'
    discover(img_path,save_path)


# main()




def remove_flaseimg(img_path,save_path,maxnum):
    '''
    塞选不符合条件的数据
    :param img_path:
    :param save_path:
    :param maxnum: 图片最大的高度，大于最大高度认为是falseimg
    :return:
    '''
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(os.path.join(img_path,file))
        else:
            img=Image.open(os.path.join(img_path,file))
            if img.size[1]>maxnum:
                img.save(os.path.join(save_path,file))
                os.remove(os.path.join(img_path,file))





#扫描数据数据增强部分----------------------------------


def rotate_bound_white_bg(image, angle):
    '''
    图像旋转背景以白色部分填充
    :param image:
    :param angle:  旋转角度
    :return:
    '''
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 0.75)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

# img_path='/Users/wywy/Desktop/level_choice/all_level'
# save_path='/Users/wywy/Desktop/level_choice/aug1'

# x=0
# all_file=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         all_file.append(file)
# random.shuffle(all_file)
# for f in all_file:
#     angle=np.random.randint(30)
#     img = cv2.imread(img_path+'/'+f)
#     index=random.sample([-1,1],1)
#     # name=file.split('.')[0].split('_')[-1]
#     imgRotation = rotate_bound_white_bg(img, angle*int(index[0]))
#     h,w,c=np.shape(imgRotation)
#     # h_setoff,w_setoff=round( h/240,1),round( w/340,1)
#     # print(h_setoff,w_setoff)
#     # a=scipy.misc.imresize(imgRotation,(240,340))
#     # print('/Users/wywy/Desktop/rotate/{}_{}_{}_.jpg'.format(x,h_setoff,w_setoff))
#     # cv2.imwrite(save_path+'aug{}_{}.jpg'.format(x,name), imgRotation)
#     cv2.imwrite(save_path+'/aug1_'+f,imgRotation)
#     x+=1
# print(x)


# img_path='/Users/wywy/Desktop/level_choice/X1'
# save_path='/Users/wywy/Desktop/level_choice/all_level'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(os.path.join(img_path,file))
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+str(c)+'_X.jpg')
#         c+=1
# print(c)



def crop_img(img_path,save_path,crop_size):
    '''
    截取黑色边框，获得无黑边的干净数据
    :param img_path:
    :param save_path:
    :param crop_size:截取区域开始位置 int
    :return:
    '''
    c=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            img_crop=img.crop((crop_size,crop_size,img.size[0]-crop_size,img.size[1]-crop_size))
            img_crop.save(save_path+'/'+str(c)+'.jpg')
            c+=1








def aug_distortion(img_path,gen_num):
    '''
    图像扭曲
    :param img_path:
    :param gen_num: 生成扭曲图像的数量
    :return:
    '''
    p = Augmentor.Pipeline(img_path)
    p.random_distortion(probability=1, grid_width=8, grid_height=8, magnitude=8)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.9)
    p.sample(gen_num)

# img_path='/Users/wywy/Desktop/level_choice/all_level'
# aug_distortion(img_path,5000)
# save_path='/Users/wywy/Desktop/level_choice/aug1'
# img_path='/Users/wywy/Desktop/output'
# save_path='/Users/wywy/Desktop/level_choice/aug2'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/aug2_'+str(c)+'_'+name+'.jpg')
#         c+=1
# print(c)




def rotate_img(img_path,save_path):
    '''
    图像翻转矫正
    :param img_path:
    :return:
    '''
    img=Image.open(img_path)
    img_rotate=img.transpose(Image.ROTATE_270)
    return img_rotate



def cut_img(img_path,area_num):
    '''
    图像随机裁剪
    :param img_path:
    :param save_path:
    :param area_num: 裁剪区域
    :return:
    '''
    crop_area=[i for i in range(area_num)]
    img=Image.open(img_path)
    w, h = img.size
    x1,y1,x2,y2=random.sample(crop_area,1)[0],random.sample(crop_area,1)[0],w-random.sample(crop_area,1)[0],h-random.sample(crop_area,1)[0]
    crop_img1=img.crop((x1,y1,x2,y2))
    return crop_img1
            # crop_img1.save(os.path.join(save_path,file))

# img_path='/Users/wywy/Desktop/level_choice/all_level'
# save_path='/Users/wywy/Desktop/level_choice/aug3'
# all_file=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         all_file.append(file)
# random.shuffle(all_file)
# for f in all_file:
#     crop_img1=cut_img(img_path+'/'+f,50)
#     crop_img1.save(save_path+'/aug3_'+f)








def salt(img, n):
    '''
    增加噪点
    :param img: numpy array 数组
    :param n: 噪点数量
    :return:
    '''
    for k in range(n):
        # 随机选择椒盐的坐标
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        # 如果是灰度图
        if img.ndim == 2:
            img[j,i] = 255
        # 如果是RBG图片
        elif img.ndim == 3:
            img[j,i,0]= 0
            img[j,i,1]= 0
            img[j,i,2]= 0
    return img

# img_path='/Users/wywy/Desktop/level_choice/all_level'
# save_path='/Users/wywy/Desktop/level_choice/aug4'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=cv2.imread(img_path+'/'+file)
#         s_img=salt(img,3000)
#         cv2.imwrite(save_path+'/aug4_'+file,s_img)






def ImgWithWhiteStripes(img_path,bg_path,save_path,white_num):
    '''
    添加背景色空白条
    :param img_path: 背景图片路径
    :param bg_path: 白色条纹路径
    :param save_path: 保存路径
    :param white_num: 随机白色条纹数量
    :return:
    '''

    width_area=[i+80 for i in range(40)]
    height_area=[i+4 for i in range(7)]

    #背景空白条
    all_bg=[]
    for file in os.listdir(bg_path):
        if file=='.DS_Store':
            os.remove(os.path.join(bg_path,file))
        else:
            img=Image.open(os.path.join(bg_path,file))
            bg_w,bg_h=random.sample(width_area,1)[0],random.sample(height_area,1)[0]
            bg_img=img.resize((bg_w,bg_h))
            all_bg.append(bg_img)


    #图片
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(os.path.join(img_path,file))
        else:
            img1=Image.open(os.path.join(img_path,file))
            for i in range(white_num):
                paste_x1,paste_y1=random.randint(0,img1.size[0]),random.randint(0,img1.size[1])
                img1.paste(random.sample(all_bg,1)[0],(paste_x1,paste_y1))
            img1.save(save_path+'/aug5_'+file)

# img_path='/Users/wywy/Desktop/level_choice/all_level'
# bg_path='/Users/wywy/Desktop/bg'
# save_path='/Users/wywy/Desktop/level_choice/aug5'
# ImgWithWhiteStripes(img_path,bg_path,save_path,8)
#

# img_path='/Users/wywy/Desktop/level_choice/all_aug'
# save_path='/Users/wywy/Desktop/train_level'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)






#等级识别作业纸指定区域获取部分---------------------------------


def SetPasre(img_path):
    '''
    答题卡四点定位
    :param img_path: 单张图片的路径
    :return: 图片的四个坐标点，图片
    '''
    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)
    _, contours, _ = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    epsilon = 0.1
    min_area = 40*40    #阈值根据实际情况设定
    max_area = 70*70
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon * cnt_len, True)
        if len(cnt) != 4:
            continue

        area = cv2.contourArea(cnt)
        if not min_area<area<max_area:
            continue
        squares.append(cnt.reshape(-1, 2))
    points = []
    all_set=[]
    for square in squares:
        #对坐标进行排序
        points.append([square[0],square[1], square[2],square[3]])
        x_set = np.array([square[0],square[1], square[2],square[3]])[:, 0]
        y_set = np.array([square[0],square[1], square[2],square[3]])[:, 1]
        np.array(x_set.sort())
        np.array(y_set.sort())
        x1, y1, x2, y2 = x_set[0], y_set[0], x_set[-1], y_set[-1]
        one_area = image_gray[y1:y2, x1:x2]  # 获取出来符合条件的区域
        all_pixel = []
        for ii in range(one_area.shape[0]):
            for jj in range(one_area.shape[1]):
                if one_area[ii][jj] == 0:
                    all_pixel.append(0)
                else:
                    all_pixel.append(1)
        mean_ = np.mean(np.array(all_pixel))
        if mean_ > 0.8 and 0.8<(x2-x1)/(y2-y1)<1.2:  # 设定阈值 0.4保证识别出来的二维码区域不被包含
            all_set.append([x1, y1, x2, y2])
    return all_set,image



#矫正
def CorrectPaper(all_set,image):
    '''
    根据四点矫正答题卡
    :param all_set: 所有符合条件的坐标
    :param image: 原始图片
    :return:
    '''
    newimage = image.copy()
    all_center_point=[]
    for se in all_set:
        x1,y1,x2,y2=se
        all_center_point.append([x1+((x2-x1)/2),y1+((y2-y1)/2)])
    if len(all_center_point) == 4:
        warped = four_point_transform(newimage, np.array(all_center_point))
        return warped


def ResizePaper(warped,size):
    '''
    矫正之后图片resize
    :param warped: 矫正之后图片
    :param size: a tupe
    :return:
    '''
    # resize(1725,2500)
    res = cv2.resize(warped, size, interpolation=cv2.INTER_CUBIC)
    return res

def AreaPaser(res,min_area):
    '''
    矫正并且resize之后的图片再次解析
    :param image: 矫正之后的图片
    :param min_area: 待选区域最小面积
    :return: 所有候选区域坐标，原始图片
    '''
    cc = 0
    image = res
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # k_size = (5,5)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
    #
    # # 闭运算
    # image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)
    _, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.dilate(image_gray, kernel, iterations=1)
    _, contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    epsilon = 0.1
    min_area = min_area
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon * cnt_len, True)
        if len(cnt) != 4:
            continue

        area = cv2.contourArea(cnt)

        if not min_area<area:
            continue
        squares.append(cnt.reshape(-1, 2))

    points = []
    all_set=[]
    for square in squares:
        points.append([square[0],square[1], square[2],square[3]])
        x_set = np.array([square[0],square[1], square[2],square[3]])[:, 0]
        y_set = np.array([square[0],square[1], square[2],square[3]])[:, 1]
        np.array(x_set.sort())
        np.array(y_set.sort())
        x1, y1, x2, y2 = x_set[0], y_set[0], x_set[-1], y_set[-1]
        all_set.append([x1, y1, x2, y2])

    return all_set,image

def ScoringArea(all_set,image):
    '''
    截取打分区域
    :param all_set: 所有候选区域的坐标
    :param image:矫正后的图片
    :return:打分区域
    '''
    all_area=[]
    for sett in all_set:
        x1, y1, x2, y2 =sett
        w,h=x2-x1,y2-y1
        all_area.append(w*h)
    min_=min(all_area)
    index=np.where(np.array(all_area)==min_)[0]
    print(index,'------')
    scoring_set=all_set[int(index)]
    print(scoring_set,'------111')
    x1,y1,x2,y2=scoring_set
    scoring_area=image[y1:y2,x1:x2]
    return scoring_area


def ScoringArea1(all_set,image):
    all_area=[]
    for sett in all_set:
        x1, y1, x2, y2 = sett

        w, h = x2 - x1, y2 - y1
        if 0.8<w/h<1.2 and 200000<w*h<280000:
            all_area.append([x1, y1, x2, y2])
            print([x1, y1, x2, y2],'-----')

    if len(all_area)==0:
        pass
    else:
        pass
        # scoring_area = image[all_area[0][0]:all_area[0][2], all_area[0][1]:all_area[0][3]]



    return all_area




def AnswerArea(all_set,image):
    '''
    截取答题区域
    :param all_set: 所有候选区域的坐标
    :param image: 矫正后的图片
    :return: 答题区域
    '''
    all_area = []
    for sett in all_set:
        x1, y1, x2, y2 = sett
        w, h = x2 - x1, y2 - y1
        all_area.append(w * h)
    max_ = max(all_area)
    index = np.where(np.array(all_area) == max_)[0]
    anser_set = all_set[int(index)]
    x1, y1, x2, y2 = anser_set

    answer_area = image[y1:y2, x1:x2]
    return answer_area


def ExamNumberArea(all_set,image):
    '''
    截取准考证号区域
    :param all_set:  所有候选区域的坐标
    :param image: 矫正后的图片
    :return: 准考证号区域
    '''
    all_area = []
    for sett in all_set:
        x1, y1, x2, y2 = sett
        w, h = x2 - x1, y2 - y1
        all_area.append(w * h)
    all_area1 = all_area.copy()
    all_area.sort()
    index_=np.where(np.array(all_area1)==all_area[1])[0]
    ExamNumberSet = all_set[int(index_)]
    x1, y1, x2, y2 = ExamNumberSet
    num_area = image[y1:y2, x1:x2]
    return num_area




def level_main():
    cc=0
    img_path='/Users/wywy/Desktop/ROI_area'
    save_path='/Users/wywy/Desktop/level_out'
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(os.path.join(img_path,file))
        else:
            image=cv2.imread(img_path+'/'+file)
            # all_set, image=SetPasre(img_path+'/'+file)
            # warped=CorrectPaper(all_set, image)

            # res=ResizePaper(warped, (1725,2500))
            all_set, image=AreaPaser(image, 5000)
            for s in all_set:
                # cv2.rectangle(image,(s[0],s[1]),(s[2],s[3]),(255,0,0),3)
                crop=image[s[1]+100:s[3]-10,s[0]+10:s[2]-10]

            # scoring_area1=ScoringArea1(all_set, warped)
            # answer_area=AnswerArea(all_set, image)
            # number_area=ExamNumberArea(all_set, image)
                cv2.imwrite(save_path+'/'+file,crop)
            # cc+=1
# level_main()


# #
# level_set=[1100,320,1800,880]
# img_path='/Users/wywy/Desktop/level_data'
# save_path='/Users/wywy/Desktop/ROI_area'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(os.path.join(img_path,file))
#     else:
#         img=Image.open(img_path+'/'+file)
#         crop=img.crop(level_set)
#         crop.save(save_path+'/'+file)


#
# img_path='/Users/wywy/Desktop/level_out'
# save_path='/Users/wywy/Desktop/train_level'
# c=1780
# for file in os.listdir(img_path):
#     if file =='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         name=file.split('.')[0].split('_')[-1]
#         img.save(save_path+'/error'+str(c)+'_'+name+'.jpg')
#         c+=1
# print(c)

















