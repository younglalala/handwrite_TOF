from PIL import Image,ImageDraw,ImageFont
import Augmentor
import os
import random


import numpy as np
import cv2
import scipy.misc


#根据扫描数据截取图片

#未用到
def crop_img(img_path,save_path,chioce):
    w_num=3
    h_num=27
    start_y=10
    end_y=1000
    dict_info = dict(zip(list('123'), [[10, 50], [60, 100], [110, 150]]))

    start_x=dict_info.get(chioce)[0]
    end_x=dict_info.get(chioce)[1]


    count=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            crop_img_w = int((end_x - start_x) / w_num)
            crop_img_h = int((end_y - start_y) / h_num)
            crop_x=10
            crop_y=10
            for i in range(h_num):
                crop_y+=(int((end_x - start_x) / w_num))*i
                for j in range(w_num):
                    crop_x+=(int((end_y - start_y) / h_num))*j
                    crop_img=img.crop((crop_x,crop_y,crop_x+crop_img_w,crop_y+crop_img_h),Image.ANTIALIAS)
                    crop_img.save(save_path+'/'+str(count)+'_1.jpg')
                    count+=1
    print(count)



#数据整合

def all_data(img_path,save_path):
    c=20000
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            img.save(save_path+'/'+str(c)+'_2.jpg')
            c+=1

#测试集和训练集分配



def distribution_data(all_train_path,test_path):
    all_file=[]
    for file in os.listdir(all_train_path):
        if file == '.DS_Store':
            os.remove(all_train_path + '/' + file)
        else:
            all_file.append(file)
    random.shuffle(all_file)
    c=0
    for f in all_file:
        if c<5000:
            img = Image.open(all_train_path + '/' + f)
            img.save(test_path + '/'+f)
            os.remove(all_train_path+'/'+f)
            c += 1

    print(c)




#数据one-hot编码
def one_hot(data):
    all_data=[]
    for d in data:

        one_hott = [0,0]
        one_hott[int(d)]=1

        all_data.append(one_hott)

    return np.array(all_data)



#把图片周围黑色边框去掉
def crop_img1(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            c_img=img.crop((10,10,int(img.size[0]-10),int(img.size[1]-10)))
            c_img.save(save_path+'/'+file)


#图片降低色度
def img_aug(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(os.path.join(img_path,file))
        else:

            def img_aug(img_path, save_path):
                for file in os.listdir(img_path):
                    if file == '.DS_Store':
                        os.remove(img_path + '/' + file)
                    else:
                        img = cv2.imread(img_path + '/' + file)
                        img1 = img
                        # name=file.split('.')[0].split('-')
                        # name0=name[0]
                        # name1=list(name[1])[2]
                        # if name1=='X':
                        #     pass
                        for i in range(img.shape[0]):
                            for j in range(92):
                                img1[i, j] = 0.2 * img[i, j][0] + 0.25 * img[i, j][1] + 0.22 * img[i, j][2]  # 阈值自己调节
                        # print(save_path+'/''aug0'+name0+'_'+name1+'.jpg')
                        scipy.misc.imsave()


#图片增加椒盐噪点

def aug_img1(img_path,save_path):

    all_file=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            all_file.append(file)
    random.shuffle(all_file)
    c=10000
    for f in all_file:
        if c < 20000:
            img=cv2.imread(img_path+'/'+f)
            count=100
            for k in range(0,count):

                xi = int(np.random.uniform(0, img.shape[1]))
                xj = int(np.random.uniform(0, img.shape[0]))
                # add noise
                if img.ndim == 2:
                    img[xj, xi] = 255
                elif img.ndim == 3:
                    img[xj, xi, 0] = 255
                    img[xj, xi, 1] = 255
                    img[xj, xi, 2] = 255
            img=Image.fromarray(img)
            # draw1 = ImageDraw.Draw(img, mode="RGB")
            #
            # draw1.line((random.randint(0, img.size[0]), random.randint(0, img.size[1]), random.randint(0, img.size[0]), random.randint(0, img.size[1])),
            #            fill=(0,0,0))
            img.save(save_path+'/ss'+f)

            # scipy.misc.imsave(save_path+'/c'+f,img)
            c+=1
    print(c)
# img_path='/Users/wywy/Desktop/data-collection/outputs/33'
# save_path='/Users/wywy/Desktop/data-collection/outputs/aug3'
# # save_path='/Users/wywy/Desktop/data-collection/outputs/33'
# aug_img1(img_path,save_path)



# img_path='/Users/wywy/Desktop/删除数据'
# save_path='/Users/wywy/Desktop/all_choice'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[1:]
#         f=name[0]+'_'+name[1]+'.jpg'
#         if os.path.isfile(save_path+'/'+f):
#             os.remove(save_path+'/'+f)
#             print(f)


#识别错误数据数据增加
# #
# img_path='/Users/wywy/Desktop/train_e'
# save_path='/Users/wywy/Desktop/判断题数据/train_choice'
# c=6813
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name = file.split('.')[0].split('_')[1:]
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/add'+str(c)+'_'+name[1]+'.jpg')
#         # print(save_path+'/add'+str(c)+'_'+name[1]+'.jpg')
#         c+=1
# print(c)


####图像扭曲




# img_path='/Users/wywy/Desktop/判断题数据/train_choice'
# save_path='/Users/wywy/Desktop/判断题数据/2'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='2':
#             img=Image.open(img_path+'/'+file)
#             img.save(save_path+'/'+file)
#
# img_path='/Users/wywy/Desktop/判断题数据/2'
# save_path='/Users/wywy/Desktop/判断题数据/all_train'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)

# img_path='/Users/wywy/Desktop/删除数据2'
# save_path='/Users/wywy/Desktop/判断题数据/test_choice'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[1:]
#         if os.path.isfile(save_path+'/'+name[0]+'_'+name[1]+'.jpg'):
#             os.remove(save_path+'/'+name[0]+'_'+name[1]+'.jpg')
#             print(file)
#         # print(save_path+'/'+name[0]+'_'+name[1]+'.jpg')

# img_path='/Users/wywy/Desktop/data-collection/outputs/aug3'
# save_path='/Users/wywy/Desktop/all_m'
# c=10000
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+str(c)+'_3.jpg')
#         # print(save_path+'/'+str(c)+'_3.jpg')
#         c += 1





