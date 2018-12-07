import tensorflow as tf
import cv2
from PIL import Image,ImageStat
import math
import os
import numpy as np
from skimage import  exposure


def clas_md(image):
    '''
    分类模型
    :param image: 任意大小的image。
    :return: 输入的图片和图片对应的分类。
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = "./cls_modle3.pb"

        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            input_x = sess.graph.get_tensor_by_name("input:0")
            cls_output = sess.graph.get_tensor_by_name("output:0")
            dp = sess.graph.get_tensor_by_name("dp:0")

            cls=sess.run(cls_output,feed_dict= {input_x:image,dp:1.})

            return image,cls

def brightness( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


def mobile_chioce(img_path,save_path,choice_num,image_Var):

    '''
    :param img_path: 图片路径
    :param save_path: 图片保存路径
    :param choice_num: 选项个数
    :param image_Var: 图片模糊度，可设置（10，20。。。），值越大，过滤掉的模糊图片越多
    :return: 识别出来的图片，对应的out，不能识别的模糊图片
    '''

    choice_set = [(0, 0, 42, 32), (42, 0, 84, 32), (84, 0, 126, 32), (126, 0, 168, 32)
        , (168, 0, 210, 32), (210, 0, 252, 32), (252, 0, 294, 32)]
    all_output=[]
    all_image=[]
    unknown_img = []
    all_lable=[]
    for file in os.listdir(img_path):
        if file == '.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=cv2.imread(img_path + '/' + file)
            imageVar = cv2.Laplacian(img, cv2.CV_64F).var()   #拉普拉斯求图像的模糊程度
            normalizedImg = np.zeros((32, 168))
            normalizedImg = cv2.normalize(img, normalizedImg, 255.0, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            all_choice_picel = []
            all_out = []
            all_crop_img = []
            if imageVar<image_Var:
                unknown_img.append(img)
            else:
                if image_Var<imageVar < 100:
                    gam1 = exposure.adjust_log(normalizedImg)   #增加图像对比度和亮度（log）
                    # NpKernel = np.uint8(np.ones((3, 3)))
                    # erosion = cv2.dilate(gam1, NpKernel)
                    erosion_gray=cv2.cvtColor(gam1,cv2.COLOR_BGR2GRAY)
                    _, erosion = cv2.threshold(erosion_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    for ii in range(choice_num):
                        #对图像进行腐蚀
                        erosion_img = erosion_gray[choice_set[ii][1]:choice_set[ii][3], choice_set[ii][0]:choice_set[ii][2]]
                        _, erosion_cropimg1 = cv2.threshold(erosion_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        erosion_cropimg2=erosion[choice_set[ii][1]:choice_set[ii][3], choice_set[ii][0]:choice_set[ii][2]]
                        if np.mean(erosion_cropimg1) > np.mean(erosion_cropimg2):
                            ok_img = erosion_cropimg1
                        else:
                            ok_img = erosion_cropimg2

                        all_crop_img.append(ok_img.reshape([32, 42, 1]))
                        # cv2.imwrite(save_path+'/'+file+'_{}.jpg'.format(ii),ok_img)
                        pixels = ok_img.reshape([-1])
                        #计算黑色像素点的数量
                        pixel_num = 0
                        for pixel in pixels:
                            if pixel == 0:
                                pixel_num += 1
                        all_choice_picel.append(pixel_num)
                    # cv2.waitKey(0)
                else:
                    bright = brightness(img_path + '/' + file)     #计算图片的亮度值
                    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    bright_img2 = exposure.adjust_gamma(gray_img, 0.3)  #增加图片的亮度（gamma）
                    _1, img2 = cv2.threshold(bright_img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   #img2图像整体OTSU
                    for ii in range(choice_num):
                        crop_img=gray_img[choice_set[ii][1]:choice_set[ii][3],choice_set[ii][0]:choice_set[ii][2]]
                        crop_img2=img2[choice_set[ii][1]:choice_set[ii][3],choice_set[ii][0]:choice_set[ii][2]]
                        crop_img = exposure.adjust_gamma(crop_img, 0.3)
                        tret2, crop_img = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #crop_img图像局部OTSU
                        if bright>190:    #图片亮度比较大的时候采用局部阈值分割的结果
                            ok_img=crop_img
                        else:     #亮度较暗采用黑色像素点较少的图片
                            if np.mean(crop_img)>np.mean(crop_img2):
                                ok_img=crop_img
                            else:
                                ok_img =crop_img2
                        # cv2.imwrite(save_path + '/' + file + '_{}.jpg'.format(ii), ok_img)
                        all_crop_img.append(ok_img.reshape([32,42,1]))
                        pixels=ok_img.reshape([-1])
                        pixel_num=0
                        for pixel in pixels:
                            if pixel==0:
                                pixel_num += 1
                        all_choice_picel.append(pixel_num)


                picel_scale = np.array(all_choice_picel) / max(all_choice_picel)   #计算所得的每个选项的比值
                #塞选数据：
                image,cls=clas_md(np.array(all_crop_img)/255-0.5)     #得到深度学习分类结果
                deeplearning_out=np.where(cls==1)[0]
                for index in range(len(all_choice_picel)):
                    if picel_scale[index] > 0.55 and cls[index]==1:
                        all_out.append(index)
                if len(all_out) > 1:    #根据选项比值第二次塞选过滤
                    r_scale = 1 - np.array(picel_scale)
                    all_out=np.where(r_scale < 0.4)[0]

                all_output.append(all_out)
                all_image.append(img)
                label=file.split('.')[0].split('_')[1:]
                all_lable.append(label)



#保存数据

        #         if len(all_out) == 0:
        #             misc.imsave(save_path + '/' + str(cc) + '_X.jpg',img)
        #         else:
        #             name = ''
        #             for c in all_out:
        #                 name += '_' + chioce_dict.get(c)
        #             misc.imsave(save_path + '/' + str(cc) + str(name) + '.jpg', img)
        #         cc += 1
        # print(cc)
    return all_image,all_output,unknown_img,all_lable



img_path='/Users/wywy/Desktop/c_cls/4_a/ab3178_B.jpg'
# save_path=''
img=cv2.imread(img_path)    #BGR
Gray=img[:,:,2]*0.3+img[:,:,1]*0.59+img[:,:,0]*0.11
print(np.mean(Gray))  #
# _, erosion = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow('test',erosion)
# cv2.waitKey(0)
