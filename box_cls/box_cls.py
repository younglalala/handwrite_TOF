import tensorflow as tf

import os
import cv2
import numpy as np
import math


def box_info(image,choice_num):

    #闭运算的核的大小为单个选项的长宽的0.3倍
    k_size = int(image.shape[0] * 0.3), int(image.shape[1] / choice_num * 0.3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)

    # 闭运算
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite('/Users/wywy/Desktop/a/1.jpg',closed)
    # cv2.imwrite('/Users/wywy/Desktop/a/0.jpg', image)


    # 二值化
    _, threshold = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 寻找二值化之后选项的外接矩形，计算矩形大小过滤掉不符合条件的轮廓
    cnts = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    contours_set = []
    choice_area = [(image.shape[1] / choice_num) * (i) for i in range(choice_num+1)]
    if len(cnts) > 0:
        for c in cnts:
            rect = cv2.minAreaRect(c)  # 寻找外接矩形，返回（（中心点x，y），（w，h），矩形角度）
            left_top_x, left_top_y, right_x, rigth_y = math.ceil(rect[0][0] - rect[1][0] / 2), math.ceil(
                rect[0][1] - rect[1][1] / 2), math.ceil(rect[0][0] + rect[1][0] / 2),\
                math.ceil(rect[0][1] + rect[1][1] / 2)
            w, h = rect[1][0], rect[1][1]
            # 塞选条件为：宽度大于原始宽度的1/2高度小于原始图片的1/4的图片过滤掉
            if w > image.shape[1] / choice_num * 1.2 or h >= image.shape[0] or h <= image.shape[0] / 4:
                pass
            else:
                contours_set.append([left_top_x, left_top_y, right_x, rigth_y])

                image = cv2.rectangle(image, (left_top_x, left_top_y), (right_x, rigth_y), (0, 255, 0), 1)
    #根据轮廓检测得到的输出（轮廓检测的缺点是有些形状奇怪的涂了的选项不能被很好的检测到，这里用深度学习的方法对轮廓检测的输出做补充。）

    contours_out = []  # 轮廓检测后的输出
    for i in range(len(contours_set)):
        contour_center=((contours_set[i][2]-contours_set[i][0])/2)+contours_set[i][0]
        for c in range(len(choice_area)):
            if choice_area[c]<contour_center<choice_area[c+1]:
                contours_out.append(c)
        contours_out.sort()
        contours_out=list(tuple(contours_out))

    h,w=int(image.shape[0]),int(image.shape[1]/choice_num)

    all_crop_gray_img = []
    all_crop_closed_img = []
    all_crop_thresh_img = []

    all_qs_avg_gray=[]
    all_area=[]
    all_mean=[]
    all_dev=[]
    for num in range(choice_num):
        # 截取灰度图像
        crop_gray_img = image[0:h, w * num:w * (num + 1)]
        #截取闭运算之后的图像
        crop_closed_img=closed[0:h, w * num:w * (num + 1)]
        #截取二值化之后的图片
        crop_thresh_img = threshold[0:h, w * num:w * (num + 1)]
        all_crop_gray_img.append(cv2.resize(crop_gray_img,(42,32),interpolation=cv2.INTER_AREA).reshape((32,42,1)) )

        #计算单个选项的平均值和方差
        (mean_, img_dev) = cv2.meanStdDev(crop_closed_img)
        all_mean.append(mean_[0][0])
        all_dev.append(img_dev[0][0])
        all_crop_closed_img.append(crop_closed_img)
        all_crop_thresh_img.append(crop_thresh_img)
        balck_index = np.where(crop_thresh_img.reshape((-1)) == 0)[0]
        # 选项填涂块面积
        qsarea = len(balck_index)
        # 根据闭运算之后的图片计算选项平均灰度
        if int(qsarea) == 0:
            qs_avg_gray = 255
        else:
            qs_avg_gray = sum(crop_closed_img.reshape((-1))[balck_index]) / qsarea

        all_qs_avg_gray.append(qs_avg_gray)
        all_area.append(qsarea)


    return all_area,contours_out,all_crop_gray_img,all_mean,all_dev,image,all_qs_avg_gray



def DL_parse(all_orig_crop_img):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = "../mobile_model/box_cls_model9.pb"     #cls_model4.pb（对填涂较浅的数据也能识别）
        # box_cls_model9(不能识别填涂较浅数据和模糊数据),
        # box_cls_model3（对填涂很浅的数据也能识别，需要对识别结果进行过滤）

        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            input_x = sess.graph.get_tensor_by_name("input:0")
            cls_output = sess.graph.get_tensor_by_name("output:0")
            all_dl_out=[]
            for batch_ in range(len(all_orig_crop_img)):
                cls = sess.run(cls_output, feed_dict={input_x: np.array(all_orig_crop_img[batch_])/255-0.5})
                deeplearning_out = np.where(cls == 1)[0]
                all_dl_out.append(deeplearning_out)


            return all_dl_out


def box_parse(img_path,choice_num,blurry=False):

    test_image=[]
    all_orig_image=[]
    all_label=[]
    box_out=[]
    all_area=[]
    all_mean=[]
    all_dev=[]
    all_b=[]
    for file in os.listdir(img_path):
        if file == '.DS_Store':
            os.remove(img_path + '/' + file)
        else:
            img = cv2.imread(img_path + '/' + file, 0)
            imageVar = cv2.Laplacian(img, cv2.CV_64F).var()
            all_b.append(int(imageVar))

            label=file.split('.')[0].split('_')[-1]
            all_label.append(label)

            area, out, all_crop_gray_img,mean,dev,image,qs_avg_gray =box_info(img,choice_num)

            all_orig_image.append(image)
            all_area.append(area)
            all_mean.append(qs_avg_gray)
            all_dev.append(dev)

            box_out.append(out)
            test_image.append(all_crop_gray_img)
    all_dl_out=DL_parse(test_image)
    all_finally_out=[]
    for index in range(len(box_out)):
        #将得到的最终得到的轮廓输出和深度学习模型输出结果合并，去重
        coucat_out=list(box_out[index])+list(all_dl_out[index])
        coucat_out.sort()
        finally_out=list(set(coucat_out))
        #如果图像较模糊，设置blurry=True，查找每个图像是否有填涂，每个选项的方差大于10认为该图像为填涂
        if blurry:
            choice_dev=np.where(np.array(all_dev[index])>10)[0]
            choice_mean=np.where(np.array(all_mean[index])<250)[0]
            f_choice=[i for i in choice_mean if i in choice_dev]

            finally_out=[i for i in finally_out if i in f_choice]
        all_finally_out.append(finally_out)


    return all_finally_out,all_orig_image,all_label

if __name__=='__main__':
    img_path = '/Users/wywy/Desktop/box_cls/all_test'
    save_path='/Users/wywy/Desktop/box_cls/all_out'
    choice_num = 7
    choice_dict = dict(zip([0, 1, 2, 3, 4, 5, 6], list('ABCDEFG')))
    all_finally_out,all_orig_image,all_label=box_parse(img_path,choice_num)

    for index in range(len(all_finally_out)):
        name=''
        if len(all_finally_out[index])==0:
            name='X'
        else:
            for n in all_finally_out[index]:
                name+=choice_dict.get(n)
        if name==all_label[index]:
            cv2.imwrite(save_path+'/t/'+name+'_'+str(index)+'_'+all_label[index]+'.jpg',all_orig_image[index])
        else:
            cv2.imwrite(save_path + '/f/' + name + '_' + str(index) + '_' + all_label[index]+ '.jpg',
                        all_orig_image[index])






