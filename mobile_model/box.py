import tensorflow as tf

from PIL import Image
import random
import os
import cv2
import numpy as np


def get_data(file_path):
    all_data=[]
    all_label=[]
    for file in os.listdir(file_path):
        if file.split('.')[-1]=='jpg':
            image=cv2.imread(os.path.join(file_path,file))
            all_data.append(image)
            label=file.split('.')[0].split('_')[-1]
            all_label.append(label)

    return all_data,all_label


def box_info(image,choice_num):

    all_choice_picel = []
    all_crop_img = []
    all_orig_crop_img=[]
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h,w=int(image.shape[0]),int(image.shape[1]/choice_num)
    for num in range(choice_num):
        #局部OTSU二值化
        orig_crop_image = image_gray[0:h, w*num:w*(num+1)]
        orig_crop_image=cv2.resize(orig_crop_image,(42,32),cv2.INTER_AREA)
        _, crop_image = cv2.threshold(orig_crop_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #整体OTSU二值化
        crop_image2=threshold[0:h, w*num:w*(num+1)]
        crop_image2 = cv2.resize(crop_image2, (42, 32), cv2.INTER_AREA)
        #选取像素点最少的图片作为目标图片
        if np.mean(crop_image) > np.mean(crop_image2):
            ok_img = crop_image
        else:
            ok_img = crop_image2
        all_crop_img.append(ok_img.reshape([32, 42, 1]))
        all_orig_crop_img.append(orig_crop_image.reshape([32,42,1]))
        # cv2.imwrite('/Users/wywy/Desktop/ff' + '/''{}.jpg'.format(num), ok_img)
        # cv2.imwrite('/Users/wywy/Desktop/ff' + '/''1{}.jpg'.format(num), orig_crop_image)



        # 计算黑色像素点的数量
        pixels = ok_img.reshape([-1])
        pixel_num=len(np.where(pixels==0)[0])

        all_choice_picel.append(pixel_num)
    # 计算所得的每个选项的比值
    picel_scale = np.array(all_choice_picel) / max(all_choice_picel)
    #每个选项像素点与最大值的差值
    r_scale = 1 - np.array(picel_scale)

    return all_orig_crop_img,all_crop_img,picel_scale,r_scale


def DL_parse(all_orig_crop_img,picel_scale,r_scale):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = "./box_cls_model2.pb"     #cls_modle3.pb（手机识别数据二值化训练模型）
        # ,box_cls_modle,box_cls_modle3(模型压缩识别准确率较高模型),
        # box_cls_modle1（未压缩模型准确率较高模型）

        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            input_x = sess.graph.get_tensor_by_name("input:0")
            cls_output = sess.graph.get_tensor_by_name("output:0")
            # dp = sess.graph.get_tensor_by_name("dp:0")
            all_out=[]
            all_dl_out=[]
            for batch_ in range(len(all_orig_crop_img)):
                cls = sess.run(cls_output, feed_dict={input_x: np.array(all_orig_crop_img[batch_])/255-0.5})
                deeplearning_out = np.where(cls == 1)[0]
                all_dl_out.append(deeplearning_out)
                finally_out = []
                # #阈值自己设定
                for index in range(len(cls)):
                    if picel_scale[batch_][index] > 0.4 and cls[index]==1 and r_scale[batch_][index]<0.4:
                        finally_out.append(index)



                # if len(finally_out)>0:
                #     max_scale=np.where(picel_scale[batch_]==1)[0][0]
                #     if max_scale not in finally_out:
                #         finally_out.append(max_scale)

                all_out.append(finally_out)

            return all_dl_out,all_out


def box_parse(img_path,box_num):
    # chioce_dict = dict(zip([0, 1, 2, 3, 4, 5, 6], list('ABCDEFG')))
    test_data,test_label=get_data(img_path)
    all_orig_crop_img=[]
    all_crop_img=[]
    all_picel_scale=[]
    all_r_scale=[]
    c=0
    for d in test_data:
        orig_crop_img, crop_img, picel_scale, r_scale=box_info(d,box_num)
        all_orig_crop_img.append(orig_crop_img)
        all_crop_img.append(crop_img)
        all_picel_scale.append(picel_scale)
        all_r_scale.append(r_scale)
        c+=1

    deeplearning_out,finally_out=DL_parse(all_orig_crop_img,all_picel_scale,all_r_scale)

    return test_data,test_label,deeplearning_out,finally_out


def main():
    chioce_dict=dict(zip([0,1,2,3,4,5,6],list('ABCDEFG')))
    img_path='/Users/wywy/Desktop/crop_image/3'
    save_path='/Users/wywy/Desktop/f_ou'
    save_path2='/Users/wywy/Desktop/crop_image/33'
    box_num=3
    data,labels,deeplearning_out,finally_out=box_parse(img_path,box_num)

    for index in range(len(deeplearning_out)):
        name=''
        if len(deeplearning_out[index])==0:
            name='X'
        else:
            for out in deeplearning_out[index]:
                name+=chioce_dict.get(out)
        label=labels[index]
        out_lable=len(list(name))-1
        # dl_label=deeplearning_out[index]
        # if int(label)==out_lable:
        #     cv2.imwrite(save_path+'/'+str(index)+'_'+str(label)+'_'+str(deeplearning_out[index])+'.jpg',data[index])
        # else:
        #     cv2.imwrite(save_path2 + '/' + str(index) + '_' + str(label)+'_'+str(name) +'_'+str(dl_label)+ '.jpg', data[index])

        cv2.imwrite(save_path2 + '/' + str(index) + '_' + str(label)+ '_' + str(name) + '_' + str(deeplearning_out[index]) + '.jpg',
                    data[index])





#
if __name__=='__main__':
    main()











