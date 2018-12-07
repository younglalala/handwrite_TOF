import os
import numpy as np
from PIL import Image
import  cv2
import  random
import imutils
import math
import tensorflow as tf


# count=0
# img_path='/Users/wywy/Desktop/test_33'
# save_path='/Users/wywy/Desktop/test_3'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=cv2.imread(img_path+'/'+file,0)
#
#         #形态学闭运算
#
#         #比运算的矩形大小是原始图像的w*0.3/0.4 ,h*0.3
#         k_size=int(img.shape[0]*0.3),int(img.shape[1]/4*0.3)
#
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
#
#         # 闭运算
#         closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#         #二值化
#         ret2,th = cv2.threshold(closed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
#         #寻找二值化之后选项的外接矩形，计算矩形大小过滤掉不符合条件的轮廓
#         cnts = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = cnts[0] if imutils.is_cv2() else cnts[1]
#         if len(cnts) > 0:
#             for c in cnts:
#                 rect = cv2.minAreaRect(c)   #寻找外接矩形，返回（（中心点x，y），（w，h），矩形角度）
#                 left_top_x,left_top_y,right_x,rigth_y=math.ceil(rect[0][0]-rect[1][0]/2),math.ceil(rect[0][1]-rect[1][1]/2),\
#                                                       math.ceil(rect[0][0] + rect[1][0] / 2), math.ceil(rect[0][1] +rect[1][1] / 2)
#                 w,h=rect[1][0],rect[1][1]
#                 #塞选条件为：宽度大于原始宽度的1/2高度小于原始图片的1/4的图片过滤掉
#                 if w>img.shape[1]/2 or h>=img.shape[0] or h<=img.shape[0]/4:
#                     pass
#                 else:
#
#                     img = cv2.rectangle(img, (left_top_x, left_top_y), (right_x, rigth_y), (0, 0, 255), 3)






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

    k_size = int(image.shape[0] * 0.2), int(image.shape[1] / choice_num * 0.2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)

    # 闭运算
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    #二值化
    ret2,th = cv2.threshold(closed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #寻找二值化之后选项的外接矩形，计算矩形大小过滤掉不符合条件的轮廓
    cnts = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    chioce_area =[(img.shape[1]/choice_num)*i for i in range(choice_num) ]
    choice_size=[]
    all_out1=[]
    if len(cnts) > 0:
        for c in cnts:
            rect = cv2.minAreaRect(c)   #寻找外接矩形，返回（（中心点x，y），（w，h），矩形角度）
            left_top_x,left_top_y,right_x,rigth_y=math.ceil(rect[0][0]-rect[1][0]/2),math.ceil(rect[0][1]-rect[1][1]/2),\
                                                  math.ceil(rect[0][0] + rect[1][0] / 2), math.ceil(rect[0][1] +rect[1][1] / 2)
            w,h=rect[1][0],rect[1][1]
            #塞选条件为：宽度大于原始宽度的1/2高度小于原始图片的1/4的图片过滤掉
            if w>image.shape[1]/choice_num*1.2 or h>=image.shape[0] or h<=image.shape[0]/4:
                pass
            else:
                choice_size.append([left_top_x,left_top_y,right_x,rigth_y])
                image= cv2.rectangle(image, (left_top_x,left_top_y), (right_x,rigth_y), (0, 255, 0), 1)

    for i in range(len(choice_size)):
        index=np.argmin(np.abs(np.array(chioce_area)-choice_size[i][0]))
        all_out1.append(index)
    all_out1.sort()
    all_out1=list(tuple(all_out1))
    h, w = int(image.shape[0]), int(image.shape[1] / choice_num)
    all_crop_gray_img=[]
    all_crop_closed_img=[]
    all_crop_thresh_img=[]
    all_qsarea=[]
    all_qs_gray_sum=[]
    all_qs_avg_gray=[]

    #计算整个图像的方差
    (mean, img_dev) = cv2.meanStdDev(closed)

    for num in range(choice_num):
        # 局部OTSU二值化
        crop_gray_img = image[0:h, w * num:w * (num + 1)]
        crop_closed_img=closed[0:h, w * num:w * (num + 1)]
        (_, img_dev1) = cv2.meanStdDev(crop_closed_img)
        print(img_dev1,'--------',num,'--')

        crop_thresh_img=th[0:h, w * num:w * (num + 1)]
        all_crop_gray_img.append(crop_gray_img)
        all_crop_closed_img.append(crop_closed_img)
        all_crop_thresh_img.append(crop_thresh_img)
        balck_index = np.where(crop_thresh_img.reshape((-1)) == 0)[0]
        #选项填涂块面积
        qsarea=len(balck_index)
        #选项平均灰度
        if int(qsarea)==0:
            qs_avg_gray=255

        else:
            qs_avg_gray=sum(crop_gray_img.reshape((-1))[balck_index])/qsarea
        all_qs_avg_gray.append(qs_avg_gray)

        #选项灰度和
        qs_gray_sum=sum(crop_closed_img.reshape((-1))[balck_index])
        all_qs_gray_sum.append(qs_gray_sum)
        all_qsarea.append(qsarea)

    if img_dev<=1:
        finally_out=np.array([])
    else:
        #特定区域的平均灰度值满足一定条件才能被认为是填涂了的。
        all_out=np.where(np.array(all_qs_avg_gray)<170)[0]

        # print(all_qs_gray_sum)
        # #根据区域面积进行塞选
        print(np.array(all_qsarea)[all_out]/max(all_qsarea))
        area= all_out[np.where(np.array(all_qsarea)[all_out]/max(all_qsarea)>0.6)[0]]
        finally_out=list(area)+list(all_out1)
        finally_out=list(set(finally_out))
        #对最后的输出进行面积塞选
        finally_out=list(np.array(finally_out)[np.where(np.array(all_qsarea)[np.array(finally_out)]/max(all_qsarea)>0.6)[0]])
        # print(finally_out_area)

    if len(finally_out)==0:
        finally_out=finally_out
    else:
        finally_out_gray=np.array(all_qs_avg_gray)[np.array(finally_out)]

        # print(np.mean(finally_out_gray*1.5),'---------max---',max(finally_out_gray))
        if int(np.mean(finally_out_gray))<90:
            gray_thresh=int(np.mean(finally_out_gray)*1.5)
        else:
            gray_thresh=int(np.mean(finally_out_gray)*1.2)
        # print(gray_thresh,file,np.mean(finally_out_gray),'---',finally_out_gray)

        out_gray_scale=finally_out_gray/min(finally_out_gray)


        gray_scale_index=np.where(out_gray_scale>1.8)[0]

        if len(finally_out_gray[gray_scale_index])!=0 and int(finally_out_gray[gray_scale_index][0])>gray_thresh:
            # print(np.array(finally_out)[gray_scale_index])

            for index in np.array(finally_out)[gray_scale_index]:
                finally_out.remove(index)
        finally_out_gray = np.array(all_qs_avg_gray)[np.array(finally_out)]
        gray_sub = finally_out_gray - min(finally_out_gray)
        # if int(np.mean(finally_out_gray))<80:
        #     gray_sub_thresh=60
        # else:
        #     gray_sub_thresh=50
        gray_sub_index=np.where(gray_sub>60)[0]
        if len(gray_sub_index)!=0:
            for i in np.array(finally_out)[gray_sub_index]:
                finally_out.remove(i)




    return finally_out,closed,th,all_qs_avg_gray








    # print(choice_size,'------',file)




img_path='/Users/wywy/Desktop/box_cls1/44/f'
save_path='/Users/wywy/Desktop/box_cls1/44'
chioce_dict = dict(zip([0, 1, 2, 3, 4, 5, 6], list('ABCDEFG')))
for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.remove(img_path+'/'+file)
    else:
        img=cv2.imread(img_path+'/'+file,0)
        all_out,image,th,all_qs_avg_gray=box_info(img,4)
        # box_info(img, 4)
        # for im in range(len(all_crop_thresh_img)):
        #     cv2.imwrite(save_path+'/'+str(im)+'_'+file,all_crop_thresh_img[im])

        # cv2.imwrite(save_path + '/ff/' + '_' + str(all_qs_avg_gray) + '_' + file, img)

        # name=''
        # if len(all_out)==0:
        #     name='X'
        # else:
        #     for n in all_out:
        #         name+=chioce_dict.get(n)
        # label=file.split('.')[0].split('_')[-1]
        # if label==name:
        #     cv2.imwrite(save_path+'/t/'+name+'_'+file,img)
        # else:
        #     cv2.imwrite(save_path + '/f/' + name +'_'+str(all_qs_avg_gray)+ '_' + file, img)
        # # cv2.imwrite(save_path + '/f2/' + name + '_' + str(all_qs_avg_gray) + '_' + file, image)

#
# img_path='/Users/wywy/Desktop/cls_box/3'
# save_path='/Users/wywy/Desktop/cls_box/33'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         if img.size[0]>210:
#             img.save(save_path+'/'+file)
#             c+=1
# print(c)





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
                    if picel_scale[batch_][index] > 0.4  and r_scale[batch_][index]<0.4:
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
    img_path='/Users/wywy/Desktop/box_cls/4'
    save_path='/Users/wywy/Desktop/f_ou'
    save_path2='/Users/wywy/Desktop/box_cls/44'
    box_num=4
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
# if __name__=='__main__':
#     main()








# img_path='/Users/wywy/Desktop/box_cls/4'
# save_path='/Users/wywy/Desktop/test_33'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#
#         img=Image.open(img_path+'/'+file)
#         bg=Image.new('RGB',(img.size[0]+5,img.size[1]+5),'white')
#         bg.paste(img,(3,3))
#         bg.save(save_path+'/'+file)





