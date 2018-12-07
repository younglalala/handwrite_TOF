import tensorflow as tf
import numpy as np

import os
import cv2
from PIL import Image



def box_info(image,choice_num):
    """针对一个选项"""
    all_orig_crop_img_=[]
    for img in image:
        all_orig_crop_img=[]
        # image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        h,w=int(img.shape[0]),int(img.shape[1]/choice_num)
        for num in range(choice_num):
            orig_crop_image = img[0:h, w*num:w*(num+1)]
            orig_crop_image = cv2.resize(orig_crop_image, (42, 32), cv2.INTER_AREA)
            all_orig_crop_img.append(orig_crop_image.reshape([32,42,1]))

            # cv2.imwrite('/Users/wywy/Desktop/ff' + '/''1{}.jpg'.format(num), orig_crop_image)
        all_orig_crop_img_.append(all_orig_crop_img)

    return all_orig_crop_img_


def DL_parse(all_orig_crop_img):
    '''深度学习识别结果'''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = "./box_cls_model9.pb"

        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            input_x = sess.graph.get_tensor_by_name("input:0")
            cls_output = sess.graph.get_tensor_by_name("output:0")
            # dp = sess.graph.get_tensor_by_name("dp:0")
            all_dl_out=[]
            len_dl=[]
            for batch_ in range(len(all_orig_crop_img)):
                cls = sess.run(cls_output, feed_dict={input_x: np.array(all_orig_crop_img[batch_])/255-0.5})
                deeplearning_out = np.where(cls == 1)[0]
                if len(deeplearning_out)==0 or len(deeplearning_out)==1:
                    len_dl.append(0)
                else:
                    len_dl.append(len(deeplearning_out)-1)



                all_dl_out.append(deeplearning_out)

            return all_dl_out,len_dl

def recognition_box(image,len_dl):
    models=dict(zip([0,1,2,3,4,5],["../box_model/train1.pb","../box_model/r2.pb",
                                     "../box_model/r3.pb","../box_model/r4.pb",
                                     "../box_model/r5.pb","../box_model/r6.pb"]))



    all_image=[]
    all_out=[]
    all_label=[]
    for i in range(6):
        if i==0 or i ==1:
            with tf.Graph().as_default():
                output_graph_def = tf.GraphDef()
                output_graph_path = models.get(i)
                with open(output_graph_path, "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    tf.import_graph_def(output_graph_def, name="")
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    index = np.where(np.array(len_dl) == i)[0]
                    image_ = np.array(image)[index]

                    label = np.array(len_dl)[index]

                    input_x = sess.graph.get_tensor_by_name("input:0")
                    cls_output = sess.graph.get_tensor_by_name("output:0")
                    cls = sess.run(cls_output, feed_dict={input_x: np.array(image_) / 255 - 0.5})
                    all_image.append(image)
                    all_out.append(cls)
                    all_label.append(label)

        else:
            with tf.Graph().as_default():
                output_graph_def = tf.GraphDef()
                output_graph_path = models.get(i)
                with open(output_graph_path, "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    tf.import_graph_def(output_graph_def, name="")
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    index = np.where(np.array(len_dl) == i)[0]
                    image_ = np.array(image)[index]
                    label = np.array(len_dl)[index]

                    input_x = sess.graph.get_tensor_by_name("input:0")
                    cls_output = sess.graph.get_tensor_by_name("output:0")
                    dp = sess.graph.get_tensor_by_name("dp:0")
                    cls = sess.run(cls_output, feed_dict={input_x: np.array(image_) / 255 - 0.5,dp:1.})
                    all_image.append(image)
                    all_out.append(cls)
                    all_label.append(label)

    return all_image,all_out,all_label

def get_data(file_path):
    all_data=[]
    all_label=[]
    for file in os.listdir(file_path):
        if file.split('.')[-1]=='jpg':
            image=cv2.imread(os.path.join(file_path,file),0)
            image=image.reshape([image.shape[0],image.shape[1],1])
            all_data.append(image)
            label=int(file.split('.')[0].split('_')[-1])
            all_label.append(label)

    return all_data,all_label

def main():
    img_path='/Users/wywy/Desktop/test_data'
    save_path='/Users/wywy/Desktop/test_out'
    test_image,test_label=get_data(img_path)
    all_orig_crop_img_=box_info(test_image,7)
    all_dl_out, len_dl=DL_parse(all_orig_crop_img_)
    all_image, all_out, all_label=recognition_box(test_image,len_dl)
    c=0
    for index in range(len(all_image)):
        for i in range(len(all_image[index])):

            cv2.imwrite(save_path+'/'+str(c)+'_'+str(all_out[index][i])+'.jpg',all_image[index][i].reshape((32,168)))
            c+=1
    print(c)


# main()
# img_path='/Users/wywy/Desktop/4'
# save_path='/Users/wywy/Desktop/4'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file).convert('RGB').resize((168,32),Image.ANTIALIAS)
#         img.save(save_path+'/'+str(c)+'.jpg')
#         c+=1
# print(c)















# a=np.array([1,2,3,4,5,6])
# b=np.array([0,3,4])
# print(a[b])




