import os
import scipy.misc
from PIL import Image

import numpy as np
import tensorflow as tf
import cv2 as cv
from  scipy import misc
import time
start = time.clock()
img_path='/Users/wywy/Desktop/time'
all_image=[]
for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.remove(img_path+'/'+file)
    else:
        img = Image.open(img_path + '/' + file)
        img=img.convert('RGB').resize((168,32))
        img=img.convert('L')
        img=np.array(img)/255-0.5
        img=img.reshape((32,168,1))
        all_image.append(img)

info_dict=dict(zip([0,1,2,3,4,5,6,7],list('ABCDEFGX')))
class Test2Sample:
    def __init__(self,img_path):
        self.all_img=[]
        self.all_label=[]
        for file in os.listdir(img_path):
            if file=='.DS_Store':
                os.remove(img_path+'/'+file)
            else:
                img=Image.open(img_path+'/'+file)
                name=file.split('.')[0].split('_')[1:]
                num_label=[info_dict.get(x) for x in name]
                num_label.sort()
                img=img.resize((168,32))
                img=img.convert('L')
                img=np.array(img)/255-0.5
                img=img.reshape((32,168,1))
                self.all_img.append(img)
                self.all_label.append(num_label)
    def get_batch(self,batch_size):
        self.batch_img=[]
        self.batch_label=[]
        for i in range(batch_size):
            index=np.random.randint(len(self.all_img))
            self.batch_img.append(self.all_img[index])
            self.batch_img.append(self.all_label[index])

        return np.array(self.all_img),np.array(self.all_label)

class ModelDetect:
    def __init__(self):
        pass

    def cls_modle(self, image):
        all_image=[]
        all_l=[]
        else_image=[]
        else_l=[]
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./cls_modle4.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                input_x = sess.graph.get_tensor_by_name("input:0")
                model_output = sess.graph.get_tensor_by_name("output:0")
                # dp=sess.graph.get_tensor_by_name("dp:0")

                output = sess.run(model_output, feed_dict={input_x:image})
                # for index in range(len(output)):
                #     if output[index]==0:
                #         all_image.append(image[index])
                #         all_l.append(label[index])
                #
                #     else:
                #         else_image.append(image[index])
                #         else_l.append(label[index])

        # return all_image,all_l,else_image,else_l
                return output

    def recognition_modle(self):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./train1.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                input_x = sess.graph.get_tensor_by_name("input:0")
                model_output = sess.graph.get_tensor_by_name("output:0")

                return input_x,model_output,sess

                # output = sess.run(model_output, feed_dict={input_x:image})
                # return output


m=ModelDetect()
input_x,model_output,sess=m.recognition_modle()
output = sess.run(model_output, feed_dict={input_x:np.array(all_image)})
print(output)
# img_path='/Users/wywy/Desktop/客观题分类数据/分类/all_test'
# # save_path='Users/wywy/Desktop/kkk/save'
# # all_image=[]
# # all_label=[]
# # for file in os.listdir(img_path):
# #     if file=='.DS_Store':
# #         os.remove(img_path+'/'+file)
# #     else:
# #         name=file.split('.')[0].split('_')[-1]
# #         num_label = [info_dict.get(x) for x in name]
# #         num_label.sort()
# #         img=Image.open(img_path+'/'+file)
# #         img=img.convert('RGB').resize((168,32))
# #         img=img.convert('L')
# #         img=np.array(img)/255-0.5
# #         img=img.reshape((32,168,1))
# #         all_image.append(img)
# #         all_label.append(num_label)
#
#
# all_image=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img = Image.open(img_path + '/' + file)
#         img=img.convert('RGB').resize((168,32))
#         img=img.convert('L')
#         img=np.array(img)/255-0.5
#         img=img.reshape((32,168,1))
#         all_image.append(img)
#
# m=ModelDetect()
# out=m.cls_modle(np.array(all_image))
# # print(out)
#
# # out=m.recognition_modle(np.array(all_image))
# elapsed = (time.clock() - start)
# print(elapsed)
#
# # out=m.recognition_modle(np.array(all_image))
#
#
#
