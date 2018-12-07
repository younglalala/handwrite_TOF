import tensorflow as tf
import numpy as np
import os
from PIL import Image
from scipy import misc


class Detect:
    def __init__(self):
        pass
    def clas_md(self,image):
        '''
        分类模型
        :param image: size为（168，32，1）的图片。
        :return: 输入的图片和图片对应的分类。
        '''
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./cls_modle.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                input_x = sess.graph.get_tensor_by_name("input:0")
                cls_output = sess.graph.get_tensor_by_name("output:0")

                cls=sess.run(cls_output,feed_dict= {input_x:image})

                return image,cls

    def cnn_md(self,cls_image,all_cls,cls):
        '''
        识别模型
        :param cls_image: 分类输出的图片
        :param all_cls: 分类输出的图片对应的类别
        :param cls:调用分类模型的编号（0：识别单选和未选客观题模型，1：2个选项模型，2：3个选项模型，3：4个选项模型，4：5个选项模型，5：6个选项模型）
        :return:对应分类模型识别出来的图片以及对应的识别结果。识别结果。
        '''
        clas_info=["./train1.pb","./r2.pb","./r3.pb","./r4.pb","./r5.pb",
                   "./r6.pb"]
        all_out=[]
        all_iamge=[]
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            if cls==0:
                cls01 = np.where(all_cls == 0)[0]
                cls01_image=cls_image[cls01]
                output_graph_path = clas_info[int(cls)]

                with open(output_graph_path, "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    tf.import_graph_def(output_graph_def, name="")
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    cls_input_x = sess.graph.get_tensor_by_name("input:0")
                    cls_output = sess.graph.get_tensor_by_name("output:0")
                    output=sess.run(cls_output,{cls_input_x:cls01_image})
                    all_out.append(output)
                    all_iamge.append(cls01_image)

            else:
                else_cls = np.where(all_cls == cls)[0]
                else_cls_iamge = cls_image[else_cls]
                output_graph_path = clas_info[int(cls)]

                with open(output_graph_path, "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    tf.import_graph_def(output_graph_def, name="")
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    cls_input_x = sess.graph.get_tensor_by_name("input:0")
                    cls_output = sess.graph.get_tensor_by_name("output:0")
                    dp=sess.graph.get_tensor_by_name("dp:0")
                    output_ = sess.run(cls_output, {cls_input_x: else_cls_iamge,dp:1.})
                    all_out.append(output_)
                    all_iamge.append(else_cls_iamge)


        return np.array(all_iamge),np.array(all_out)

def get_image(img_path):
    chioce_dict=dict(zip(list('ABCDEFGX'),[0,1,2,3,4,5,6,7]))
    all_label=[]
    all_image=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=file.split('.')[0].split('_')[1:]
            one_label=[]
            for n in name:
                one_label.append(chioce_dict.get(n))
            all_label.append(one_label)

            img=Image.open(img_path+'/'+file).resize((168,32)).convert('L')
            img=np.array(img).reshape([32,168,1])
            all_image.append(img)
    return np.array(all_image),np.array(all_label)


if __name__=='__main__':
    detect=Detect()
    img_path = '/Users/wywy/Desktop/c_test/test3'
    all_image,all_label = get_image(img_path)
    all_image=all_image/255-0.5
    image,cls=detect.clas_md(all_image)
    out_image,out=detect.cnn_md(image,cls,0)



    # out=out.reshape([-1])
    # all_label=all_label.reshape([-1])
    # acc=[]
    # for index in range(len(all_label)):
    #     if all_label[index]==out[index]:
    #         acc.append(1)
    #     else:
    #         acc.append(0)
    #         misc.imsave('/Users/wywy/Desktop/test_e/' + str(index) + '.jpg', image[index].reshape([32, 168]))
    # print(np.mean(np.array(acc)))









