import os
import scipy.misc
from PIL import Image

import numpy as np
import tensorflow as tf
import cv2 as cv


class Test2Sample:
    def __init__(self,img_path):
        self.all_img=[]
        self.all_label=[]
        for file in os.listdir(img_path):
            if file=='.DS_Store':
                os.remove(img_path+'/'+file)
            else:
                img=Image.open(img_path+'/'+file)
                name=file.split('.')[0].split('_')[-1]
                self.all_label.append(int(name))
                img=img.resize((64,64))
                img=img.convert('L')
                img=np.array(img)/255-0.5
                img=img.reshape((64,64,1))
                self.all_img.append(img)
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

    def modle(self, image):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./m_cls.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                input_x = sess.graph.get_tensor_by_name("input:0")
                model_output = sess.graph.get_tensor_by_name("output:0")
                dp=sess.graph.get_tensor_by_name("dp:0")

                output = sess.run(model_output, feed_dict={input_x: image,dp:1.})
                return output

if __name__ == '__main__':
    img_path = '/Users/wywy/Desktop/cls_test'
    save_path = '/Users/wywy/Desktop/save11'
    data = Test2Sample(img_path)
    mobiel_img,mobile_datd = data.get_batch(1)
    net = ModelDetect()
    out = net.modle(mobiel_img)
    print(out)
    print(mobile_datd)

    all_acc=[]
    for ii in range(len(out)):
        if out[ii]==mobile_datd[ii]:
            all_acc.append(1)
        else:
            all_acc.append(0)
    train_acc = np.mean(np.array(all_acc))

    print(train_acc)

    # for i in range(len(mobiel_img)):
    #     scipy.misc.imsave(save_path + '/' + str(count) + '_' + str(out[i]) + '.jpg', mobiel_img.reshape((-1,64,64))[i])
    #     count += 1