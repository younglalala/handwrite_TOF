import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from LevelModel.train_sample import  *
from LevelModel.test_sample import *
import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

from box_model.one_hott import one_hot,one_hot2,one_hot3


class Modle:
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,64,64,1],name='input')
        self.y_=tf.placeholder(dtype=tf.float32,shape=[None,5])
        # self.dp=tf.placeholder(dtype=tf.float32,name='dp')

        #主干分支权重

        # self.conv1_w = tf.Variable(tf.random_normal([3, 3, 1, 16],dtype=tf.float32,stddev=tf.sqrt(1 / 16)))
        self.conv1_w = tf.get_variable(name='conv1_w',shape=[3, 3, 1, 16],initializer=tf.contrib.layers.xavier_initializer())
        self.conv1_b = tf.Variable(tf.zeros([16]))
        self.conv1d_w = tf.get_variable(name='conv1d_w',shape=[3, 3, 16, 1],initializer=tf.contrib.layers.xavier_initializer())
        self.conv1d_b = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.get_variable(name='conv2d_w',shape=[1, 1, 16, 32],initializer=tf.contrib.layers.xavier_initializer())
        self.conv2_b = tf.Variable(tf.zeros([32]))
        self.conv2d_w = tf.get_variable(name='conv2d_w',shape=[3, 3, 32, 1],initializer=tf.contrib.layers.xavier_initializer())
        self.conv2d_b = tf.Variable(tf.zeros([32]))

        self.conv3_w = tf.get_variable(name='conv3d_w',shape=[1, 1, 32, 32],initializer=tf.contrib.layers.xavier_initializer())
        self.conv3_b = tf.Variable(tf.zeros([32]))
        self.conv3d_w =tf.get_variable(name='conv3d_w',shape=[3, 3, 32, 1],initializer=tf.contrib.layers.xavier_initializer())
        self.conv3d_b = tf.Variable(tf.zeros([32]))

        self.conv4_w = tf.get_variable(name='conv4d_w',shape=[1, 1, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
        self.conv4_b = tf.Variable(tf.zeros([64]))
        self.conv4d_w =tf.get_variable(name='conv4d_w',shape=[3, 3, 64, 1],initializer=tf.contrib.layers.xavier_initializer())
        self.conv4d_b = tf.Variable(tf.zeros([64]))

        self.conv5_w = tf.get_variable(name='conv5d_w',shape=[1, 1, 64, 64],initializer=tf.contrib.layers.xavier_initializer())
        self.conv5_b = tf.Variable(tf.zeros([64]))
        self.conv5d_w = tf.get_variable(name='conv5d_w',shape=[3, 3, 64, 1],initializer=tf.contrib.layers.xavier_initializer())
        self.conv5d_b = tf.Variable(tf.zeros([64]))

        self.conv6_w = tf.get_variable(name='conv6d_w',shape=[1, 1, 64, 128],initializer=tf.contrib.layers.xavier_initializer())
        self.conv6_b = tf.Variable(tf.zeros([128]))
        self.conv6d_w = tf.get_variable(name='conv6d_w',shape=[3, 3, 128, 1],initializer=tf.contrib.layers.xavier_initializer())
        self.conv6d_b = tf.Variable(tf.zeros([128]))

        self.conv7_w = tf.get_variable(name='conv7d_w',shape=[1, 1, 128, 128],initializer=tf.contrib.layers.xavier_initializer())
        self.conv7_b = tf.Variable(tf.zeros([128]))
        self.conv7d_w = tf.get_variable(name='conv7d_w',shape=[3, 3, 128, 1],initializer=tf.contrib.layers.xavier_initializer())
        self.conv7d_b = tf.Variable(tf.zeros([128]))


        # self.fc_w=tf.Variable(tf.random_normal([2*2*128,128],dtype=tf.float32,stddev=tf.sqrt(1 / 128)))
        # self.fc_b=tf.Variable(tf.zeros([128]))

        self.out_w=tf.get_variable(name='out_w',shape=[2*2*128,5],initializer=tf.contrib.layers.xavier_initializer())
        self.out_b=tf.Variable(tf.zeros([5]))


    def forward(self):

        self.conv1=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding='SAME')+self.conv1_b))
        self.conv1d=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv1,self.conv1d_w,strides=[1,2,2,1],padding="SAME")+self.conv1d_b))           #32,32

        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv1d, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b))
        self.conv2d = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv2, self.conv2d_w, strides=[1, 1, 1, 1], padding="SAME") + self.conv2d_b))   #32,32

        self.conv3 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv2d, self.conv3_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b))
        self.conv3d = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv3, self.conv3d_w, strides=[1, 2, 2, 1], padding="SAME") + self.conv3d_b))   #16,16

        self.conv4 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv3d, self.conv4_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b))
        self.conv4d = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv4, self.conv4d_w, strides=[1, 2, 2, 1], padding="SAME") + self.conv4d_b))   #8,8

        self.conv5 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv4d, self.conv5_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b))
        # self.conv5=tf.nn.dropout(self.conv5,keep_prob=self.dp)
        self.conv5d = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv5, self.conv5d_w, strides=[1, 2, 2, 1], padding="SAME") + self.conv5d_b))   #4,4

        self.conv6=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv5d,self.conv6_w,strides=[1,1,1,1],padding='SAME')+self.conv6_b))
        self.conv6d=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv6, self.conv6d_w, strides=[1, 1, 1, 1], padding="SAME") + self.conv6d_b))   #4,4
        #
        # self.conv7 = tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.conv2d(self.conv6d, self.conv7_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv7_b))
        # self.conv7d = tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.depthwise_conv2d(self.conv7, self.conv7d_w, strides=[1, 1, 1, 1], padding="SAME") + self.conv7d_b))   #4,4

        self.avg_pool = tf.nn.avg_pool(self.conv6d,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        print(self.avg_pool)

        self.flat = tf.reshape(self.avg_pool,[-1,2*2*128])

        # self.fc = tf.nn.relu(tf.matmul(self.flat,self.fc_w)+self.fc_b)
        # self.fc = tf.nn.dropout(self.fc,keep_prob=self.dp)
        self.out = tf.matmul(self.flat,self.out_w)+self.out_b


    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,logits=self.out))
        self.opt=tf.train.AdamOptimizer(1e-3).minimize(self.loss)

        self.argamx_label=tf.argmax(self.y_,axis=1)
        self.argamx_out= tf.reshape(tf.argmax(self.out,axis=1),[-1],name='output')
        self.acc=tf.reduce_mean(tf.cast(tf.equal(self.argamx_label,self.argamx_out),'float'))



if __name__=='__main__':
    net=Modle()
    net.forward()
    net.backward()
    train_data = train_shuffle_batch(train_filename, [64, 64, 1], 128)
    test_data = test_shuffle_batch(test_filename, [64, 64, 1], 1000)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    x = []
    y = []
    chioce_dict=dict(zip([0,1,2,3,4],list('ABCDX')))
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        # saver.restore(sess,'./save/m_4.dpk')
        for i in range(100000):
            train_img,train_label,tarin_file=sess.run(train_data)
            train_img1=train_img/255-0.5
            train_label=one_hot(train_label.tolist(),5)

            _,train_loss,train_acc,train_out,train_l=sess.run(
                [net.opt,net.loss,net.acc,net.argamx_out,net.argamx_label],
                     feed_dict={net.x:train_img1,net.y_:train_label})



            tr_acc = []
            for train_index in range(len(train_out)):
                if train_out[train_index] == train_l[train_index]:
                    tr_acc.append(1)
                else:
                    tr_acc.append(0)
                    # misc.imsave('/Users/wywy/Desktop/train_e1'+'/'+str(chioce_dict.get(train_out[train_index]))+'_'+bytes.decode(tarin_file[train_index]),train_img.reshape([-1,64,64]) [train_index])

            train_acc = np.mean(np.array(tr_acc))

            print('train iter :{},  train loss:{},  train acc:{}'.format(i,train_loss,train_acc))
            # # #

            if i%100==0:
                test_img, test_label1, test_name = sess.run(test_data)


                test_label = one_hot(test_label1.tolist(),5)
                test_img1 = test_img / 255 - 0.5

                test_loss ,test_acc,test_out,test_l= sess.run(
                    [net.loss,net.acc,net.argamx_out,net.argamx_label],
                    feed_dict={net.x: test_img1, net.y_: test_label})

                tes_acc = []
                for test_index in range(len(test_out)):
                    if test_out[test_index] == test_l[test_index]:
                        tes_acc.append(1)
                    else:
                        tes_acc.append(0)
                        # misc.imsave('/Users/wywy/Desktop/test_e'+'/'+str(chioce_dict.get(test_out[test_index])) +'_'+bytes.decode(test_name[test_index]),test_img.reshape([-1,64,64])[test_index])

                test_acc = np.mean(np.array(tes_acc))

                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                                ['output'])

                with tf.gfile.GFile("./m_4.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())


                #保存成tflite
                # frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                #                                                                ['output'])  # 这里 ['output']是输出tensor的名字
                # tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [net.x],
                #                                             [net.argamx_out])  # 这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
                # open("level_mobile.tflite", "wb").write(tflite_model)


                saver.save(sess,'./save/m_4.dpk')

                print('------------test iter :{},  test loss:{},  test acc:{}----------'.format(i,test_loss,test_acc))
        # plt.show()









