import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

from scipy import misc

from model1.train_sample import *
from model1.test_sample import *
from tool.crop_img import one_hot

import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess



class TOrF:
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,64,64,1],name='input')
        self.y_=tf.placeholder(dtype=tf.float32,shape=[None,2])
        self.dp=tf.placeholder(dtype=tf.float32,name='dp')

    def forward(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            ,weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(self.x, 10, [3, 3], stride=2)
            net = slim.max_pool2d(net, [2, 2], padding='SAME')
            net = slim.conv2d(net, 16, [3, 3], stride=1)
            net = slim.max_pool2d(net, [2, 2], padding='SAME')
            net = slim.conv2d(net, 32, [3, 3], stride=1)
            net = slim.max_pool2d(net, [2, 2], padding='SAME')
            net = slim.conv2d(net, 64, [3, 3], stride=1)
            net = slim.max_pool2d(net, [2, 2], padding='SAME')
            # net = slim.conv2d(net, 64, [3, 3], stride=1)
            # net = slim.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.reshape(net, [-1, 2 * 2 * 64])

            print(net)
            net = slim.fully_connected(net, 256)
            net = slim.dropout(net, self.dp)
            print(net)

            self.predictions=slim.fully_connected(net,2,activation_fn=None)

    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictions,labels=self.y_))
        # self.optimizer = tf.train.AdamOptimizer(tf.train.exponential_decay(1e-3,100000,1000,0.96)).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y_, 1))
        self.rst = tf.cast(self.correct_prediction, "float")
        self.accuracy = tf.reduce_mean(self.rst)
        self.out_argmax = tf.argmax(self.predictions, 1)
        self.out_argmax1 = tf.reshape(self.out_argmax, [-1], name='output')


if __name__=='__main__':
    net=TOrF()
    net.forward()
    net.backward()

    train_data = train_shuffle_batch(train_filename, [64, 64, 1], 128)
    test_data = test_shuffle_batch(test_filename, [64, 64, 1], 2000)

    init=tf.global_variables_initializer()
    saver = tf.train.Saver()
    x=[]
    y=[]
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        # saver.restore(sess, './modle_save/m_train（all_m）.dpk')
        for ii in range(30000):
            train_img,train_label1,file_name=sess.run(train_data)
            train_label=one_hot(train_label1.tolist())
            train_img1=train_img/255-0.5

            out_argmax1, _, loss, out, accuracy = sess.run(
                [net.out_argmax1, net.optimizer, net.loss, net.predictions, net.accuracy],
                feed_dict={net.x: train_img1, net.y_: train_label,net.dp:0.5})
            tr_acc=[]
            for train_index in range(len(out_argmax1)):
                if out_argmax1[train_index]==train_label1[train_index]:
                    tr_acc.append(1)
                else:
                    tr_acc.append(0)
                    misc.imsave('/Users/wywy/Desktop/train_e'+'/'+str(out_argmax1[train_index])+'_'+bytes.decode(file_name[train_index]),train_img.reshape([-1,64,64]) [train_index])


            train_acc = np.mean(np.array(tr_acc))
            # x.append(ii)
            # y.append(loss)
            # plt.plot(x,y,'red')
            #
            # plt.pause(0.001)
            # plt.clf()


            print('第{}次的误差为{},精度{}'.format(ii,loss,train_acc))
            if ii%200==0:
                test_img, test_label1, test_name = sess.run(test_data)
                test_label = one_hot(test_label1.tolist())
                test_img1=test_img/255-0.5

                test_out_argmax1, test_loss, test_out, test_accuracy = sess.run(
                    [net.out_argmax1, net.loss, net.predictions, net.accuracy],
                    feed_dict={net.x: test_img1, net.y_: test_label,net.dp:1.})

                t_acc = []
                for test_index in range(len(test_out_argmax1)):
                    if test_out_argmax1[test_index] == test_label1[test_index]:
                        t_acc.append(1)
                    else:
                        t_acc.append(0)

                        misc.imsave('/Users/wywy/Desktop/test_e' + '/'+str(test_out_argmax1[test_index])+'_'+bytes.decode(test_name[test_index]) , test_img.reshape([-1,64,64]) [test_index])

                test_acc = np.mean(np.array(t_acc))


                print('测试集第{}次的误差为{},精度{}**********************'.format(ii,test_loss,test_acc))
                saver.save(sess,'./modle_save/m_cls.dpk')

                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                                ['output'])

                with tf.gfile.GFile("./m_cls.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())

                # frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                #                                                                ['output'])  # 这里 ['output']是输出tensor的名字
                # tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [net.x],[net.out_argmax1])  # 这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
                # open("mobile_choice1.tflite", "wb").write(tflite_model)






















