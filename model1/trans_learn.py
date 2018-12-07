import inspect
import os

import numpy as np
import tensorflow as tf
import time

from model1.train_sample import *
from model1.test_sample import *
from tool.crop_img import one_hot

VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg16:
    def __init__(self,vgg16_npy_path=None):
        """ 判断当前文件上一级目录下面是否有vgg16_npy_path，
            如果存在则加载VGG16与训练权重。
            """
        if vgg16_npy_path is None:
            path=inspect.getfile(Vgg16)   #返回Vgg16所在的文件夹的路径
            path = os.path.abspath(os.path.join(path, os.pardir))   #os.pardir文件上一级目录
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path=path

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

        print('npy file loaded')

        self.x=tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
        self.y_=tf.placeholder(dtype=tf.float32,shape=[None,3])
        self.dp=tf.placeholder(dtype=tf.float32)



    def avg_pool(self, bottom, name):   #平均池化

        """ bottom：输入
            name：变量名称
                    """
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):    #最大池化
        """ bottom：输入
            name：变量名称
                    """
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")   #获取卷积核的权重

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")   #获取偏值

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")   #获取全联接的权重


    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])  #把卷积输出的数组flat成二维数组【batch_size，H*W*C】

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def build(self):
        """ 加载VGG16的变量
                    """
        start_time = time.time()
        print("build model started")

        rgb_scaled = self.x * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.flatten = tf.reshape(self.pool5, [-1, 7 * 7 * 512])

        self.fc6 = tf.layers.dense(self.flatten, 128, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 3, name='out')
        print(("build model finished: %ds" % (time.time() - start_time)))

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out, labels=self.y_))
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y_, 1))
        self.rst = tf.cast(self.correct_prediction, "float")
        self.accuracy = tf.reduce_mean(self.rst)
        self.out_argmax = tf.argmax(self.out, 1)
        self.out_argmax1 = tf.reshape(self.out_argmax, [-1], name='output')


if __name__=='__main__':
    net = Vgg16(vgg16_npy_path='./vgg16.npy')
    net.build()
    net.backward()
    print('Net built')

    train_data = train_shuffle_batch(train_filename, [224, 224, 3], 100)
    test_data = test_shuffle_batch(test_filename, [224, 224, 3], 500)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        # saver.restore(sess, './modle_save/train.dpk')
        for ii in range(10000):
            train_img, train_label1, file_name = sess.run(train_data)
            train_label = one_hot(train_label1.tolist())
            _,train_loss,train_acc=sess.run([net.optimizer,net.loss,net.accuracy],feed_dict={net.x:train_img,net.y_:train_label,net.dp:0.5})
            print(train_loss,train_acc)



