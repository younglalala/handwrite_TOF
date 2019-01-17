# -*- coding: utf-8 -*-
#
import tensorflow as tf
from scipy import misc
from LevelModel.train_sample import  *
from LevelModel.test_sample import *
from box_model.one_hott import one_hot,one_hot2,one_hot3

import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

from tensorflow.python.training import moving_averages

UPDATE_OPS_COLLECTION = "_update_ops_"



def create_variable(name, shape, initializer,
                    dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype,
                           initializer=initializer, trainable=trainable)



# BN1
# def bacthnorm(inputs, scope, epsilon=0.001, momentum=0.99, is_training=True):
#     inputs_shape = inputs.get_shape().as_list()  # 输出 形状尺寸
#     params_shape = inputs_shape[-1:]  # 输入参数的长度
#     axis = list(range(len(inputs_shape) - 1))
#
#     with tf.variable_scope(scope):
#         beta = create_variable("beta", params_shape,
#                                initializer=tf.zeros_initializer())
#         gamma = create_variable("gamma", params_shape,
#                                 initializer=tf.ones_initializer())
#         # 均值 常量 不需要训练 for inference
#         moving_mean = create_variable("moving_mean", params_shape,
#                                       initializer=tf.zeros_initializer(), trainable=False)
#         # 方差 常量 不需要训练
#         moving_variance = create_variable("moving_variance", params_shape,
#                                           initializer=tf.ones_initializer(), trainable=False)
#     if is_training:
#         mean, variance = tf.nn.moments(inputs, axes=axis)  # 计算均值和方差
#         # 移动平均求 均值和 方差  考虑上一次的量 xt = a * x_t-1 +(1-a)*x_now
#         update_move_mean = moving_averages.assign_moving_average(moving_mean,
#                                                                  mean, decay=momentum)
#         update_move_variance = moving_averages.assign_moving_average(moving_variance,
#                                                                      variance, decay=momentum)
#         tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_mean)
#         tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_variance)
#     else:
#         mean, variance = moving_mean, moving_variance
#     return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)



#BN2

def bn_layer(inputs, scope, epsilon=0.001, momentum=0.99, is_training=True):
    """
    Performs a batch normalization layer
    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay
    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope):
        shape = inputs.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, np.arange(len(shape) - 1), keep_dims=True)
            avg = tf.reshape(avg, [avg.shape.as_list()[-1]])
            var = tf.reshape(var, [var.shape.as_list()[-1]])
            # update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_avg = tf.assign(moving_avg, moving_avg * momentum + avg * (1 - momentum))
            # update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            update_moving_var = tf.assign(moving_var, moving_var * momentum + var * (1 - momentum))
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(inputs, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output








#############
# depthwise conv2d layer
def depthwise_conv2d(inputs, scope, filter_size=3, channel_multiplier=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()  # 输入通道 形状尺寸 64*64* 512
    in_channels = inputs_shape[-1]  # 输入通道数量 最后一个参数 512
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,
                                                  in_channels, channel_multiplier],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
    print(scope + '_w:', [filter_size, filter_size, in_channels, channel_multiplier])
    return tf.nn.depthwise_conv2d(inputs, filter, strides=[1, strides, strides, 1],
                                  padding="SAME", rate=[1, 1])


#################################################################
# 正常的卷积层 conv2d layer 输出通道    核大小
def conv2d(inputs, scope, num_filters, filter_size=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()  # 输入通道 形状尺寸 64*64* 512
    in_channels = inputs_shape[-1]  # 输入通道数量 最后一个参数 512
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,
                                                  in_channels, num_filters],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
        print(scope+'_w:',[filter_size, filter_size,in_channels, num_filters])
    return tf.nn.conv2d(inputs, filter, strides=[1, strides, strides, 1],
                        padding="SAME")


################################################################
# 均值池化层 avg pool layer
def avg_pool(inputs, pool_size, scope):
    print(scope+'_avgpool:',[1, pool_size, pool_size, 1],[1, pool_size, pool_size, 1])
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1],
                              strides=[1, pool_size, pool_size, 1], padding="VALID")


################################################################
# 全连接层 fully connected layer
def fc(inputs, n_out, scope, use_bias=True):
    inputs_shape = inputs.get_shape().as_list()  # 输入通道 形状尺寸 1*1* 512 输入时已经被展开了
    n_in = inputs_shape[-1]  # 输入通道数量 最后一个参数 512
    with tf.variable_scope(scope):
        weight = create_variable("weight", shape=[n_in, n_out],
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        if use_bias:  # 带偏置 与输出通道数量 同维度
            bias = create_variable("bias", shape=[n_out, ],
                                   initializer=tf.zeros_initializer())
            return tf.nn.xw_plus_b(inputs, weight, bias)  # 带偏置 相乘
        return tf.matmul(inputs, weight)  # 不带偏置 相乘



##### MobileNet模型结构定义 ####################################
class MobileNet:
    def __init__(self):
          # 输入数据
        self.num_classes = 5  # 类别数量
        # self.is_training = True  # 训练标志
        self.width_multiplier = 1  # 模型 输入输出通道数量 宽度乘数 因子

    def forward(self):
        # 定义模型结构 construct model
        with tf.variable_scope('MobileNet'):
            self.x = tf.placeholder(dtype=tf.float32,shape=[None,64,64,1],name='input')
            self.y_=tf.placeholder(dtype=tf.float32,shape=[None,5])
            # self.is_training = tf.placeholder(tf.bool, [])
            net = conv2d(self.x, "conv_1", round(32 * self.width_multiplier), filter_size=3,
                         strides=2)  # ->
            print("conv_1:",net)
            net = tf.nn.relu(tf.layers.batch_normalization(net, name="conv_1/bn"))  # NB+RELU
            print("conv_1/bn:", net)
            net = self._depthwise_separable_conv2d(net, 10, self.width_multiplier,
                                                   "ds_conv_2")  # ->[N, 112, 112, 64]
            print("ds_conv_2:", net)
            net = self._depthwise_separable_conv2d(net, 16, self.width_multiplier,
                                                   "ds_conv_3", downsample=True)  # ->[N, 56, 56, 128]   #32,32
            print("ds_conv_3:", net)
            net = self._depthwise_separable_conv2d(net, 16, self.width_multiplier,
                                                   "ds_conv_4")  # ->[N, 56, 56, 128]
            print("ds_conv_4:", net)
            net = self._depthwise_separable_conv2d(net, 32, self.width_multiplier,
                                                   "ds_conv_5", downsample=True)  # ->[N, 28, 28, 256]   #16,16
            print("ds_conv_5:", net)
            net = self._depthwise_separable_conv2d(net, 32, self.width_multiplier,
                                                   "ds_conv_6")  # ->[N, 28, 28, 256]
            print("ds_conv_6:", net)
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                                   "ds_conv_7", downsample=True)  # ->[N, 14, 14, 512]   #8,8
            print("ds_conv_7:", net)
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                                   "ds_conv_8")  # ->[N, 14, 14, 64]
            print("ds_conv_8:", net)
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                                   "ds_conv_9")  # ->[N, 14, 14, 64】
            print("ds_conv_9:", net)
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                                   "ds_conv_10")
            print("ds_conv_10:", net)
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                                   "ds_conv_11")
            print("ds_conv_11:", net)
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                                   "ds_conv_12")  # ->[N, 14, 14, 64]
            print("ds_conv_12:", net)
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                                   "ds_conv_13", downsample=True)  # ->[N, 7, 7, 128]   #4,4
            print("ds_conv_13:", net)
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                                   "ds_conv_14")  # ->[N, 7, 7, 1024]
            print("ds_conv_14:", net)
            net = avg_pool(net, 2, "avg_pool_15")  # ->[N, 1, 1, 128]
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")  # 去掉维度为1的维[N, 1, 1, 128] => [N,128]
            self.logits = fc(net, self.num_classes, "fc_16")  # -> [N, 5]
            self.predictions = tf.nn.softmax(self.logits)

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.predictions))
        # self.opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        self.opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

        self.argamx_label = tf.argmax(self.y_, axis=1)
        self.argamx_out = tf.reshape(tf.argmax(self.predictions, axis=1), [-1], name='output')
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.argamx_label, self.argamx_out), 'float'))


    #
    def _depthwise_separable_conv2d(self, inputs, num_filters, width_multiplier,
                                    scope, downsample=False):
        """depthwise separable convolution 2D function"""
        num_filters = round(num_filters * width_multiplier)  # 输出通道数量  取整数部分

        strides = 2 if downsample else 1  # 下采样 确定卷积步长

        with tf.variable_scope(scope):
            dw_conv = depthwise_conv2d(inputs, "depthwise_conv", strides=strides)
            # bn = bacthnorm(dw_conv, "dw_bn", is_training=self.is_training)
            bn=tf.layers.batch_normalization(dw_conv,name="dw_bn")
            relu = tf.nn.relu(bn)
            pw_conv = conv2d(relu, "pointwise_conv", num_filters)
            # bn = bacthnorm(pw_conv, "pw_bn", is_training=self.is_training)
            bn=tf.layers.batch_normalization(pw_conv,name='pointwise_conv')
            return tf.nn.relu(bn)


if __name__ == "__main__":

    net = MobileNet()
    net.forward()
    net.backward()
    train_data = train_shuffle_batch(train_filename, [64, 64, 1], 128)
    test_data = test_shuffle_batch(test_filename, [64, 64, 1], 1)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    x = []
    y = []
    chioce_dict = dict(zip([0, 1, 2, 3, 4], list('ABCDX')))
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        saver.restore(sess, './save/test_model.dpk')
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

            if i % 100== 0:
                test_img, test_label1, test_name = sess.run(test_data)

                test_label = one_hot(test_label1.tolist(), 5)
                test_img1 = test_img / 255 - 0.5

                test_loss, test_acc, test_out, test_l = sess.run(
                    [net.loss, net.acc, net.argamx_out, net.argamx_label],
                    feed_dict={net.x: test_img1, net.y_: test_label})

                tes_acc = []
                for test_index in range(len(test_out)):
                    if test_out[test_index] == test_l[test_index]:
                        tes_acc.append(1)
                    else:
                        tes_acc.append(0)
                        # misc.imsave('/Users/wywy/Desktop/test_e'+'/'+str(chioce_dict.get(test_out[test_index])) +'_'+bytes.decode(test_name[test_index]),test_img.reshape([-1,64,64])[test_index])

                test_acc = np.mean(np.array(tes_acc))

                # graph_def = tf.get_default_graph().as_graph_def()
                # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                #                                                                 ['output'])
                #
                # with tf.gfile.GFile("./mobile222.pb", 'wb') as f:
                #     f.write(output_graph_def.SerializeToString())


                #保存成tflite格式
                # frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                #                                                                ['output'])  # 这里 ['output']是输出tensor的名字
                # tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [net.x],
                #                                             [net.argamx_out])  # 这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
                # open("mobile_model222.tflite", "wb").write(tflite_model)
                #

                saver.save(sess, './save/test_model.dpk')
                #
                print('------------test iter :{},  test loss:{},  test acc:{}----------'.format(i, test_loss, test_acc))
        # plt.show()
