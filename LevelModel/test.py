import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from LevelModel.train_sample import  *
from LevelModel.test_sample import *
from box_model.one_hott import one_hot,one_hot2,one_hot3
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

# def fully_connected(prev_layer, num_units, is_training):
#     """
#     num_units参数传递该层神经元的数量，根据prev_layer参数传入值作为该层输入创建全连接神经网络。
#
#    :param prev_layer: Tensor
#         该层神经元输入
#     :param num_units: int
#         该层神经元结点个数
#     :param is_training: bool or Tensor
#         表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息
#     :returns Tensor
#         一个新的全连接神经网络层
#
#     """
#     layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)
#     layer = tf.layers.batch_normalization(layer, training=is_training)
#     layer = tf.nn.relu(layer)
#     return layer
#
#
#
# def conv_layer(prev_layer, layer_depth, is_training):
#     """
#    使用给定的参数作为输入创建卷积层
#     :param prev_layer: Tensor
#         传入该层神经元作为输入
#     :param layer_depth: int
#         我们将根据网络中图层的深度设置特征图的步长和数量。
#         这不是实践CNN的好方法，但它可以帮助我们用很少的代码创建这个示例。
#     :param is_training: bool or Tensor
#         表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息
#     :returns Tensor
#         一个新的卷积层
#     """
#
#     strides = 2 if layer_depth%3 == 0 else 1
#     print(layer_depth*4)
#     #每层卷积深度（1-30）*4  一共20层
#     conv_layer = tf.layers.conv2d(prev_layer, layer_depth*4, 3, strides, 'same', use_bias=False, activation=None)
#     conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
#     conv_layer = tf.nn.relu(conv_layer)
#
#
#     return conv_layer

def create_variable(name, shape, initializer,
                    dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype,
                           initializer=initializer, trainable=trainable)


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


def _depthwise_separable_conv2d(inputs, num_filters, width_multiplier,
                                scope, downsample=False,istraining=True):
    """depthwise separable convolution 2D function"""
    num_filters = round(num_filters * width_multiplier)  # 输出通道数量  取整数部分

    strides = 2 if downsample else 1  # 下采样 确定卷积步长

    with tf.variable_scope(scope):
        dw_conv = depthwise_conv2d(inputs, "depthwise_conv", strides=strides)
        # bn = bacthnorm(dw_conv, "dw_bn", is_training=True)
        bn=tf.layers.batch_normalization(dw_conv,name="dw_bn",training=istraining)
        relu = tf.nn.relu(bn)
        pw_conv = conv2d(relu, "pointwise_conv", num_filters)
        # bn = bacthnorm(pw_conv, "pw_bn", is_training=self.is_training)
        bn=tf.layers.batch_normalization(pw_conv,name='pointwise_conv',training=istraining)
        return tf.nn.relu(bn)


def train(num_batches, batch_size, learning_rate,save_dir=None):
    # Build placeholders for the input samples and labels
    # 创建输入样本和标签的占位符
    width_multiplier=1
    inputs = tf.placeholder(tf.float32, [None, 64, 64, 1],name='input')
    labels = tf.placeholder(tf.float32, [None, 5])

    # Add placeholder to indicate whether or not we're training the model
    # 创建占位符表明当前是否正在训练模型
    is_training = tf.placeholder(tf.bool,name='is_training')

    # Feed the inputs into a series of 20 convolutional layers
    # 把输入数据填充到一系列20个卷积层的神经网络中
    layer = inputs
    # for layer_i in range(1, 20):
    #     layer = conv_layer(layer, layer_i, is_training)
    #
    # # Flatten the output from the convolutional layers
    # # 将卷积层输出扁平化处理
    # orig_shape = layer.get_shape().as_list()
    # layer = tf.reshape(layer, shape=[-1, orig_shape[1]*orig_shape[2]*orig_shape[3]])
    #
    # # Add one fully connected layer
    # # 添加一个具有100个神经元的全连接层
    # layer = fully_connected(layer, 100, is_training)
    #
    # # Create the output layer with 1 node for each
    # # 为每一个类别添加一个输出节点
    # logits = tf.layers.dense(layer, 5)

    # self.is_training = tf.placeholder(tf.bool, [])
    net = conv2d(layer, "conv_1", round(32 * width_multiplier), filter_size=3,
                 strides=2)  # ->
    net = tf.nn.relu(tf.layers.batch_normalization(net, name="conv_1/bn",training=is_training))  # NB+RELU
    net = _depthwise_separable_conv2d(net, 10, width_multiplier,
                                           "ds_conv_2",istraining=is_training)  # ->[N, 112, 112, 64]
    net = _depthwise_separable_conv2d(net, 16, width_multiplier,
                                           "ds_conv_3", downsample=True,istraining=is_training)  # ->[N, 56, 56, 128]   #32,32
    net = _depthwise_separable_conv2d(net, 16, width_multiplier,
                                           "ds_conv_4",istraining=is_training)  # ->[N, 56, 56, 128]
    net = _depthwise_separable_conv2d(net, 32, width_multiplier,
                                           "ds_conv_5", downsample=True,istraining=is_training)  # ->[N, 28, 28, 256]   #16,16
    net = _depthwise_separable_conv2d(net, 32, width_multiplier,
                                           "ds_conv_6",istraining=is_training)  # ->[N, 28, 28, 256]
    net = _depthwise_separable_conv2d(net, 64, width_multiplier,
                                           "ds_conv_7", downsample=True,istraining=is_training)  # ->[N, 14, 14, 512]   #8,8
    net = _depthwise_separable_conv2d(net, 64, width_multiplier,
                                           "ds_conv_8",istraining=is_training)  # ->[N, 14, 14, 64]
    net = _depthwise_separable_conv2d(net, 64, width_multiplier,
                                           "ds_conv_9",istraining=is_training)  # ->[N, 14, 14, 64】
    net = _depthwise_separable_conv2d(net, 64, width_multiplier,
                                           "ds_conv_10",istraining=is_training)
    net = _depthwise_separable_conv2d(net, 64, width_multiplier,
                                           "ds_conv_11",istraining=is_training)
    net = _depthwise_separable_conv2d(net, 64, width_multiplier,
                                           "ds_conv_12",istraining=is_training)  # ->[N, 14, 14, 64]
    net = _depthwise_separable_conv2d(net, 128, width_multiplier,
                                           "ds_conv_13", downsample=True,istraining=is_training)  # ->[N, 7, 7, 128]   #4,4
    net = _depthwise_separable_conv2d(net, 128, width_multiplier,
                                           "ds_conv_14",istraining=is_training)  # ->[N, 7, 7, 1024]
    net = avg_pool(net, 2, "avg_pool_15")  # ->[N, 1, 1, 128]
    net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")  # 去掉维度为1的维[N, 1, 1, 128] => [N,128]
    logits = fc(net, 5, "fc_16")  # -> [N, 5]
    predictions = tf.nn.softmax(logits)

    # Define loss and training operations
    # 定义loss 函数和训练操作
    model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels))

    # Tell TensorFlow to update the population statistics while training
    # 通知Tensorflow在训练时要更新均值和方差的分布
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

    # Create operations to test accuracy
    # 创建计算准确度的操作
    arg_out=tf.reshape(tf.argmax(predictions, 1),[-1,1],name='output')
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train and test the network
    # 训练并测试网络模型

    train_data = train_shuffle_batch(train_filename, [64, 64, 1], 128)
    test_data = test_shuffle_batch(test_filename, [64, 64, 1], 30)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        saver.restore(sess, './save/test_model.dpk')
        for batch_i in range(num_batches):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs, batch_ys, tarin_file = sess.run(train_data)
            batch_xs = batch_xs / 255 - 0.5
            # batch_xs=[cv2.resize(pic, (28, 28), interpolation=cv2.INTER_CUBIC).reshape([28,28,1]) for pic in batch_xs]

            batch_ys = one_hot(batch_ys.tolist(), 5)
            # train this batch
            # 训练样本批次
            sess.run(train_opt, {inputs: np.array(batch_xs), labels: batch_ys, is_training: True})

            # Periodically check the validation or training loss and accuracy
            # 定期检查训练或验证集上的loss和精确度
            if batch_i%100 == 0:
                test_img, test_label1, test_name = sess.run(test_data)

                test_label = one_hot(test_label1.tolist(), 5)

                test_img1 = test_img / 255 - 0.5
                for tt,test_image in enumerate(test_img1):


                    # test_image1 = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_CUBIC).reshape([-1,28,28,1])
                    test_image1=test_image.reshape([-1,64,64,1])

                    test_label11=test_label[tt].reshape([-1,5])

                    loss, acc ,test_out= sess.run([model_loss, accuracy,arg_out], {inputs: test_image1,
                                                          labels: test_label11,
                                                          is_training: False})
                    print(
                    'Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, loss, acc))
                    print('label:{},out:{}'.format(test_label1[tt],test_out))
            elif batch_i%25 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: batch_xs, labels: batch_ys, is_training: False})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss, acc))
            if save_dir and batch_i%1000==0:
                saver.save(sess,save_dir)
                print('modle save at ./save/test_model.dpk'
                      'save succuccy!!')
                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                            ['output'])

                with tf.gfile.GFile("./mobile222.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())






num_batches = 10000  # 迭代次数
batch_size = 128  # 批处理数量
learning_rate = 1e-4  # 学习率

tf.reset_default_graph()
with tf.Graph().as_default():
    train(num_batches, batch_size, learning_rate,save_dir='./save/test_model.dpk')