import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from LevelModel.train_sample import  *
from LevelModel.test_sample import *


from box_model.one_hott import one_hot,one_hot2,one_hot3


class Modle:
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,64,64,1],name='input')
        self.y_=tf.placeholder(dtype=tf.float32,shape=[None,5])
        # self.dp=tf.placeholder(dtype=tf.float32,name='dp')

        #主干分支权重

        self.conv11_w = tf.Variable(tf.random_normal([3, 1, 1, 16],dtype=tf.float32,stddev=tf.sqrt(1 / 16)))
        self.conv11_b = tf.Variable(tf.zeros([16]))

        self.conv12_w = tf.Variable(tf.random_normal([1, 3, 16, 16],dtype=tf.float32,stddev=tf.sqrt(1 / 16)))
        self.conv12_b = tf.Variable(tf.zeros([16]))

        self.conv21_w = tf.Variable(tf.random_normal([3, 1, 16, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv21_b = tf.Variable(tf.zeros([32]))

        self.conv22_w = tf.Variable(tf.random_normal([1, 3, 32, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv22_b = tf.Variable(tf.zeros([32]))

        self.conv3_w = tf.Variable(tf.random_normal([3, 3, 32, 64], dtype=tf.float32, stddev=tf.sqrt(1 / 64)))
        self.conv3_b = tf.Variable(tf.zeros([64]))

        self.conv4_w = tf.Variable(tf.random_normal([3, 3, 64, 64], dtype=tf.float32, stddev=tf.sqrt(1 / 64)))
        self.conv4_b = tf.Variable(tf.zeros([64]))

        self.conv5_w = tf.Variable(tf.random_normal([3, 3, 64, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.conv5_b = tf.Variable(tf.zeros([128]))

        #分支1
        #input shape:(64,64,1)
        self.dev_1_11w = tf.Variable(tf.random_normal([3, 1, 1, 32],dtype=tf.float32,stddev=tf.sqrt(1 / 32)))
        self.dev_1_11b = tf.Variable(tf.zeros([32]))

        self.dev_1_12w = tf.Variable(tf.random_normal([1, 3, 32, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.dev_1_12b = tf.Variable(tf.zeros([32]))

        self.dev_1_2w = tf.Variable(tf.random_normal([3, 3, 32, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.dev_1_2b = tf.Variable(tf.zeros([128]))

        #分支2
        #input shape:(16,16,32)
        self.dev_2_11w = tf.Variable(tf.random_normal([3, 1, 32, 128],dtype=tf.float32,stddev=tf.sqrt(1 / 128)))
        self.dev_2_11b = tf.Variable(tf.zeros([128]))

        self.dev_2_12w = tf.Variable(tf.random_normal([1, 3, 128,128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.dev_2_12b = tf.Variable(tf.zeros([128]))

        self.dev_2_2w = tf.Variable(tf.random_normal([3, 3, 128, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.dev_2_2b = tf.Variable(tf.zeros([128]))

        #分支3
        #input shape:(8,8,64)
        self.dev_3_11w = tf.Variable(tf.random_normal([3, 1, 64, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.dev_3_11b = tf.Variable(tf.zeros([128]))

        self.dev_3_12w = tf.Variable(tf.random_normal([1, 3, 128, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.dev_3_12b = tf.Variable(tf.zeros([128]))


        #降纬度
        #input shape:(2,2,384)
        self.out_conv1_w = tf.Variable(tf.random_normal([1,1,512,128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.out_conv1_b = tf.Variable(tf.zeros([128]))
        #output shape:(2,2,128)

        self.fc1_w = tf.Variable(tf.random_normal([2*2*128, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.fc1_b = tf.Variable(tf.zeros([128]))

        self.fc2_w = tf.Variable(tf.random_normal([128, 5], dtype=tf.float32, stddev=tf.sqrt(1 / 5)))
        self.fc2_b = tf.Variable(tf.zeros([5]))


    def forward(self):

        #主分支卷积
        #input shape:(64,64,1)
        self.conv11=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x,self.conv11_w,strides=[1,1,1,1],padding='SAME')+self.conv11_b))

        self.conv12 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv11, self.conv12_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv12_b))

        self.pool1=tf.nn.avg_pool(self.conv12,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')   #32,32,16

        self.conv21 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool1, self.conv21_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv21_b))

        self.conv22 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv21, self.conv22_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv22_b))

        self.pool2 = tf.nn.avg_pool(self.conv22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 16,16,32

        self.conv3 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool2, self.conv3_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b))

        self.pool3 = tf.nn.avg_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 8,8,64

        self.conv4 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool3, self.conv4_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b))

        self.pool4 = tf.nn.avg_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 4,4,64

        self.conv5 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool4, self.conv5_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b))

        self.pool5 = tf.nn.avg_pool(self.conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 2,2,64
        #output shape:(2,2,128)

        #分支1
        #input shape:(64,64,1)
        self.dev1_11 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x,self.dev_1_11w,strides=[1,2,2,1],padding='SAME')+self.dev_1_11b))

        self.dev1_12 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.dev1_11, self.dev_1_12w, strides=[1, 2, 2, 1], padding='SAME') + self.dev_1_11b))

        self.dev1_pool1=tf.nn.avg_pool(self.dev1_12,ksize=[1,4,4,1],strides=[1,4,4,1],padding='VALID')

        self.dev1_2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.dev1_pool1, self.dev_1_2w, strides=[1, 2, 2, 1], padding='SAME') + self.dev_1_2b))
        #output shape:(2,2,128)

        #分支2
        #input shape:(16,16,32)

        self.dev2_11 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool2,self.dev_2_11w,strides=[1,2,2,1],padding='SAME')+self.dev_2_11b))


        self.dev2_12 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.dev2_11, self.dev_2_12w, strides=[1, 2, 2, 1], padding='SAME') + self.dev_2_12b))


        self.dev2_2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.dev2_12, self.dev_2_2w, strides=[1, 2, 2, 1], padding='SAME') + self.dev_2_2b))
        #output shape:(2,2,128)


        #分支3
        #input shape:(8,8,64)
        self.dev3_11=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool3,self.dev_3_11w,strides=[1,2,2,1],padding='SAME')+self.dev_3_11b))

        self.dev3_12 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.dev3_11, self.dev_3_12w, strides=[1, 2, 2, 1], padding='SAME') + self.dev_3_12b))

        #output shape:(2,2,128)


        #输出连接
        self.concat=tf.concat([self.dev3_12,self.dev1_2,self.dev2_2,self.pool5],axis=3)
        #卷积降纬度
        self.out_conv=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.concat,self.out_conv1_w,strides=[1,1,1,1],padding='SAME')+self.out_conv1_b))

        self.flat=tf.reshape(self.out_conv,[-1,2*2*128])

        self.fc1=tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.flat,self.fc1_w)+self.fc1_b))
        # self.fc1=tf.nn.dropout(self.fc1,keep_prob=self.dp)
        self.out=tf.matmul(self.fc1,self.fc2_w)+self.fc2_b


    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,logits=self.out))
        self.opt=tf.train.AdamOptimizer(1e-4).minimize(self.loss)

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
        saver.restore(sess,'./save/new_3modle.dpk')
        for i in range(100000):
            # train_img,train_label,tarin_file=sess.run(train_data)
            # train_img1=train_img/255-0.5
            # train_label=one_hot(train_label.tolist(),5)
            #
            # _,train_loss,train_acc,train_out,train_l=sess.run(
            #     [net.opt,net.loss,net.acc,net.argamx_out,net.argamx_label],
            #          feed_dict={net.x:train_img1,net.y_:train_label,net.dp:0.5})
            # # plt.clf()
            # # plt.subplot(2,4,1)
            # # plt.imshow(conv1[0][:,:,0])
            # # plt.title('conv1')
            # #
            # # plt.subplot(2, 4, 2)
            # # plt.imshow(pool1[0][:, :, 0])
            # # plt.title('pool1')
            # #
            # # plt.subplot(2, 4, 3)
            # # plt.imshow(conv2[0][:, :, 0])
            # # plt.title('conv2')
            # #
            # # plt.subplot(2, 4, 4)
            # # plt.imshow(pool2[0][:, :, 0])
            # # plt.title('pool2')
            # #
            # # plt.subplot(2, 4, 5)
            # # plt.imshow(conv3[0][:, :, 0])
            # # plt.title('conv3')
            # #
            # # plt.subplot(2, 4, 6)
            # # plt.imshow(pool3[0][:, :, 0])
            # # plt.title('pool3')
            #
            #
            #
            # # plt.pause(0.1)
            #
            #
            #
            # tr_acc = []
            # for train_index in range(len(train_out)):
            #     if train_out[train_index] == train_l[train_index]:
            #         tr_acc.append(1)
            #     else:
            #         tr_acc.append(0)
            #         # misc.imsave('/Users/wywy/Desktop/train_e1'+'/'+str(chioce_dict.get(train_out[train_index]))+'_'+bytes.decode(tarin_file[train_index]),train_img.reshape([-1,64,64]) [train_index])
            #
            # train_acc = np.mean(np.array(tr_acc))
            #
            # print('train iter :{},  train loss:{},  train acc:{}'.format(i,train_loss,train_acc))
            # # # #

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

                with tf.gfile.GFile("./new_3modle.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())


                saver.save(sess,'./save/new_3modle.dpk')

                print('------------test iter :{},  test loss:{},  test acc:{}----------'.format(i,test_loss,test_acc))
        # plt.show()









