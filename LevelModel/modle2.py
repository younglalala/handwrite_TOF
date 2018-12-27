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

        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 1, 10],dtype=tf.float32,stddev=tf.sqrt(1 / 10)))
        self.conv1_b = tf.Variable(tf.zeros([10]))

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 10, 16],dtype=tf.float32,stddev=tf.sqrt(1 / 16)))
        self.conv2_b=tf.Variable(tf.zeros([16]))


        self.conv_dev1_1w = tf.Variable(tf.random_normal([3, 1, 16, 16],dtype = tf.float32,stddev = tf.sqrt(1 / 16)))
        self.conv_dev1_1b = tf.Variable(tf.zeros([16]))

        self.conv_dev1_2w = tf.Variable(tf.random_normal([1, 3, 16, 16], dtype = tf.float32, stddev = tf.sqrt(1 / 16)))
        self.conv_dev1_2b = tf.Variable(tf.zeros([16]))

        self.conv_dev1_3w = tf.Variable(tf.random_normal([3, 1, 16, 32], dtype = tf.float32, stddev = tf.sqrt(1 / 32)))
        self.conv_dev1_3b = tf.Variable(tf.zeros([32]))

        self.conv_dev1_4w = tf.Variable(tf.random_normal([1, 3, 32, 32], dtype = tf.float32, stddev = tf.sqrt(1 / 32)))
        self.conv_dev1_4b = tf.Variable(tf.zeros([32]))

        self.conv_dev1_5w = tf.Variable(tf.random_normal([1, 1, 32, 64], dtype = tf.float32, stddev = tf.sqrt(1 / 64)))
        self.conv_dev1_5b = tf.Variable(tf.zeros([64]))


        self.conv_dev2_1w = tf.Variable(tf.random_normal([3, 1, 16, 16],dtype = tf.float32,stddev = tf.sqrt(1 / 16)))
        self.conv_dev2_1b = tf.Variable(tf.zeros([16]))

        self.conv_dev2_2w = tf.Variable(tf.random_normal([1, 3, 16, 16],dtype = tf.float32,stddev = tf.sqrt(1 / 16)))
        self.conv_dev2_2b = tf.Variable(tf.zeros([16]))

        self.conv_dev2_3w = tf.Variable(tf.random_normal([1, 1, 16, 64], dtype = tf.float32, stddev = tf.sqrt(1 / 64)))
        self.conv_dev2_3b = tf.Variable(tf.zeros([64]))


        self.conv_dev3_1w = tf.Variable(tf.random_normal([1, 1, 16, 64], dtype = tf.float32, stddev = tf.sqrt(1 / 64)))
        self.conv_dev3_1b = tf.Variable(tf.zeros([64]))

        #pool

        self.conv_dev4_1w = tf.Variable(tf.random_normal([1, 1, 16, 64],dtype = tf.float32, stddev = tf.sqrt(1 / 64)))
        self.conv_dev4_1b=tf.Variable(tf.zeros([64]))


        self.conv3_w=tf.Variable(tf.random_normal([3,3,256,64],dtype=tf.float32,stddev=tf.sqrt(1/64)))
        self.conv3_b=tf.Variable(tf.zeros([64]))

        self.conv4_w=tf.Variable(tf.random_normal([1,1,64,64],dtype=tf.float32,stddev=tf.sqrt(1/64)))
        self.conv4_b=tf.Variable(tf.zeros([64]))

        #全联接
        self.fc1_w=tf.Variable(tf.random_normal([2*2*64,128],dtype=tf.float32,stddev=tf.sqrt(1/128)))
        self.b1=tf.Variable(tf.zeros([128]))

        self.fc2_w = tf.Variable(tf.random_normal([128, 5], dtype=tf.float32, stddev=tf.sqrt(1 / 5)))
        self.b2 = tf.Variable(tf.zeros([5]))




    def forward(self):

        #前部分卷积网络

        self.conv1=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding='SAME')+self.conv1_b))

        self.pool1=tf.nn.avg_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')   #32,32,10

        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b))

        self.pool2 = tf.nn.avg_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')   #16,16,16

        #分支1卷积
        self.conv_dev1_1=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool2,self.conv_dev1_1w,strides=[1,1,1,1],padding='SAME')+self.conv_dev1_1b)   #16
        )

        self.conv_dev1_2=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv_dev1_1,self.conv_dev1_2w,strides=[1,1,1,1],padding='SAME')+self.conv_dev1_2b
        ))

        self.conv_dev1_3 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv_dev1_2,self.conv_dev1_3w,strides=[1,1,1,1],padding='SAME')+self.conv_dev1_3b
        ))

        self.conv_dev1_4 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv_dev1_3, self.conv_dev1_4w, strides=[1, 1, 1, 1], padding='SAME') + self.conv_dev1_4b
        ))

        self.conv_dev1_5 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv_dev1_4, self.conv_dev1_5w, strides=[1, 1, 1, 1], padding='SAME') + self.conv_dev1_5b   #16,16,64
        ))

        #分支2卷积

        self.conv_dev2_1 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool2, self.conv_dev2_1w, strides=[1, 1, 1, 1], padding='SAME') + self.conv_dev2_1b
        ))

        self.conv_dev2_2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv_dev2_1, self.conv_dev2_2w, strides=[1, 1, 1, 1], padding='SAME') + self.conv_dev2_2b
        ))

        self.conv_dev2_3 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv_dev2_2, self.conv_dev2_3w, strides=[1, 1, 1, 1], padding='SAME') + self.conv_dev2_3b
        ))                                                                                                              #16，16，64


        #分支3卷积

        self.conv_dev3_1 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool2, self.conv_dev3_1w, strides=[1, 1, 1, 1], padding='SAME') + self.conv_dev3_1b
        ))

        self.pool_dev=tf.nn.avg_pool(self.conv_dev3_1,ksize=[1,1,1,1],strides=[1,1,1,1],padding='SAME')   #16,16,64


        #分支4

        self.conv_dev4_1 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool2, self.conv_dev4_1w, strides=[1, 1, 1, 1], padding='SAME') + self.conv_dev4_1b   #16,16,64
        ))

        #把所有分支上的结果合并
        self.concat=tf.concat(axis=3,values=[self.conv_dev1_5,self.conv_dev2_3,self.pool_dev,self.conv_dev4_1])    #16,16,64
        #合并之后卷积得到feature


        self.conv3=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.concat,self.conv3_w,strides=[1,2,2,1],padding='SAME')+self.conv3_b
        ))

        self.pool3 = tf.nn.avg_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv4 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool3, self.conv4_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b
        ))

        self.pool4 = tf.nn.avg_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   #2，2，64
        print(self.pool4)

        self.flat=tf.reshape(self.pool4,[-1,2*2*64])


        self.fc1=tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.flat,self.fc1_w)+self.b1))
        # self.fc1=tf.nn.dropout(self.fc1,keep_prob=self.dp)
        self.out=tf.matmul(self.fc1,self.fc2_w)+self.b2
        self.out=tf.reshape(self.out,[-1,5])


    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,logits=self.out))
        self.opt=tf.train.AdamOptimizer(1e-5).minimize(self.loss)

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
        saver.restore(sess,'./save/level_modle12.dpk')
        for i in range(100000):
            # train_img,train_label,tarin_file=sess.run(train_data)
            # train_img1=train_img/255-0.5
            # train_label=one_hot(train_label.tolist(),5)
            #
            # _,train_loss,train_acc,train_out,train_l,conv1,pool1,conv2,pool2,conv3,pool3,fc1,out=sess.run(
            #     [net.opt,net.loss,net.acc,net.argamx_out,net.argamx_label,net.conv1,net.pool1,
            #                                                    net.conv2,net.pool2,net.conv3,net.pool3,net.fc1,net.out],
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

                with tf.gfile.GFile("./level_modle13.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())


                saver.save(sess,'./save/level_modle12.dpk')

                print('------------test iter :{},  test loss:{},  test acc:{}----------'.format(i,test_loss,test_acc))
        # plt.show()









