import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from scipy import misc

from box_model.train_sample import  *
from box_model.test_sample import *

from box_model.one_hott import one_hot,one_hot2,one_hot3


class Modle:
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,32,168,1],name='input')
        self.y_=tf.placeholder(dtype=tf.float32,shape=[None,2])
        self.dp=tf.placeholder(dtype=tf.float32,name='dp')

        self.conv1_w=tf.Variable(tf.random_normal([3,3,1,10],dtype=tf.float32))
        self.conv1_b=tf.Variable(tf.zeros([10]))

        self.conv2_w=tf.Variable(tf.random_normal([3,3,10,16],dtype=tf.float32))
        self.conv2_b=tf.Variable(tf.zeros([16]))

        self.conv3_w=tf.Variable(tf.random_normal([3,3,16,32],dtype=tf.float32))
        self.conv3_b=tf.Variable(tf.zeros([32]))
        #
        # self.conv4_w=tf.Variable(tf.random_normal([3,3,32,64],dtype=tf.float32,stddev=tf.sqrt(1/64)))
        # self.conv4_b=tf.Variable(tf.zeros([64]))

        self.fc1_w=tf.Variable(tf.random_normal([1*5*32,64],dtype=tf.float32))
        self.fc1_b=tf.Variable(tf.zeros([64]))

        self.fc2_w=tf.Variable(tf.random_normal([64,7],dtype=tf.float32))
        self.fc2_b=tf.Variable(tf.zeros([7]))


    def forward(self):
        self.conv1=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x,self.conv1_w,strides=[1,2,2,1],padding='SAME')+self.conv1_b
        ))

        self.pool1=tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_b
        ))

        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        self.conv3 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool2, self.conv3_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b
        ))

        self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # self.conv4=tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.conv2d(self.pool3,self.conv4_w,strides=[1,2,2,1],padding='SAME')+self.conv4_b
        # ))

        # self.pool4=tf.nn.max_pool(self.conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        # # print(self.pool4)
        print(self.pool3)
        self.flat=tf.reshape(self.pool3,shape=[-1,1*5*32])

        self.fc1=tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.flat,self.fc1_w)+self.fc1_b))
        self.fc1=tf.nn.dropout(self.fc1,keep_prob=self.dp)
        self.out=tf.matmul(self.fc1,self.fc2_w)+self.fc2_b
        self.out=tf.reshape(self.out,[-1,7])


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
    train_data = train_shuffle_batch(train_filename, [32, 168, 1], 128)
    test_data = test_shuffle_batch(test_filename, [32, 168, 1], 10000)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    x = []
    y = []
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        saver.restore(sess,'./save/s_cls2.dpk')
        for i in range(100000):
            # train_img,train_label,tarin_file=sess.run(train_data)
            # train_img1=train_img/255-0.5
            # # print(train_img1)
            # train_label=one_hot(train_label.tolist(),7)
            #
            # _,train_loss,train_acc,train_out,train_l=sess.run([net.opt,net.loss,net.acc,net.argamx_out,net.argamx_label],
            #          feed_dict={net.x:train_img1,net.y_:train_label,net.dp:0.6})
            #
            # tr_acc = []
            # for train_index in range(len(train_out)):
            #     if train_out[train_index] == train_l[train_index]:
            #         tr_acc.append(1)
            #     else:
            #         tr_acc.append(0)
            #         # misc.imsave('/Users/wywy/Desktop/train_e'+'/'+str(train_out[train_index])+'_'+bytes.decode(tarin_file[train_index]),train_img.reshape([-1,32,168]) [train_index])
            #
            # train_acc = np.mean(np.array(tr_acc))
            #
            # print('train iter :{},  train loss:{},  train acc:{}'.format(i,train_loss,train_acc))

            if i%100==0:
                test_img, test_label1, test_name = sess.run(test_data)

                test_label = one_hot(test_label1.tolist(),7)
                test_img1 = test_img / 255 - 0.5

                test_loss ,test_acc,test_out,test_l= sess.run(
                    [net.loss,net.acc,net.argamx_out,net.argamx_label],
                    feed_dict={net.x: test_img1, net.y_: test_label,net.dp:1.})
                print(test_out)

                tes_acc = []
                for test_index in range(len(test_out)):
                    if test_out[test_index] == test_l[test_index]:
                        tes_acc.append(1)
                        # misc.imsave('/Users/wywy/Desktop/test_e' + '/' + str(test_index) + '_' + str(
                        #     test_out[test_index]) + '.jpg',
                        #             test_img.reshape([-1, 32, 168])[test_index])
                    else:
                        tes_acc.append(0)
                        # misc.imsave('/Users/wywy/Desktop/test_f'+'/'+str(test_out[test_index])+'_'+bytes.decode(test_name[test_index]),test_img.reshape([-1,32,168])[test_index])
                        # misc.imsave('/Users/wywy/Desktop/test_e' + '/' + str(test_index)+'_'+str(test_out[test_index])+'.jpg',
                        #             test_img.reshape([-1, 32, 168])[test_index])

                test_acc = np.mean(np.array(tes_acc))

                # graph_def = tf.get_default_graph().as_graph_def()
                # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                #                                                                 ['output'])
                #
                # with tf.gfile.GFile("./s_cls2.pb", 'wb') as f:
                #     f.write(output_graph_def.SerializeToString())
                # #
                #
                # saver.save(sess,'./save/s_cls2.dpk')

                print('------------test iter :{},  test loss:{},  test acc:{}----------'.format(i,test_loss,test_acc))






















