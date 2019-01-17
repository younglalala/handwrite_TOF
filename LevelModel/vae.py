import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from LevelModel.train_sample import  *
from LevelModel.test_sample import *



# def lrelu(x, alpha=0.3):
#     return tf.maximum(x, tf.multiply(x, alpha))

# def conv2d(x,input_dim,output_dim,k_size,strides,name):
#     w=tf.get_variable(name =name+'_w',shape=[k_size,k_size,input_dim,output_dim],initializer=tf.random_normal_initializer())
#     b=tf.get_variable(name=name+'_b',shape=[output_dim],initializer=tf.ones_initializer())
#     cond2d_out=tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME',name=name)+b
#     cond2d_out=tf.nn.leaky_relu(cond2d_out,alpha=0.3)
#
#     return cond2d_out
#
#
# def fc(x,input_dim,output_dim,name):
#     w=tf.get_variable(name =name+'_w',shape=[input_dim,output_dim],initializer=tf.random_normal_initializer())
#     b=tf.get_variable(name=name+'_b',shape=[output_dim],initializer=tf.ones_initializer())
#     fc_out=tf.multiply(x,w)+b
#     fc_out=tf.nn.leaky_relu(fc_out,alpha=0.3)
#
#     return fc_out
#
#
#
# batch_size=64
# dec_in_chanels=1
# n_latent=8
# inputs_decoder = 49 * dec_in_chanels / 2
# reshaped_dim = [-1, 7, 7, dec_in_chanels]
#
# class VAE:
#     def __init__(self):
#
#         self.x=tf.placeholder(dtype=tf.float32,shape=[None,64,64,1],name='X')
#         self.y_=tf.placeholder(dtype=tf.float32,shape=[None,64,64,1])
#
#         self.y_flat=tf.reshape(self.y_,[-1,64,64])
#         self.dp=tf.placeholder(dtype=tf.float32)
#
#
#     def encoder(self):
#         with tf.variable_scope('encoder',reuse=None):
#             self.x=conv2d(self.x,1,64,k_size=4,strides=2,name='encoder_conv1')
#             self.x=tf.nn.dropout(self.x,keep_prob=self.dp)
#
#             self.x=conv2d(self.x,64,64,k_size=4,strides=2,name='encoder_conv2')
#             self.x=tf.nn.dropout(self.x,keep_prob=self.dp)
#
#             self.x=conv2d(self.x,64,64,k_size=4,strides=1,name='encoder_conv3')
#             self.x=tf.nn.dropout(self.x,keep_prob=self.dp)
#
#             self.flat=tf.contrib.layers.flatten(self.x)   #展平成【batch,-1】
#             #全连接
#             self.mean=tf.layers.dense(self.flat,units=n_latent)
#             self.stddev=0.5*tf.layers.dense(self.flat,units=n_latent)
#
#             self.epsilon=tf.random_normal(tf.stack([tf.shape(self.flat)[0], n_latent]))
#             self.z=self.mean+tf.multiply(self.epsilon,tf.exp(self.stddev))
#
#     def decoder(self):
#         with tf.variable_scope('decoder',reuse=None):
#             self.x=fc(self.z,)

# class VAE:
#     def __init__(self):
#         self.batch_size = 1
#
#         self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64], name='X')
#         self.Y    = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64], name='Y')
#         self.Y_flat = tf.reshape(self.Y, shape=[-1, 64 * 64])
#         self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
#
#         self.dec_in_channels = 1
#         self.n_latent = 12
#
#         self.reshaped_dim = [-1, 4, 4, self.dec_in_channels]
#         self.inputs_decoder = 16 * self.dec_in_channels / 2
#
#
#     def encoder(self):
#         with tf.variable_scope("encoder", reuse=None):
#             X = tf.reshape(self.X_in, shape=[-1, 64, 64, 1])
#             x = tf.nn.leaky_relu(tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same'),alpha=0.3)    #32
#             x = tf.nn.dropout(x, self.keep_prob)
#             x = tf.nn.leaky_relu(tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same' ),alpha=0.3  ) #16
#             x = tf.nn.dropout(x, self.keep_prob)
#             x = tf.nn.leaky_relu(tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='same'),alpha=0.3 ) #8
#             x = tf.nn.dropout(x, self.keep_prob)
#             x = tf.nn.leaky_relu(tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='same'),alpha=0.3  )#4
#             x = tf.nn.dropout(x, self.keep_prob)
#             x = tf.nn.leaky_relu(tf.layers.conv2d(x, filters=128, kernel_size=4, strides=1, padding='same'),alpha=0.3  ) #4
#             x = tf.nn.dropout(x, self.keep_prob)
#             x = tf.contrib.layers.flatten(x)
#             self.mn = tf.layers.dense(x, units=self.n_latent)
#             self.sd = 0.5 * tf.layers.dense(x, units=self.n_latent)
#             self.epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))
#             self.z = self.mn + tf.multiply(self.epsilon, tf.exp(self.sd))
#
#             # return self.z, self.mn, self.sd
#
#
#     def decoder(self):
#         with tf.variable_scope("decoder", reuse=None):
#             x = tf.nn.leaky_relu(tf.layers.dense(self.z, units=self.inputs_decoder),alpha=0.3  )#(?, 32)
#             x = tf.nn.leaky_relu(tf.layers.dense(x, units=self.inputs_decoder * 2 ),alpha=0.3  )#(?, 64)
#             x = tf.reshape(x, self.reshaped_dim)   #(?, 8, 8, 1)
#             x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)  #(?, 16, 16, 128)
#             x = tf.nn.dropout(x, self.keep_prob)
#             x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same',
#                                            activation=tf.nn.relu)  # (?, 32, 32, 128)
#             x = tf.nn.dropout(x, self.keep_prob)
#             x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same',
#                                            activation=tf.nn.relu)  # (?, 64, 64, 128)
#             x = tf.nn.dropout(x, self.keep_prob)
#             x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)  #(?, 64, 64, 128)
#             x = tf.nn.dropout(x, self.keep_prob)
#             x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)  #(?, 64, 64, 128)
#             x = tf.contrib.layers.flatten(x)   #(?, 64*64*128)
#             x = tf.layers.dense(x, units=64 * 64, activation=tf.nn.sigmoid)#(?, 64 * 64)
#             self.img = tf.reshape(x, shape=[-1, 64, 64])  #
#
#
#     def banckward(self):
#         self.unreshaped = tf.reshape(self.img, [-1, 64 * 64])
#         self.img_loss = tf.reduce_sum(tf.squared_difference(self.unreshaped, self.Y_flat), 1)
#         self.latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.sd - tf.square(self.mn) - tf.exp(2.0 * self.sd), 1)
#         self.loss = tf.reduce_mean(self.img_loss + self.latent_loss)
#         self.optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.loss)
#
#
# # sampled, mn, sd = encoder(X_in, keep_prob)
# # print('encoder----')
# # dec = decoder(sampled, keep_prob)
# # print('decoder----')
# if __name__=='__main__':
#     vae=VAE()
#     vae.encoder()
#     vae.decoder()
#     vae.banckward()
#     init=tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         for i in range(100):
#             print(i)


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(30):
#     print(i)
# 随机获取训练图片
def all_file(img_path):
    all_file=list()
    for file in os.listdir(img_path):
        all_file.append(os.path.join(img_path,file))
    random.shuffle(all_file)

    return all_file

def get_data(all_file,batch_size):
    file=random.sample(all_file,batch_size)
    all_img=list()
    for f in file:
        img=Image.open(f).resize((64,64)).convert('L')
        img=np.array(img)
        all_img.append(img)
    all_img=np.array(all_img).reshape([-1,64,64])

    return all_img
img_path='/Users/wywy/Desktop/train_level'
all_file1=all_file(img_path)


#
# for i in range(30000):
#
#     batch =get_data(all_file1,16)
#     print(batch.shape)
#     _,d=sess.run([optimizer,dec], feed_dict={X_in: batch, Y: batch, keep_prob: 0.8})
#     plt.clf()
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(batch[0], cmap='gray')
#     plt.title('out')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(d[0], cmap='gray')
#     plt.title('out')
#     plt.pause(0.1)
#
# plt.show()
#
#     # if not i % 200:
#     #     ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd],
#     #                                            feed_dict={X_in: batch, Y: batch, keep_prob: 1.0})
#     #     plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
#     #     plt.show()
#     #     plt.imshow(d[0], cmap='gray')
#     #     plt.show()
#     #     print(i, ls, np.mean(i_ls), np.mean(d_ls))
#

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


tf.reset_default_graph()

batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 64 * 64])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 16, 16, dec_in_channels]
inputs_decoder = 16*16 * dec_in_channels / 2


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 64, 64, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)  #32
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)   #16
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=64*64, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 64, 64])
        return img


sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)


unreshaped = tf.reshape(dec, [-1, 64*64])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(30000):
    # batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    batch1 = get_data(all_file1, batch_size)
    batch=batch1/255-0.5


    _,d=sess.run([optimizer,dec], feed_dict={X_in: batch, Y: batch, keep_prob: 0.8})
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.imshow(batch1[0], cmap='gray')
    plt.title('out')

    plt.subplot(1, 2, 2)
    plt.imshow(d[0], cmap='gray')
    plt.title('out')
    plt.pause(0.1)

    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd],
                                               feed_dict={X_in: batch, Y: batch, keep_prob: 1.0})

        print(i, ls, np.mean(i_ls), np.mean(d_ls))

plt.show()
