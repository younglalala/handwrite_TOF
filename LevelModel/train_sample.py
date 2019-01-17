import tensorflow as tf
import os
import random
import cv2
import numpy as np

from PIL import Image


trainimg_path="/Users/wywy/Desktop/train_img1"
train_filename = './train.tfrecords'

data_dict=dict(zip(list('ABCDX'),[0,1,2,3,4]))


def saver_lables(img_path,train_filename,img_size):
    writer = tf.python_io.TFRecordWriter(train_filename)

    all_filename=[]
    for file in os.listdir(img_path):
        if file =='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            all_filename.append(file)
    random.shuffle(all_filename)
    for f in all_filename:
        label = float(data_dict.get(f.split('.')[0].split('_')[-1]))

        img=Image.open(os.path.join(img_path,f))
        img = img.resize(img_size, Image.ANTIALIAS)
        img=img.convert('L')
        image = img.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'lables': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'file_name':tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(f)]))
        }))

        writer.write(example.SerializeToString())
    writer.close()


def read_data_for_file(file, capacity,image_size):
    filename_queue = tf.train.string_input_producer([file], num_epochs=None, shuffle=False, capacity=capacity)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'lables': tf.FixedLenFeature([], tf.float32),
            'images':tf.FixedLenFeature([], tf.string),
            'file_name':tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features['images'], tf.uint8)
    img=tf.reshape(img,image_size)
    img = tf.cast(img, tf.float32)
    setoff_lables=features['lables']
    file_name= tf.cast(features['file_name'], tf.string)


    return img, setoff_lables,file_name


def train_shuffle_batch(train_file_path,image_size, batch_size, capacity=7000, num_threads=3):
    images, setoff_lables ,file_name= read_data_for_file(train_file_path, 10000,image_size)

    images_,  setoff_lables_,file_name_ = tf.train.shuffle_batch([images,setoff_lables,file_name], batch_size=batch_size, capacity=capacity,
                                               min_after_dequeue=1000,
                                               num_threads=num_threads)

    return images_, setoff_lables_,file_name_




if __name__=='__main__':
    init = tf.global_variables_initializer()
    saver_lables(trainimg_path,train_filename,(64,64))
    # read_data_for_file(train_filename,100,[16,168,3])
    a=train_shuffle_batch(train_filename,[64,64,1],100)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)

        sess.run(init)
        aa,bb,cc=sess.run(a)
        print(aa[0])
        print(bb[0])
        print(cc[0])

