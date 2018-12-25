import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np

img_path='/Users/wywy/Desktop/level_out'
all_image=[]
all_image1=[]
all_label=[]
for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.remove(os.path.join(img_path,file))
    else:
        # img=cv2.imread(os.path.join(img_path,file),0)
        # img=cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        # img=img.reshape((64,64,1))


        img = Image.open(os.path.join(img_path, file))
        img = img.resize((64,64), Image.ANTIALIAS)
        img = img.convert('L')
        img=np.array(img)
        all_image.append(img.reshape([64,64,1])/255-0.5)
        all_image1.append(img)
        label=file.split('.')[0].split('_')[-1]
        all_label.append(label)





choice_dict=dict(zip([0,1,2,3,4],list('ABCDX')))

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = "./level_modle11.pb"

    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        input_x = sess.graph.get_tensor_by_name("input:0")
        model_output = sess.graph.get_tensor_by_name("output:0")
        # dp=sess.graph.get_tensor_by_name("dp:0")

        output = sess.run(model_output, feed_dict={input_x:np.array(all_image)})
        for ii in range(len(output)):
            if choice_dict.get(output[ii])==all_label[ii]:
                pass
                # cv2.imwrite('/Users/wywy/Desktop/t/' + str(ii) + '_' + choice_dict.get(output[ii]) + '.jpg',
                #             all_image1[ii].reshape([64, 64]))
            else:
                cv2.imwrite('/Users/wywy/Desktop/f/'+str(ii)+'_'+choice_dict.get(output[ii])+'.jpg',all_image1[ii].reshape([64,64]))