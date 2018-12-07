

import os
from PIL import Image
import cv2
import random
import tensorflow as tf
import numpy as np
from scipy import misc

from skimage import data, exposure, img_as_float
def choice1(list,index,number):
    new_list=[]
    for i in range(len(list)):
        if i in index:
            pass
        else:
            new_list.append(list[i])
    choice=random.sample(new_list,number)
    all_choice=[]
    for c in choice:
        p = list.index(c)
        all_choice.append(p)
    return all_choice,choice
#
# choice_set = [(0, 0, 42, 32), (42, 0, 84, 32), (84, 0, 126, 32), (126, 0, 168, 32)
#         , (168, 0, 210, 32), (210, 0, 252, 32), (252, 0, 294, 32)]




# len_chioce=5
# choice_num=4
# img_path2='/Users/wywy/Desktop/选项1'
# img_path='/Users/wywy/Desktop/数据分类/55_1'.format(len_chioce)
# save_path='/Users/wywy/Desktop/数据分类/4个选项'.format(choice_num)

def gen_mchoice(len_chioce,choice_num,choice_path,orig_path,save_path):


    sett = [(0, 0), (42, 0), (84, 0), (126, 0), (168, 0), (210, 0), (252, 0)]
    all_pp=[]
    for file1 in os.listdir(choice_path):
        if file1=='.DS_Store':
            os.remove(choice_path+'/'+file1)
        else:
            all_pp.append(file1)
    random.shuffle(all_pp)

    cc=0
    for i in range(10):
        all_file=[]
        for file in os.listdir(orig_path):
            if file=='.DS_Store':
                os.remove(orig_path+'/'+file)
            else:
                all_file.append(file)
        random.shuffle(all_file)
        for f in all_file:
            if cc<20000+0:
                new_setdict=dict(zip([3,4,5,6,7],[sett[:2],
                                                  sett[:3],
                                                  sett[:4],
                                                  sett[:5],
                                                  sett[:6]]))
                new_set=new_setdict.get(len_chioce)
                chioce_num=choice_num-1
                img=Image.open(orig_path+'/'+f)
                f_name=f.split('.')[0].split('_')[-1]
                index_dict1=dict(zip(list('ABCDEFG'),sett))
                index_dict2 = dict(zip(sett, list('ABCDEFG')))
                if index_dict1.get(f_name) in new_set:
                    new_set.remove(index_dict1.get(f_name))
                randomlist = random.sample(new_set[:], chioce_num)
                for ss in randomlist:
                    name=index_dict2.get(ss)
                    p_img=Image.open(choice_path+'/'+random.sample(all_pp,choice_num)[0])
                    img.paste(p_img,ss)
                    f_name+='_'+name
                img.save(save_path+'/'+str(cc)+'_'+f_name+'.jpg')
                cc+=1
    print(cc)

# gen_mchoice(len_chioce,choice_num,img_path2,img_path,save_path)
# img_path='/Users/wywy/Desktop/客观题分类模型9 30/data/all_cls_ok10'
# # img_path='/Users/wywy/Desktop/all_train1'
# save_path='/Users/wywy/Desktop/cls_train'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#
#         name=file.split('.')[0].split('_')[-1]
#         if name=='7':
#             if c<40000:
#                 img=Image.open(img_path+'/'+file)
#                 img.save(save_path+'/'+str(c)+'_7.jpg')
#                 c+=1
# print(c)


# img_path='/Users/wywy/Desktop/cls_train'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='6':
#             os.remove(img_path+'/'+file)
#             c+=1
# print(c)


# img_path='/Users/wywy/Desktop/分类/7'
# save_path='/Users/wywy/Desktop/分类/all_test'
# c=1010
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         bg=Image.new('RGB',(294,32),'white')
#         img=Image.open(img_path+'/'+file).resize((294,32))
#         bg.paste(img,(0,0))
#         bg.resize((168,32))
#         name=list(file.split('.')[0].split('_')[-1])
#         if len(name)==2:
#             bg.save(save_path+'/'+str(c)+'_1.jpg')
#         if len(name)==1:
#             bg.save(save_path + '/' + str(c) + '_0.jpg')
#         c+=1
#
#
#
# print(c)



#
# all_file=[]
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         # img=Image.open(img_path+'/'+file)
#         all_file.append(file)
# random.shuffle(all_file)
# for f in all_file:
#     # if c<5000:
#         bg=Image.new('RGB',(294,32),'white')
#         img=Image.open(img_path+'/'+f).resize((210,32))
#         bg.paste(img,(0,0))
#         bg.save(save_path+'/'+f)
#         c+=1
# print(c)
# img_path='/Users/wywy/Desktop/数据分类/all_chioce/all_chioce'
# save_path='/Users/wywy/Desktop/数据分类/all_chioce/aa'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[1:]
#         if len(name)==1:
#             if c<10000:
#                 img=Image.open(img_path+'/'+file)
#                 img.save(save_path+'/g'+str(c)+'_0.jpg')
#                 c+=1
# print(c)
# img_path='/Users/wywy/Desktop/数据分类/all_chioce/all_chioce'
# save_path='/Users/wywy/Desktop/all_cls_ok10'
#
# all_file=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         all_file.append(file)
# random.shuffle(all_file)
# c=8478
# for f in all_file:
#     name=f.split('.')[0].split('_')[1:]
#     if len(name)==4 and c<10000:
#
#         img=Image.open(img_path+'/'+f)
#         img.save(save_path+'/g'+str(c)+'_3.jpg')
#         c+=1
# print(c)

# choice_set = [(0, 0, 42, 32), (42, 0, 84, 32), (84, 0, 126, 32), (126, 0, 168, 32)
#         , (168, 0, 210, 32), (210, 0, 252, 32), (252, 0, 294, 32)]

# img_path='/Users/wywy/Desktop/box_cls/7'
# save_path='/Users/wywy/Desktop/box_cls/all_test'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file).resize((294,32),Image.ANTIALIAS)
#         bg=Image.new('RGB',(294,32),'white')
#         bg.paste(img,(0,0))
#         bg=bg.resize((168,32),Image.ANTIALIAS)
#         bg.save(save_path+'/'+file)




# img_path='/Users/wywy/Desktop/客观题分类数据/分类/all_test'
# for file in os.listdir(img_path):
#     if file == '.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img =  img.resize((168,32))
#         img.save(img_path+'/'+file)

# #
# import time
# start = time.clock()
#
# #
# #
# img_path='/Users/wywy/Desktop/cls_1'
# save_path1='/Users/wywy/Desktop/test_save'
# save_path2='/Users/wywy/Desktop/f'
def get_data(img_path):
    all_file=[]
    all_label=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=cv2.imread(img_path+'/'+file,0)
            img=img.reshape(([32,168,1]))/255-0.5
            name=file.split('.')[0].split('_')[-1]
            all_label.append(int(name))
            all_file.append(img)
    return all_file,all_label





# with tf.Graph().as_default():
#     output_graph_def = tf.GraphDef()
#     output_graph_path = "./s_cls1.pb"
#     elapsed1 = (time.clock() - start)
#     print(elapsed1,'恢复图')
#
#     with open(output_graph_path, "rb") as f:
#         output_graph_def.ParseFromString(f.read())
#         tf.import_graph_def(output_graph_def, name="")
#         elapsed2 = (time.clock() - start)
#         print(elapsed2, '恢复数据')
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#
#         input_x = sess.graph.get_tensor_by_name("input:0")
#         model_output = sess.graph.get_tensor_by_name("output:0")
#         elapsed3 = (time.clock() - start)
#         print(elapsed3, '模型恢复')
#         # dp=sess.graph.get_tensor_by_name("dp:0")
#         all_file,all_label=get_data(img_path)
#         elapsed4 = (time.clock() - start)
#         print(elapsed4, '数据读取')
#         output = sess.run(model_output, feed_dict={input_x: np.array(all_file)})
#         elapsed5 = (time.clock() - start)
#         print(elapsed5, '识别成功')
#         for f in range(len(all_file)):
#             if output[f]!=all_label[f]:
#                 cv2.imwrite(save_path2+'/'+str(f)+'_'+str(output[f])+'.jpg',((all_file[f]+0.5)*255).reshape(32,168))
#             else:
#                 cv2.imwrite(save_path1 + '/' + str(f) + '_' + str(output[f]) + '.jpg',
#                             ((all_file[f] + 0.5) * 255).reshape(32, 168))
#
#         print(output)
#
#



# def bright(img):
#     gam2 = exposure.adjust_gamma(img, 0.5)
#     return gam2
#
# #数据分类
# #
# img_path='/Users/wywy/Desktop/数据增强'
# save_path='/Users/wywy/Desktop/all_cls_ok'
# c=5200
# for i in range(100):
#     for file in os.listdir(img_path):
#         if file=='.DS_Store':
#             os.remove(os.path.join(img_path,file))
#         else:
#             img=Image.open(os.path.join(img_path,file))
#             name=file.split('.')[0].split('_')[-1]
#             img.save(save_path+'/radd'+str(c)+'_'+name+'.jpg')
#             # print(save_path+'/radd'+str(c)+'_'+name+'.jpg')
#             c+=1
# print(c)


# img_path='/Users/wywy/Desktop/error'
# save_path='/Users/wywy/Desktop/error_cls/all_image'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         for f in os.listdir(img_path+'/'+file):
#             if f == '.DS_Store':
#                 os.remove(img_path + '/' + file+'/'+f)
#             else:
#                 img=Image.open(img_path+'/'+file+'/'+f)
#                 name=f.split('.')[0].split('_')[-1]
#                 img.save(save_path+'/'+str(c)+'_'+name+'.jpg')
#                 c+=1
# print(c)

# img_path='/Users/wywy/Desktop/error_cls/all_image'
# save_path='/Users/wywy/Desktop/error_cls/7'
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img_size=img.size
#         if img_size[0]>210:
#             img.save(save_path+'/'+file)
#             c+=1
# print(c)
#
# img_path='/Users/wywy/Desktop/error_cls/44'
# save_path='/Users/wywy/Desktop/error_cls/crop_image/00'
# save_path1='/Users/wywy/Desktop/error_cls/crop_image/11'
# choice_set = [(0, 0, 42, 32), (42, 0, 84, 32), (84, 0, 126, 32), (126, 0, 168, 32)
#         , (168, 0, 210, 32), (210, 0, 252, 32), (252, 0, 294, 32)]
#
# chioce_dict=dict(zip(list('ABCDEFG'),[0,1,2,3,4,5,6]))
# chioce_num=4
# c=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file).resize((168,32),Image.ANTIALIAS)
#         img=np.array(img)
#         name=list(file.split('.')[0].split('_')[-1])
#         if len(name)==1 and name[0]=='X':
#             for ii in range(chioce_num):
#                 erosion_img = img[choice_set[ii][1]:choice_set[ii][3], choice_set[ii][0]:choice_set[ii][2]]
#                 cv2.imwrite(save_path + '/' + str(c) + '_0.jpg', erosion_img)
#                 c+=1
#         else:
#             n_index=chioce_dict.get(name[0])
#             for ii in range(chioce_num):
#                 if ii==n_index:
#                     erosion_img = img[choice_set[ii][1]:choice_set[ii][3], choice_set[ii][0]:choice_set[ii][2]]
#                     cv2.imwrite(save_path1+'/'+str(c)+'_1.jpg',erosion_img)
#                     c+=1
#
#                 else:
#                     erosion_img = img[choice_set[ii][1]:choice_set[ii][3], choice_set[ii][0]:choice_set[ii][2]]
#                     cv2.imwrite(save_path + '/' + str(c) + '_0.jpg', erosion_img)
#                     c+=1


img_path='/Users/wywy/Desktop/1'
save_path='/Users/wywy/Desktop/train_cls1'
all_file=[]
for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.remove(img_path+'/'+file)
    else:
        all_file.append(file)
random.shuffle(all_file)
c=0
for f in all_file:
    if c<110000:
        img=Image.open(img_path+'/'+f)
        img.save(save_path+'/'+f)
        c+=1
print(c)












