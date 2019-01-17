import  tfcoreml as tf_converter
import tensorflow as tf
import numpy as np


# #转换成mlmodel
# tf_converter.convert(tf_model_path='./m_3.pb',
#                      mlmodel_path='./level_mobile.mlmodel',
#                      output_feature_names=['output:0'],
#                      input_name_shape_dict={'input:0':[1,64,64,1]})
# #



#转换成tflite

# graph_def_file = "./mobile2"
# input_arrays = ["input"]
# output_arrays = ["output"]
# converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(graph_def_file,input_arrays,output_arrays)#,input_shapes={"image_tensor":[None,300,300,3]})
# tflite_model = converter.convert()
# open("111111.tflite", "wb").write(tflite_model)