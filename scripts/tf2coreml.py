#from tfcoreml import tfcoreml as tf_converter
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.saved_model import tag_constants

from keras.preprocessing.image import load_img
import sys
sys.path.append("./")
from tfcoreml._tf_coreml_converter import convert

#import tfcoreml
#import coremltools
import yaml

def frozen_graph_maker(export_dir,output_graph):
        with tf.Session(graph=tf.Graph()) as sess:
                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
                output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
                output_graph_def = tf.graph_util.convert_variables_to_constants(
                        sess, # The session is used to retrieve the weights
                        sess.graph_def,
                        output_nodes# The output node names are used to select the usefull nodes
               )       
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())

f = open("config.yaml", "r+")
cfg = yaml.load(f)
imageSize = cfg['imageSize']
checkpoints = cfg['checkpoints']
chk = cfg['chk']
chkpoint = checkpoints[chk]
versionName = chkpoint.lstrip('mobilenet_')

# Provide these to run freeze_graph:
# Graph definition file, stored as protobuf TEXT
graph_def_file = './models/model.pbtxt'
# Trained model's checkpoint name
checkpoint_file = './checkpoints/model.ckpt'
# Frozen model's output name
frozen_model_file = './models/frozen_model.pb'
# Output nodes. If there're multiple output ops, use comma separated string, e.g. "out1,out2".
#output_node_names = 'heatmap,offset_2,displacement_fwd_2,displacement_bwd_2'
output_node_names = 'MobileNetn2nConv2dnfc2n219_bias'
# output_node_names = 'Softmax' 

#frozen_graph_maker('result/tensorflow/tf_MobileNet_2',frozen_model_file)
image = tf.placeholder(tf.float32, shape=[1, 224, 224, 3],name='image')
with tf.Session() as sess:  
        meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], 'result/tensorflow/tf_MobileNet_2/')
        h_est = sess.run('MobileNetn2nConv2dnfc2n219_bias:0', feed_dict={
            image: [np.ndarray(shape=(224, 224, 3),dtype=np.float32)]
        })
        saver = tf.train.Saver()
        model_signature = meta_graph.signature_def['serving_default']
        input_signature = model_signature.inputs
        output_signature = model_signature.outputs
        print(model_signature)
        print(input_signature)
        print(input_signature)
        input_tensor_name = input_signature['inputs'].name
        prob_tensor_name = output_signature['probabilities'].name
        label_tensor_name = output_signature['classes'].name
        print(input_tensor_name)
        print(prob_tensor_name)
        print(label_tensor_name)

        save_dir = './result/tensorflow/tf_MobileNet_2'
        save_path = os.path.join(save_dir, 'model.ckpt')
        save_path = saver.save(sess, save_path)

        tf.train.write_graph(sess.graph,"./result/tensorflow/tf_MobileNet_2/","model.pbtxt")
graph_def_file = 'result/tensorflow/tf_MobileNet_2/model.pbtxt'
checkpoint_file = 'result/tensorflow/tf_MobileNet_2/model.ckpt'
'''
input_saved_model_dir = "result/tensorflow/tf_MobileNet_2"
input_meta_graph = False
saved_model_tags = tag_constants.SERVING
'''
'''
freeze_graph(None, "",
                            True, None, "MobileNetn2nConv2dnfc2n219_bias",
                              None, None,
                              frozen_model_file, True, "", "", "",
                              input_meta_graph, input_saved_model_dir, saved_model_tags)
'''
'''
export_dir = 'result/tensorflow/tf_MobileNet_2'
freeze_graph(
            input_graph=os.path.join(export_dir, 'saved_model.pb'), input_saver="", input_binary=True,
            input_checkpoint=os.path.join(export_dir, 'variables', 'variables.data-00000-of-00001'),
            output_node_names="MobileNetn2nConv2dnfc2n219_bias", restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0",
            output_graph=os.path.join(export_dir, 'saved_model_frozen.pb'), clear_devices=True, initializer_nodes="")
'''

# Call freeze graph
freeze_graph(input_graph=graph_def_file,
             input_saver="",
             input_binary=False,
             input_checkpoint=checkpoint_file,
             output_node_names=output_node_names,
             restore_op_name="save/restore_all",
             filename_tensor_name="save/Const:0",
             output_graph=frozen_model_file,
             clear_devices=True,
             initializer_nodes="")


input_tensor_shapes = {"image:0":[1,224, 224, 3]} 
coreml_model_file = './tfcoremodel.mlmodel'
# output_tensor_names = ['output:0']
output_tensor_names = ['MobileNetn2nConv2dnfc2n219_bias:0']
#output_tensor_names = ['heatmap:0','offset_2:0','displacement_fwd_2:0','displacement_bwd_2:0']

print("convert coreml")

coreml_model = convert(
        tf_model_path=frozen_model_file, 
        mlmodel_path=coreml_model_file, 
        input_name_shape_dict=input_tensor_shapes,
        image_input_names=['image:0'],
        output_feature_names=output_tensor_names,
        is_bgr=False,
        red_bias = -1, 
        green_bias = -1, 
        blue_bias = -1, 
        image_scale = 0.00392157)


coreml_model.author = 'Infocom TPO'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Ver.0.0.1'

coreml_model.save('./models/posenet'+ str(imageSize) + '_' + versionName +'.mlmodel')
'''
img = load_img("./checkpoints/tennis_in_crowd.jpg", target_size=(imageSize, imageSize))
print(img)
out = coreml_model.predict({'image__0': img})['heatmap__0']
print("#output coreml result.")

print(out.shape)
print(np.transpose(out))
print(out)
# print(out[:, 0:1, 0:1])
print(np.mean(out))
'''