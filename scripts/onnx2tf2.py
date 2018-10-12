import os
import onnx_tf.backend
import onnx
import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np
from PIL import Image

img_path = "./checkpoints/tennis_in_crowd.jpg"

def main():

    # 画像の読み込みと加工
    img = Image.open(img_path)
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32)[np.newaxis, :, :, :]
    arr = arr.transpose(0, 3, 1, 2)

    save_dir = './result/tensorflow/tf_MobileNet_'
    pb_save_path = os.path.join(save_dir, 'model.pb')
    save_path = os.path.join(save_dir, 'model.ckpt')

    print("Session")
    inti = tf.global_variables_initializer()

    with tf.Session() as sess:
        print("load graph")
        sess.run(inti)

        with gfile.FastGFile(pb_save_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        inp = sess.graph.get_tensor_by_name('0:0')
        out1 = sess.graph.get_tensor_by_name('Add_1:0')
        out2 = sess.graph.get_tensor_by_name('Sigmoid:0')

        feed_dict = {inp: arr}

        result = sess.run([out1, out2], feed_dict)

        print("save file")

        tf.train.write_graph(sess.graph,"./result/tensorflow/tf_MobileNet_/","model.pbtxt")
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

if __name__ == "__main__":
    main()