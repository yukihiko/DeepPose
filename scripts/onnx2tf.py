import os
import onnx_tf.backend
import onnx
import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np
from PIL import Image

img_path = "./checkpoints/tennis_in_crowd.jpg"
model_path = "result/onnx_output.onnx"


def main():
    
    # 画像の読み込みと加工
    img = Image.open(img_path)
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32)[np.newaxis, :, :, :]
    arr = arr.transpose(0, 3, 1, 2)


    # ONNX形式のモデル読み込み
    onnx_model = onnx.load(model_path)

    # TensorFlowでONNX形式のモデルを実行
    tf_model = onnx_tf.backend.prepare(onnx_model, device='CPU')
    

    save_dir = './result/tensorflow/tf_MobileNet_'
    save_path = os.path.join(save_dir, 'model.pb')

    #result = sess.run(tf_model, feed_dict={ 0: arr })
    #result = tf_model.runandsave(arr, save_path)
    result = tf_model.run(arr)
    tf_model.export_graph(path=save_path)

    with tf.Session() as sess:
        print("load graph")
        sess.run(tf.global_variables_initializer())

        with gfile.FastGFile(save_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')


        inp = sess.graph.get_tensor_by_name('0:0')
        out1 = sess.graph.get_tensor_by_name('Add_1:0')
        out2 = sess.graph.get_tensor_by_name('Sigmoid:0')

        feed_dict = {inp: arr}

        result = sess.run([out1, out2], feed_dict)

        saver = tf.train.Saver()


    # 確率が高い順にクラスIDを昇順で出力
    #prob = np.argsort(result.prob_1[0])[::-1]
    print("===== [Prob] =====")
    print(result)

    # 確率が上位5個のクラスIDとその確率を表示する
    #print("===== [TOP 5] =====")
    #for i in range(5):
    #    print("{}: {}%".format(prob[i], result.prob_1[0][prob[i]] * 100))


if __name__ == "__main__":
    main()