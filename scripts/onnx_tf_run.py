import onnx_tf.backend
import onnx

import numpy as np
from PIL import Image

img_path = "im07276.jpg"
model_path = "result/onnx_output.onnx"


def main():
    # 画像の読み込みと加工
    img = Image.open(img_path)
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32)[np.newaxis, :, :, :]
    arr = arr.transpose(0, 3, 1, 2)/255.

    # ONNX形式のモデル読み込み
    onnx_model = onnx.load(model_path)

    # TensorFlowでONNX形式のモデルを実行
    tf_model = onnx_tf.backend.prepare(onnx_model, device='CPU')
    offset, heatmap = tf_model.run(arr)

    print("onnx heatmap")
    col = 14
    for i in range(col):
        str = ""
        for j in range(col):
            str = str + ",{:.3f}".format(heatmap[0, 9, i, j]) 
        print(str)
    print("onnx offset")
    for i in range(col):
        str = ""
        for j in range(col):
            str = str + ",{:.3f}".format(offset[0, 9, i, j]) 
        print(str)


if __name__ == "__main__":
    main()
