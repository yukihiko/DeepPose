import onnx_tf.backend

import onnx

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchvision import transforms

import sys
sys.path.append("./")
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale

img_path = "data/images/lsp_dataset/im0001.jpg"
model_path = "test.onnx"
filename = "data/test"

def main():
    # 画像の読み込み
    dataset = PoseDataset(
        filename,
        input_transform=transforms.Compose([
            transforms.ToTensor(),
            RandomNoise()]),
        output_transform=Scale(),
        transform=Crop(data_augmentation=False))

    img, pose, _, _ = dataset[10]
    arr = img.unsqueeze(0)

    #img = Image.open(img_path)
    #img = img.resize((224, 224))
    #arr = np.asarray(img, dtype=np.float32)[np.newaxis, :, :, :]
    #arr = arr.transpose(0, 3, 1, 2)

    # ONNX形式のモデル読み込み
    onnx_model = onnx.load(model_path)

    # モデル（グラフ）を構成するノードを全て出力する
    print("====== Nodes ======")
    for i, node in enumerate(onnx_model.graph.node):
        print("[Node #{}]".format(i))
        print(node)

    # TensorFlowでONNX形式のモデルを実行
    tf_model = onnx_tf.backend.prepare(onnx_model, device='CPU')
    result = tf_model.run(arr)

    # 出力
    print("===== [result] =====")
    print(result)
    print("===== [Pose] =====")
    print(pose)

    fig = plt.figure(figsize=(2.24, 2.24))
    img = img.numpy().transpose(1, 2, 0)
    plt.imshow(img, vmin=0., vmax=1.)
    prob = result[0]
    prob1 = prob[0]
    for i in range(14):
        x = prob1[i * 2]
        y = prob1[i * 2 + 1]
        plt.scatter(x * 224, y * 224, color=cm.hsv(i/14.0),  s=5)
    plt.axis("off")
    plt.savefig('check.png')
    plt.close(fig)



if __name__ == "__main__":
    main()
