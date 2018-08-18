# -*- coding: utf-8 -*-

import argparse
from torch.autograd import Variable
import torch.onnx
import torchvision.models as models
import onnx
import numpy as np
#from keras.models import load_model

import sys
sys.path.append("./")
from onnx_coreml.converter import convert
#from pytorch2keras.converter import pytorch_to_keras
from modules.errors import FileNotFoundError, GPUNotFoundError, UnknownOptimizationMethodError, NotSupportedError
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNetV2, MobileNet_, MobileNet_2, MobileNet_3, MobileNet___, MnasNet
#from coremltools.converters.keras import convert
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale
from torchvision import transforms

print('ArgumentParser')
parser = argparse.ArgumentParser(description='Convert PyTorch model to CoreML')
parser.add_argument('--input', '-i', required=True, type=str)
parser.add_argument('--output', '-o', required=True, type=str)
parser.add_argument('--NN', '-n', required=True, type=str)
parser.add_argument('--onnx_output', required=True, type=str)
args = parser.parse_args()

print('Set up model')
if args.NN == "MobileNet":
    model = MobileNet( )
elif args.NN == "MobileNet_":
    model = MobileNet_( )
elif args.NN == "MobileNet___":
    model = MobileNet___( )
elif args.NN == "MobileNet_3":
    model = MobileNet_3( )
elif args.NN == "MnasNet":
    model = MnasNet( )

model.load_state_dict(torch.load(args.input))
#model = model.cpu()
model.eval()

# export to ONNF
dummy_input = Variable(torch.randn(1, 3, 224, 224))

print('converting to ONNX')
torch.onnx.export(model, dummy_input, args.onnx_output)
onnx_model = onnx.load(args.onnx_output)

# モデル（グラフ）を構成するノードを全て出力する
print("====== Nodes ======")
for i, node in enumerate(onnx_model.graph.node):
    print("[Node #{}]".format(i))
    print(node)

# モデルの入力データ一覧を出力する
print("====== Inputs ======")
for i, input in enumerate(onnx_model.graph.input):
    print("[Input #{}]".format(i))
    print(input)

# モデルの出力データ一覧を出力する
print("====== Outputs ======")
for i, output in enumerate(onnx_model.graph.output):
    print("[Output #{}]".format(i))
    print(output)

print('converting coreml model')
mlmodel = convert(
        onnx_model, 
        preprocessing_args={'is_bgr':True, 'red_bias':0., 'green_bias':0., 'blue_bias':0., 'image_scale':0.00392157},
        image_input_names='0')
mlmodel.save(args.output)

print('checking converted model')
#onnx.checker.check_model(onnx_model)
'''
# 画像の読み込み
filename = "data/test"
dataset = PoseDataset(
    filename,
    input_transform=transforms.Compose([
        transforms.ToTensor(),
        RandomNoise()]),
    output_transform=Scale(),
    transform=Crop(data_augmentation=False))

img, pose, _, _ = dataset[0]
arr = img.unsqueeze(0)
out = mlmodel.predict({'img__0': img})['out__0']
print("#output coreml result.")

print(out.shape)
print(np.transpose(out))
print(out)
# print(out[:, 0:1, 0:1])
print(np.mean(out))
'''
