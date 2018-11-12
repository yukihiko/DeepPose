# -*- coding: utf-8 -*-

import argparse
from torch.autograd import Variable
import torch.onnx
import torchvision.models as models
import onnx
import numpy as np
#from keras.models import load_model
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import sys
sys.path.append("./")
from onnx_coreml.converter import convert
#from pytorch2keras.converter import pytorch_to_keras
from modules.errors import FileNotFoundError, GPUNotFoundError, UnknownOptimizationMethodError, NotSupportedError
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNetV2, MobileNet_, MobileNet_2, MobileNet_3, MobileNet___, MnasNet, MnasNet_,MnasNet56_,MnasNet16_,MobileNet16_,MobileNet14_
#from coremltools.converters.keras import convert
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale
from torchvision import transforms
from PIL import Image
from onnx_tf.backend import prepare


print('ArgumentParser')
parser = argparse.ArgumentParser(description='Convert PyTorch model to CoreML')
parser.add_argument('--input', '-i', required=True, type=str)
parser.add_argument('--NJ', required=True, type=int)
parser.add_argument('--Col', required=True, type=int)
parser.add_argument('--image_size', required=True, type=int)

args = parser.parse_args()

print('load model')

# export to ONNF
img_path = "im07276.jpg"
img = Image.open(img_path).convert('RGB')
img = img.resize((args.image_size, args.image_size))
arr = np.asarray(img, dtype=np.float32)[np.newaxis, :, :, :]
dummy_input = Variable(torch.from_numpy(arr.transpose(0, 3, 1, 2)/255.))

onnx_model = onnx.load(args.input)

# run the loaded model at Tensorflow
output = prepare(onnx_model).run(dummy_input)
if len(output) == 2:
    out = np.array(output[1]).squeeze()
else:
    out = np.array(output).squeeze()

print("Onnx Index")
for k in range(args.NJ):
    x = -1
    y = -1
    max = -1.0
    for i in range(args.Col):
        str = ""
        for j in range(args.Col):
            v =  out[k, i, j]
            if v > 0.5 and v > max:
                x = j
                y = i
                max = v
    print("{}: {}, {}".format(k, x, y))
