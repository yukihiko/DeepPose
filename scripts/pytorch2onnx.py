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
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNetV2, MobileNet_, MobileNet_2, MobileNet_3
#from coremltools.converters.keras import convert
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale
from torchvision import transforms

print('ArgumentParser')
parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
parser.add_argument('--input', '-i', required=True, type=str)
parser.add_argument('--output', '-o', required=True, type=str)
parser.add_argument('--NN', '-n', required=True, type=str)
args = parser.parse_args()

print('Set up model')
if args.NN == "MobileNet":
    model = MobileNet( )
elif args.NN == "MobileNet_3":
    model = MobileNet_3( )

model.load_state_dict(torch.load(args.input))
model.eval()
'''
pytorch_model = args.input
keras_output = 'model.hdf5'
# export to ONNF
dummy_input = Variable(torch.randn(1, 3, 224, 224))

print('converting to ONNX')
torch.onnx.export(model, dummy_input, args.output)

print('checking converted model')
onnx_model = onnx.load(args.output)

k_model = load_model(keras_output)
coreml_model = convert(k_model)
coreml_model.save('modle.mlmodel')
'''

'''
input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
input_var = Variable(torch.FloatTensor(input_np))
k_model = pytorch_to_keras(model, input_var, [(3, 224, 224,)], verbose=True)  
coreml_model = convert(k_model)
coreml_model.save('modle.mlmodel')
'''

# export to ONNF
dummy_input = Variable(torch.randn(1, 3, 224, 224))

print('converting to ONNX')
torch.onnx.export(model, dummy_input, args.output)
onnx_model = onnx.load(args.output)

print('converting coreml model')
mlmodel = convert(
        onnx_model, 
        image_input_names='0')
mlmodel.save('coreml_model.mlmodel')
'''
mlmodel = convert(onnx_model, 
    image_input_names=['image'], 
    image_output_names=['output'],
    )
mlmodel.save('coreml_model.mlmodel')
'''

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
