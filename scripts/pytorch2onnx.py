# -*- coding: utf-8 -*-

import argparse
print('Start')

from torch.autograd import Variable
import torch.onnx
import torchvision.models as models
import onnx

import sys

sys.path.append("./")
from modules.errors import FileNotFoundError, GPUNotFoundError, UnknownOptimizationMethodError, NotSupportedError
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNetV2, MobileNet_
from onnx_coreml.converter import convert

print('ArgumentParser')
parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
parser.add_argument('--input', '-i', required=True, type=str)
parser.add_argument('--output', '-o', required=True, type=str)
args = parser.parse_args()

print('Set up model')
model = MobileNet( )
model.load_state_dict(torch.load(args.input))

# export to ONNF
dummy_input = Variable(torch.randn(1, 3, 224, 224))

print('converting to ONNX')
torch.onnx.export(model, dummy_input, args.output)

print('checking converted model')
onnx_model = onnx.load(args.output)
mlmodel = convert(onnx_model, image_input_names=['image'], image_output_names=['output'])
#mlmodel = convert(onnx_model, image_input_names=['0'], image_output_names=['186'])
mlmodel.save('coreml_model.mlmodel')

onnx.checker.check_model(onnx_model)
