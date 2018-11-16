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
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNetV2, MobileNet_, MobileNet_2, MobileNet_3, MobileNet_4, MobileNet___, MnasNet, MnasNet_,MnasNet56_,MnasNet16_,MobileNet16_,MobileNet14_
#from coremltools.converters.keras import convert
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale
from torchvision import transforms
from PIL import Image

'''
再帰的に呼び出してpruningを行う
'''
def pruning(module, threshold):
    print(module)

    if module != None:
        if isinstance(module, torch.nn.Sequential):
            for child in module.children():
                pruning(child, threshold)

        if isinstance(module, torch.nn.Conv2d):
            old_weights = module.weight.data.cpu().numpy()
            new_weights = (np.absolute(old_weights) > threshold) * old_weights
            module.weight.data = torch.from_numpy(new_weights)

        if isinstance(module, torch.nn.BatchNorm2d):
            #module.track_running_stats = False
            print(module.weight)
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

            '''
            module.weight is gamma = 1
            running_mean is mean = 0
            running_var is variance = 1
            bias is beta
            '''
            #module.weight.data = torch.from_numpy(np.ones_like(module.weight.data)) 
            #module.running_mean.data = torch.from_numpy(np.zeros_like(module.running_mean.data)) 
            #module.running_var.data = torch.from_numpy(np.ones_like(module.running_var.data)) 


print('ArgumentParser')
parser = argparse.ArgumentParser(description='Convert PyTorch model to CoreML')
parser.add_argument('--input', '-i', required=True, type=str)
parser.add_argument('--NN', '-n', required=True, type=str)
parser.add_argument('--NJ', required=True, type=int)
parser.add_argument('--Col', required=True, type=int)
parser.add_argument('--image_size', required=True, type=int)
parser.add_argument('--is_checkpoint', required=True, type=int)

args = parser.parse_args()

model = eval(args.NN)()

cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

print('load model')

if args.is_checkpoint == 1:
    checkpoint = torch.load(args.input)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    optimizer_state_dict = checkpoint['optimizer']
else:
    model.load_state_dict(torch.load(args.input))

#model = model.cpu()
model.eval()

# export to ONNF
img_path = "im07276.jpg"
img = Image.open(img_path)
img = img.resize((args.image_size, args.image_size))
arr = np.asarray(img, dtype=np.float32)[np.newaxis, :, :, :]
dummy_input = Variable(torch.from_numpy(arr.transpose(0, 3, 1, 2)/255.))
#dummy_input = Variable(torch.randn(1, 3, args.image_size, args.image_size))
################

if args.NN == "MobileNet14_":
    output = model.forward(dummy_input)
    heatmap = output[:, 0:16, :, :]
    offset = output[:, 16:48, :, :] 
elif  args.NN == "MobileNet_":
    offset, heatmap = model.forward(dummy_input)
elif  args.NN == "MobileNet_3":
    offset, heatmap = model.forward(dummy_input)
elif  args.NN == "MobileNet_4":
    offset, heatmap = model.forward(dummy_input)


print("pytorch heatmap")
for k in range(args.NJ):
    x = -1
    y = -1
    max = -1.0
    print("{}: ".format(k,))
    for i in range(args.Col):
        str = ""
        for j in range(args.Col):
            v =  heatmap[0, k, i, j]
            str = str + ",{:.3f}".format(v) 
        print(str)

print("pytorch offset")
for k in range(args.NJ):
    x = -1
    y = -1
    max = -1.0
    print("{}: ".format(k,))
    for i in range(args.Col):
        str = ""
        for j in range(args.Col):
            v =  offset[0, k, i, j]
            str = str + ",{:.3f}".format(v) 
        print(str)

print("pytorch Index")
for k in range(args.NJ):
    x = -1
    y = -1
    max = -1.0
    for i in range(args.Col):
        str = ""
        for j in range(args.Col):
            v =  heatmap[0, k, i, j]
            if v > 0.5 and v > max:
                x = j
                y = i
                max = v
    print("{}: {}, {}".format(k, x, y))
