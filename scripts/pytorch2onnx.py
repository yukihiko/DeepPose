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
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNetV2, MobileNet_, MobileNet_2, MobileNet_3, MobileNet___, MnasNet, MnasNet_,MnasNet56_,MnasNet16_,MobileNet16_
#from coremltools.converters.keras import convert
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale
from torchvision import transforms

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
elif args.NN == "MnasNet_":
    model = MnasNet_( )
elif args.NN == "MnasNet56_":
    model = MnasNet56_( )
elif args.NN == "MnasNet16_":
    model = MnasNet16_( )
elif args.NN == "MobileNet16_":
    model = MobileNet16_( )

cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

model.load_state_dict(torch.load(args.input))
'''    
checkpoint = torch.load(args.input)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
optimizer_state_dict = checkpoint['optimizer']
'''     
'''
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
'''
#model = model.cpu()
model.eval()

# export to ONNF
dummy_input = Variable(torch.randn(1, 3, 224, 224))
################
_ = model(dummy_input)

all_weights = []
for p in model.parameters():
    if len(p.data.size()) != 1:
        all_weights += list(p.cpu().data.abs().numpy().flatten())
threshold = np.percentile(np.array(all_weights), 80.)

pruning(model.model, threshold)
'''
for child in model.children():
    for param in child.parameters():
        param.reguired_grand = False
'''
model.eval()
#model.cuda()

##################
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

