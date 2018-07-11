# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class MobileNet_(nn.Module):
    def __init__(self):
        super(MobileNet_, self).__init__()
        self.col = 14
        self.Nj = 14

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1),
            conv_dw(1024, 1024, 1),
        )
        self.heatmap = nn.Conv2d(1024, self.Nj, 1)
        self.offset = nn.Conv2d(1024, 2, 1)

    def forward(self, x):
        x = self.model(x)
        #x = x.view(-1, 1024)
        h = self.heatmap(x)
        h = F.sigmoid(h)
        o = self.offset(x)

        reshaped = h.view(-1, self.Nj, self.col*self.col)
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col
        xc =  xCoords.cpu().data[0].numpy()
        yc =  yCoords.cpu().data[0].numpy()
        op = o.cpu().data.numpy()
        #px = (op[:, 0, xc, yc] + xc * self.col)/227.0
        #py = (op[:, 1, xc, yc] + yc * self.col)/227.0
        px = xc * self.col/227.0
        py = yc * self.col/227.0
        
        res = np.hstack([px, py])
        p=Variable(torch.from_numpy(res), requires_grad=True).float()
        return p.cuda()
