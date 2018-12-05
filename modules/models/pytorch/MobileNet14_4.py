# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

import sys

class MobileNet14_4(nn.Module):
    def __init__(self):
        super(MobileNet14_4, self).__init__()
        self.col = 14
        self.Nj = 16

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_last(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            )
        '''
        +
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
        '''
        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
        )
        self.model1_1 = conv_dw(256, 256, 1)
        self.model1_2 = conv_dw(256, 512, 2)
        self.model1_3 = conv_dw(512, 512, 1)
        self.model1_4 = conv_dw(512, 512, 1)
        self.model1_5 = conv_dw(512, 512, 1)
        self.model1_6 = conv_dw(512, 512, 1)
        self.model1_7 = conv_dw(512, 512, 1)
        self.model1_8 = conv_dw(512, 1024, 1)
        self.model1_9 = conv_dw(1024, 1024, 1)
        
        #self.heatmap = nn.Conv2d(1024, self.Nj, 1)
        #self.offset = nn.Conv2d(1024, self.Nj*2, 1)
        self.model2 = conv_dw(1024, 1024, 1)
        self.heatmap = conv_last(1024, self.Nj, 1)
        self.offset = conv_last(1024, self.Nj*2, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.model1_1(x) + x
        x12 = self.model1_2(x)
        x13 = self.model1_3(x12) + x12
        x14 = self.model1_4(x13) + x13
        x15 = self.model1_5(x14) + x14
        x16 = self.model1_6(x15) + x15 + x13
        x17 = self.model1_7(x16) + x16 + x12

        x18 = self.model1_8(x17)

        x19 = self.model1_9(x18) + x18
        x20 = self.model2(x19) + x19

        h = self.heatmap(x20)
        h = F.sigmoid(h)
        o = self.offset(x20)

        return o, h
