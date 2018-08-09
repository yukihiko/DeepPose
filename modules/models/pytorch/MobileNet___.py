# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./")
from modules.models.pytorch.Lin_View import Lin_View

class MobileNet___(nn.Module):
    def __init__(self):
        super(MobileNet___, self).__init__()
        self.col = 14
        self.Nj = 14

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
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

        self.model1 = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
        )
        self.model2 = nn.Sequential(
            conv_dw(512, 512, 1),
        )
        self.model3 = nn.Sequential(
            conv_dw(512, 512, 1),
        )
        self.model4 = nn.Sequential(
            conv_dw(512, 512, 1),
        )
        self.model5 = nn.Sequential(
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1),
            conv_dw(1024, 1024, 1),
        )
        self.heatmap = nn.Sequential(
            nn.Conv2d(1024, self.Nj + 4, 1),
        )

        self.offset = nn.Sequential(
            nn.Conv2d(1024, self.Nj*2, 1),
        )


    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        #x = x.view(-1, 1024)
        h = self.heatmap(x)
        h = F.sigmoid(h)

        os = self.offset(x)
        #print(h[0, 1])

        return os, h
