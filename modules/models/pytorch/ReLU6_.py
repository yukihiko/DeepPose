# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU6_(nn.Module):
    def __init__(self):
        super(ReLU6_, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.clamp(self.model(x), max=6)
