# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class Lin_View(nn.Module):
    def __init__(self, view_size):
        super(Lin_View, self).__init__()
        self.view_size = view_size
   
    def forward(self, x):
        return x.view(-1, self.view_size) 
