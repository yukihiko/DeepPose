# -*- coding: utf-8 -*-
""" VGG19Net implementation. """

import torch.nn as nn
import torch.nn.functional as F


class VGG19Net(nn.Module):
    """ VGG19Net AlexNet :
    Args:
        Nj (int): Size of joints.
    """

    def __init__(self, Nj):
        super(VGG19Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(256, 128, 3, stride=1, padding=1)

        self.fc6 = nn.Linear(128*28*28, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, Nj*2)
        self.Nj = Nj

    def forward(self, x):
        # layer1
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2, stride=2)    # (227 + 2*0 - 2 ) / 2 + 1= 113.5
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(h, 2, stride=2)    # (113 + 2*0 - 2 ) / 2 + 1= 56.5
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pool2d(h, 2, stride=2)    # (56 + 2*0 - 2 ) / 2 + 1= 28
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = h.view(-1, 128*28*28)
        # layer6-8
        h = F.dropout(F.relu(self.fc6(h)), training=self.training)
        h = F.dropout(F.relu(self.fc7(h)), training=self.training)
        h = self.fc8(h)
        return h.view(-1, self.Nj, 2)
