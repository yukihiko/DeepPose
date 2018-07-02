# -*- coding: utf-8 -*-
""" Train models module. """

from modules.models.pytorch.alex_net import AlexNet
from modules.models.pytorch.VGG19_net import VGG19Net
from modules.models.pytorch.inceptionv3 import Inceptionv3, inception_v3_
from modules.models.pytorch.resnet_finetune import Resnet


__all__ = ['AlexNet', 'VGG19Net', 'Inceptionv3', 'Resnet']
