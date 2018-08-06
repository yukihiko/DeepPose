# -*- coding: utf-8 -*-
""" Train models module. """

from modules.models.pytorch.alex_net import AlexNet
from modules.models.pytorch.VGG19_net import VGG19Net
from modules.models.pytorch.inceptionv3 import Inceptionv3, inception_v3_
from modules.models.pytorch.resnet_finetune import Resnet
from modules.models.pytorch.MobileNet import MobileNet
from modules.models.pytorch.MobileNetV2 import MobileNetV2
from modules.models.pytorch.MobileNet_ import MobileNet_
from modules.models.pytorch.MobileNet_2 import MobileNet_2
from modules.models.pytorch.MobileNet_3 import MobileNet_3
from modules.models.pytorch.Lin_View import Lin_View
from modules.models.pytorch.MobileNet__ import MobileNet__


__all__ = ['AlexNet', 'VGG19Net', 'Inceptionv3', 'Resnet', 'MobileNet', 'MobileNetV2', 'MobileNet_', 'MobileNet_2', 'MobileNet_3', 'Lin_View', 'MobileNet__']
