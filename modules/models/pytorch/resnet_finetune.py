import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Resnet(nn.Module):

    def __init__(self, num_classes=14):
        self.num_classes = num_classes
        super(Resnet,self).__init__()
        resnet = models.resnet50(num_classes=self.num_classes*2, pretrained=False)
        childlen = resnet.children()
        lst = list(childlen)
        lst = list(childlen)[:-2]
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        # self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        #   self.maxpool = nn.MaxPool2d(kernel_size=7)
        self.fc = nn.Linear(8192, self.num_classes*2)

    def forward(self, x):

        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        # 64 8192
        x = self.fc(x)

        return x.view(-1, self.num_classes, 2)
