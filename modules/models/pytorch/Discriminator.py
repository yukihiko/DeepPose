import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(14, 96, 3, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 256, 3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(256*2*2, 1)

        self.conv1 = nn.Conv2d(14, 96, 3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, 3, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3)
        self.conv4 = nn.Conv2d(384, 384, 3)
        self.conv5 = nn.Conv2d(384, 256, 3)
        self.bn1 = nn.BatchNorm2d(96)
        
    def forward(self, x):

        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        #x = self.conv5(x)
        x = self.conv(x)

        x = x.view(-1, 256*2*2)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return x.squeeze()
