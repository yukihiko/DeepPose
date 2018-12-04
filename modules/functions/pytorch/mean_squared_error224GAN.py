# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import scipy.ndimage.filters as fi

class MeanSquaredError224GAN(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False, Nj=14, col=14, image_size=224):
        super(MeanSquaredError224GAN, self).__init__()
        self.use_visibility = use_visibility
        self.Nj = Nj
        self.col = col
        self.image_size = image_size
        self.gaussian = 1.0
        self.loss_f = nn.BCEWithLogitsLoss()

    def min_max(self, x, axis=None, maxV=1.0):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return torch.Tensor(result)

    def checkSize(self, xi, yi, size=224):
        f = False
        if xi >= 0 and xi < size and yi >= 0 and yi < size:
            f = True
        return xi, yi, f

    def forward(self, *inputs):
        offset, heatmap, pose, visibility, discriminator, discriminator2 = inputs
        
        s = heatmap.size()
        scale = 1./float(self.col)*self.image_size
        reshaped = heatmap.view(-1, self.Nj, self.col*self.col)
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col
        xx = Variable(torch.zeros(s[0], self.Nj, self.image_size, self.image_size).float(), requires_grad=True).cuda()
        ax = Variable(torch.zeros(pose.size()).float(), requires_grad=True).cuda()

        tt224 = torch.zeros(s[0], self.Nj, self.image_size, self.image_size).float().cuda()
        tt = torch.zeros(s).float().cuda()
        ti = pose*self.col
        ti224 = pose*self.image_size
        v = visibility
        for i in range(s[0]):
            for j in range(self.Nj):
                xc = xCoords[i, j]
                yc = yCoords[i, j]
                ax[i, j, 0] = (offset[i, j, yc, xc] + xc.float()) * scale
                ax[i, j, 1] = (offset[i, j + self.Nj, yc, xc] + yc.float()) * scale
                
                x = (ax[i, j, 0] + 0.5).int()
                y = (ax[i, j, 1] + 0.5).int()
                x, y, f = self.checkSize(x, y)

                if f == True:
                    hm = heatmap[i, j, yc, xc]
                    xx[i, j, y, x] = 1
                    xx[i, j] = Variable(self.min_max(fi.gaussian_filter(xx[i, j].data, self.gaussian)).data, requires_grad=True).cuda() * hm
                
                if int(v[i, j, 0]) == 1:
                    
                    xi, yi, f = self.checkSize((ti[i, j, 0] + 0.5).int(), (ti[i, j, 1] + 0.5).int(), size=self.col)
                    
                    if f == True:
                        # 正規分布に近似したサンプルを得る
                        # 平均は 100 、標準偏差を 1 
                        tt[i, j, yi, xi]  = 1
                        tt[i, j] = self.min_max(fi.gaussian_filter(tt[i, j], 1.0))
                    else:
                        v[i, j, 0] = 0
                    
                    xi224, yi224, f = self.checkSize((ti224[i, j, 0] + 0.5).int(), (ti224[i, j, 1] + 0.5).int())
            
                    if f == True:
                        tt224[i, j, yi224, xi224]  = 1
                        tt224[i, j] = self.min_max(fi.gaussian_filter(tt224[i, j], self.gaussian))
                    
        
        #print(xx[0,0])
        xx_tensor = xx.data

        
        out = discriminator(xx)
        ones = Variable(torch.ones(s[0])).cuda()
        lossf1 = self.loss_f(out, ones) 
        
        out2 = discriminator2(heatmap)
        ones2 = Variable(torch.ones(s[0])).cuda()
        lossf2 = self.loss_f(out2, ones2) 

        
        
        '''
        diff1 = xx - tt224
        cnt = 0
        for i in range(s[0]):
            for j in range(self.Nj):
                if int(v[i, j, 0]) == 0:
                    diff1[i, j] = diff1[i, j]*0
                else:
                    cnt = cnt + 1
        diff1 = diff1.view(-1)
        loss_m1 = diff1.dot(diff1) / cnt
        '''
        diff1 = heatmap - tt
        cnt = 0
        for i in range(s[0]):
            for j in range(self.Nj):
                if int(v[i, j, 0]) == 0:
                    diff1[i, j] = diff1[i, j]*0
                else:
                    cnt = cnt + 1
        diff1 = diff1.view(-1)
        loss_m1 = diff1.dot(diff1) / cnt

        diff2 = (ax - ti224)/224.
        diff2 = diff2*v
        N2 = (v.sum()/2)
        diff2 = diff2.view(-1)
        loss_m2 = diff2.dot(diff2)/N2
        #loss_m = torch.Tensor.sqrt(diff2.dot(diff2))/N2

        #return loss_m1 + loss_m2
        return loss_m1 + loss_m2, lossf1, lossf2, tt, tt224, xx_tensor
        

def mean_squared_error224GAN(o, h, t, v, discriminator, discriminator2, use_visibility=False):
    """ Computes mean squared error over the minibatch.

    Args:
        x (Variable): Variable holding an float32 vector of estimated pose.
        t (Variable): Variable holding an float32 vector of ground truth pose.
        v (Variable): Variable holding an int32 vector of ground truth pose's visibility.
            (0: invisible, 1: visible)
        use_visibility (bool): When it is ``True``,
            the function uses visibility to compute mean squared error.
    Returns:
        Variable: A variable holding a scalar of the mean squared error loss.
    """
    return MeanSquaredError224GAN(use_visibility)(o, h, t, v, discriminator, discriminator2)
