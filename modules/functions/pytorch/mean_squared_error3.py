# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import scipy.ndimage.filters as fi

class MeanSquaredError3(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False, Nj=14, col=14):
        super(MeanSquaredError3, self).__init__()
        self.use_visibility = use_visibility
        self.Nj = Nj
        self.col = col
        self.gaussian = 1.0

    def min_max(self, x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return torch.Tensor(result)

    def checkMatrix(self, xi, yi):
        f = False
        if xi >= 0 and xi <= self.col - 1 and yi >= 0 and yi <= self.col - 1:
            f = True

        '''
        if xi < 0:
            xi = 0
        if xi > 13:
            xi = 13
        if yi < 0:
            yi = 0
        if yi > 13:
            yi = 13
        '''
        return xi, yi, f

    def forward(self, *inputs):
        out, t, v = inputs
        h = out[:, 0:14, :, :]
        o = out[:, 14:, :, :]
        
        #最終
        scale = 1./float(self.col)
        reshaped = h.view(-1, self.Nj, self.col*self.col)
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col

        x = Variable(torch.zeros(t.size()).float(), requires_grad=True).cuda()

        s = h.size()
        tt = torch.zeros(s).float()
        ti = t*self.col

        for i in range(s[0]):
            for j in range(self.Nj):
                #if h[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                x[i, j, 0] = (o[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float()) * scale
                x[i, j, 1] = (o[i, j + self.Nj, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float()) * scale

                if int(v[i, j, 0]) == 1:
                    xi, yi, f = self.checkMatrix(int(ti[i, j, 0]), int(ti[i, j, 1]))
                    
                    if f == True:
                        # 正規分布に近似したサンプルを得る
                        # 平均は 100 、標準偏差を 1 
                        tt[i, j, yi, xi]  = 1
                        tt[i, j] = self.min_max(fi.gaussian_filter(tt[i, j], self.gaussian))
                    else:
                        v[i, j, 0] = 0
                        v[i, j, 1] = 0
                        
        tt = Variable(tt).cuda()
        
        diff1 = h - tt
        cnt = 0
        for i in range(s[0]):
            for j in range(self.Nj):
                if int(v[i, j, 0]) == 0:
                    diff1[i, j].data[0] = diff1[i, j].data[0]*0
                else:
                    cnt = cnt + 1
     
        #N1 = (v.sum()/2)

        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1) / cnt
        #return d1

        diff2 = x - t
        diff2 = diff2*v
        N2 = (v.sum()/2)
        diff2 = diff2.view(-1)
        d2 = diff2.dot(diff2)/N2
        return d1 + d2


def mean_squared_error3(x, t, v, use_visibility=False):
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
    return MeanSquaredError3(use_visibility)(x, t, v)
