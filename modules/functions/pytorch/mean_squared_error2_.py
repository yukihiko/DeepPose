# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import scipy.ndimage.filters as fi

class MeanSquaredError2_(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False, Nj=14, col=14):
        super(MeanSquaredError2_, self).__init__()
        self.use_visibility = use_visibility
        self.Nj = Nj
        self.col = col

    def min_max(self, x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return torch.Tensor(result)

    def forward(self, *inputs):
        os, h, op, t, v = inputs

        #最終
        v3 = torch.cat([op[:, :, 2:] , op[:, :, 2:]], dim=2)
        v3 = torch.floor(v3 + 0.5)
        op = op[:, :, 0:2]

        scale = 1./float(self.col)
        #print(v[0,0])
        #print(h[0,0])
        reshaped = h.view(-1, self.Nj, self.col*self.col)
        #print(reshaped[0,0])
        _, argmax = reshaped.max(-1)
        #print(argmax[0, 0])
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col
        #print(h[0,0, yCoords[0, 0], xCoords[0, 0]])

        x = Variable(torch.zeros(t.size()).float(), requires_grad=True).cuda()

        s = h.size()
        tt = torch.zeros(s).float()
        ott = torch.zeros(s).float()
        ti = t*self.col

        for i in range(s[0]):
            for j in range(self.Nj):
                if h[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                    x[i, j, 0] = (os[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float()) * scale
                    x[i, j, 1] = (os[i, j + 14, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float()) * scale

                if int(v[i, j, 0]) == 1:
                    xi = int(ti[i, j, 0])
                    yi = int(ti[i, j, 1])
                    
                    if xi < 0:
                        xi = 0
                    if xi > 13:
                        xi = 13
                    if yi < 0:
                        yi = 0
                    if yi > 13:
                        yi = 13
                    
                    # 正規分布に近似したサンプルを得る
                    # 平均は 100 、標準偏差を 1 
                    tt[i, j, yi, xi]  = 1
                    tt[i, j] = self.min_max(fi.gaussian_filter(tt[i, j], 1))

                if int(v3[i, j, 0]) >= 0.5:
                    oxi = int(op[i, j, 0])
                    oyi = int(op[i, j, 1])
                    
                    if oxi < 0:
                        oxi = 0
                    if oxi > 13:
                        oxi = 13
                    if oyi < 0:
                        oyi = 0
                    if oyi > 13:
                        oyi = 13
                    
                    # 正規分布に近似したサンプルを得る
                    # 平均は 100 、標準偏差を 1 
                    ott[i, j, oyi, oxi]  = 1

        #print(h[0, 1])
        tt = Variable(tt).cuda()
        #print(tt[0, 1])
        #diff1 = h[:, :, yi, xi] - tt[:, :, yi, xi]
        #vv = v[:,:,0]
        #N1 = (vv.sum()/2).data[0]
        #diff1 = diff1*vv
        diff1 = h - tt
        for i in range(s[0]):
            for j in range(self.Nj):
                if int(v[i, j, 0]) == 0:
                    diff1[i, j].data[0] = diff1[i, j].data[0]*0
        N1 = (v.sum()/2).data[0]
        

        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1) / N1

        diff3 = op - t
        diff3 = diff3*v
        diff3 = diff3.view(-1)
        N3 = (v.sum()/2).data[0]
        d3 = diff3.dot(diff3)/N3

        ott = Variable(ott).cuda()
        diff4 = h - ott
        diff4 = diff4.view(-1)
        d4 = diff4.dot(diff4) / N3

        #return d1 + d3 + d4

        diff2 = x - t
        diff2 = diff2*v
        N2 = (v.sum()/2).data[0]
        diff2 = diff2.view(-1)
        d2 = diff2.dot(diff2)/N2
        return d1 + d2
        return d1 + d2 + d3 + d4
        



def mean_squared_error2_(os, h, op, t, v, use_visibility=False):
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
    return MeanSquaredError2_(use_visibility)(os, h, op, t, v)
