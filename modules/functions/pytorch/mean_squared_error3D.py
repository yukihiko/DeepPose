# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import scipy.ndimage.filters as fi

class MeanSquaredError3D(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False, Nj=24, col=14):
        super(MeanSquaredError3D, self).__init__()
        self.use_visibility = use_visibility
        self.Nj = Nj
        self.col = col
        self.tmp_size = 3.0
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

        return xi, yi, f

    def forward(self, *inputs):
        o, h, t, v = inputs
        
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

        feat_stride = 224. / float(self.col)

        for i in range(s[0]):
            for j in range(self.Nj):
                #if h[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                x[i, j, 0] = o[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float() * scale
                x[i, j, 1] = o[i, j + self.Nj, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float() * scale
                x[i, j, 2] = o[i, j + self.Nj*2, yCoords[i, j], xCoords[i, j]]

                if int(v[i, j, 0]) == 1:
                    mu_x = int(t[i, j, 0]*self.col + 0.5)
                    mu_y = int(t[i, j, 1]*self.col + 0.5)
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - self.tmp_size), int(mu_y - self.tmp_size)]
                    br = [int(mu_x + self.tmp_size + 1), int(mu_y + self.tmp_size + 1)]
                    if ul[0] >= self.col or ul[1] >= self.col or br[0] < 0 or br[1] < 0:
                        # If not, just return the image as is
                        v[i, j, 0] = 0
                        v[i, j, 1] = 0
                        v[i, j, 2] = 0
                        continue

                    # # Generate gaussian
                    gsize = 2 * self.tmp_size + 1
                    gx = np.arange(0, gsize, 1, np.float32)
                    gy = gx[:, np.newaxis]
                    x0 = y0 = gsize // 2
                    # The gaussian is not normalized, we want the center value to equal 1
                    #g = np.exp(- ((gx - x0) ** 2 + (gy - y0) ** 2) / (2 * 1. ** 2))
                    g = np.exp(- ((gx - x0) ** 2 + (gy - y0) ** 2) / 2)

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.col) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.col) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.col)
                    img_y = max(0, ul[1]), min(br[1], self.col)

                    tt[i, j][img_y[0]:img_y[1], img_x[0]:img_x[1]] = torch.Tensor(g[g_y[0]:g_y[1], g_x[0]:g_x[1]]) 
                        
                    '''
                    xi, yi, f = self.checkMatrix(int(ti[i, j, 0]), int(ti[i, j, 1]))
                    
                    if f == True:
                        # 正規分布に近似したサンプルを得る
                        # 平均は 100 、標準偏差を 1 
                        tt[i, j, yi, xi]  = 1
                        tt[i, j] = self.min_max(fi.gaussian_filter(tt[i, j], self.gaussian))
                    else:
                        v[i, j, 0] = 0
                        v[i, j, 1] = 0
                        v[i, j, 2] = 0
                    '''
        tt = Variable(tt).cuda()

        diff1 = h - tt
        cnt = 0
        for i in range(s[0]):
            for j in range(self.Nj):
                if int(v[i, j, 0]) == 0:
                    diff1[i, j] = diff1[i, j]*0
                else:
                    cnt = cnt + 1
        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1) / cnt
        #return d1

        diff2 = x - t
        diff2 = diff2*v
        N2 = (v.sum()/3)
        diff2 = diff2.view(-1)
        d2 = diff2.dot(diff2)/N2

        return d1 + d2
        


def mean_squared_error3D(o, h, t, v, use_visibility=False, col=14):
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
    return MeanSquaredError3D(use_visibility, col=col)(o, h, t, v)
