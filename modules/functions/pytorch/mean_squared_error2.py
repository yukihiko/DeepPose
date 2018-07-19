# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

class MeanSquaredError2(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False, Nj=14, col=14):
        super(MeanSquaredError2, self).__init__()
        self.use_visibility = use_visibility
        self.Nj = Nj
        self.col = col

    def forward(self, *inputs):
        o, h, t, v = inputs

        s = h.size()
        tt = torch.zeros(s).float()
        pp = torch.zeros(o.size()).float()
        ti = t*self.col
        tpos = t*224
        one = np.ones(self.col).reshape(-1,1) # 縦ベクトルに変換
        arg = (torch.arange(self.col) * self.col)
        for i in range(s[0]):
            for j in range(self.Nj):
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
                    tt[ i, j, xi, yi]  = 1

                x = one*(tpos[i, j, 0].cpu().data  - arg)
                y = one*(tpos[i, j, 1].cpu().data  - arg)
                pp[ i, j, :, :]  = x
                pp[ i, j + self.Nj, :, :]  = y.t()

        '''
        reshaped = h.view(-1, self.Nj, self.col*self.col)
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col
        xc =  xCoords.cpu().data.numpy()
        yc =  yCoords.cpu().data.numpy()
        point = Variable(torch.zeros(t.size()).float(), requires_grad=True).float().cuda()
        op = o.cpu().data.numpy()
        for i in range(s[0]):
            for j in range(self.Nj):
                point[i, j, 0] = (op[i, j, xc[i, j], yc[i, j]] + xc[i, j] * self.col)/224.0
                point[i, j, 1] = (op[i, j + self.Nj, xc[i, j], yc[i, j]] + yc[i, j] * self.col)/224.0
        '''
        #xxx = o[:, 0, xc, yc]
        #xc =  xCoords.cpu().data[0].numpy()
        #yc =  yCoords.cpu().data[0].numpy()
        #x = op[:, :, xc, yc]
        #px = (op[:, 0, xc, yc] + xc * self.col)/224.0
        #py = (op[:, 1, xc, yc] + yc * self.col)/224.0
        #res = np.hstack([px, py])
        #p=Variable(torch.from_numpy(res), requires_grad=True).float()
        #return p.cuda()

        #px = xCoords.float() * self.col/224.0
        #py = yCoords.float() * self.col/224.0
        #res = torch.cat([px, py], dim=1).float()
        #x = res.view(-1, self.Nj, 2)

        #px = xc * self.col/224.0
        #py = yc * self.col/224.0
        #p = np.hstack([px, py])
        #x=Variable(torch.from_numpy(p), requires_grad=True).float().cuda().view(-1, self.Nj, 2)
        
        #torch.masked_select

        #heatmapのみの学習の時
        diff1 = h - Variable(tt).cuda()
        diff2 = o - Variable(pp).cuda()
        #diff2 = point - t
        #if self.use_visibility:
        N = (v.sum()/2).data[0]
        #diff1 = diff1*v
        #diff2 = diff2*v
        #else:
        #    N = diff.numel()/2
        diff1 = diff1.view(-1)
        diff2 = diff2.view(-1)
        d1 = diff1.dot(diff1)
        d2 = diff2.dot(diff2)
        #return (d1)/N
        return (d1 + d2)/N


def mean_squared_error2(o, h, t, v, use_visibility=False):
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
    return MeanSquaredError2(use_visibility)(o, h, t, v)
