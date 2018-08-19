# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import scipy.ndimage.filters as fi

class MeanSquaredError2(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False, Nj=14, col=14):
        super(MeanSquaredError2, self).__init__()
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
        if xi < 0:
            xi = 0
        if xi > 13:
            xi = 13
        if yi < 0:
            yi = 0
        if yi > 13:
            yi = 13
        return xi, yi

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

        for i in range(s[0]):
            for j in range(self.Nj):
                if h[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                    x[i, j, 0] = (o[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float()) * scale
                    x[i, j, 1] = (o[i, j + 14, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float()) * scale

                if int(v[i, j, 0]) == 1:
                    xi, yi = self.checkMatrix(int(ti[i, j, 0]), int(ti[i, j, 1]))
                    
                    # 正規分布に近似したサンプルを得る
                    # 平均は 100 、標準偏差を 1 
                    tt[i, j, yi, xi]  = 1
                    tt[i, j] = self.min_max(fi.gaussian_filter(tt[i, j], self.gaussian))

        tt = Variable(tt).cuda()

        '''
        _, tt_argmax = tt.view(-1, self.Nj, self.col*self.col).max(-1)
        tt_yCoords = tt_argmax/self.col
        tt_xCoords = tt_argmax - tt_yCoords*self.col

        diff1 = h[:, :, tt_yCoords, tt_xCoords] - tt[:, :, tt_yCoords, tt_xCoords]
        vv = v[:,:,0]
        N1 = (vv.sum()/2).data[0]
        diff1 = diff1*vv
        diff3 = h[:, :, yCoords, xCoords] - tt[:, :, yCoords, xCoords]
        diff3 = diff3*vv
        diff3 = diff3.view(-1)
        d3 = diff3.dot(diff3) / N1
        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1) / N1
        return (d1 + d3) / 2.0
        '''
        
        diff1 = h - tt
        for i in range(s[0]):
            for j in range(self.Nj):
                if int(v[i, j, 0]) == 0:
                    diff1[i, j].data[0] = diff1[i, j].data[0]*0
        
        N1 = (v.sum()/2)

        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1) / N1
        #return d1

        diff2 = x - t
        diff2 = diff2*v
        N2 = (v.sum()/2)
        diff2 = diff2.view(-1)
        d2 = diff2.dot(diff2)/N2
        return d1 + d2
        
        '''
        #最終
        scale = 1./float(self.col)
        reshaped = h.view(-1, self.Nj, self.col*self.col)
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col

        x = Variable(torch.zeros(t.size()).float(), requires_grad=True).cuda()

        s = h.size()
        for i in range(s[0]):
            for j in range(self.Nj):
                #if h[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                x[i, j, 0] = (o[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float()) * scale
                x[i, j, 1] = (o[i, j + 14, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float()) * scale

        diff = x - t
        if self.use_visibility:
            N = (v.sum()/2).data[0]
            diff = diff*v
        else:
            N = diff.numel()/2
        diff = diff.view(-1)
        return diff.dot(diff)/N
        '''
        
        '''
        #heatmapのみの学習の時
        s = h.size()
        tt = torch.zeros(s).float()
        ti = t*self.col
    
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
                    
                    # 正規分布に近似したサンプルを得る
                    # 平均は 100 、標準偏差を 1 
                    tt[i, j, yi, xi]  = 1
                    tt[i, j] = self.min_max(fi.gaussian_filter(tt[i, j], 0.3))
        #print(h[0, 1])
        tt = Variable(tt).cuda()
        #print(tt[0, 1])
        diff1 = h - tt
        for i in range(s[0]):
            for j in range(self.Nj):
                if int(v[i, j, 0]) == 0:
                    diff1[i, j].data[0] = diff1[i, j].data[0]*0
 
        N = (v.sum()/2).data[0]

        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1)
        return (d1)/N
        '''
        '''
        #heatmapのみの学習の時第２弾
        s = h.size()
        tt = torch.zeros(t.size()).float()
        ti = t*self.col
        reshaped = h.view(-1, self.Nj, self.col*self.col)
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col

        x = Variable(torch.zeros(t.size()).float(), requires_grad=True).cuda()


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
                    
                    # 正規分布に近似したサンプルを得る
                    # 平均は 100 、標準偏差を 1 
                    tt[i, j, 0]  = xi
                    tt[i, j, 1]  = yi
                
                #if h[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                x[i, j, 0] = xCoords[i, j].float()
                x[i, j, 1] = yCoords[i, j].float()

        #print(h[0, 1])
        tt = Variable(tt).cuda()
        #print(tt[0, 1])
        diff = x - tt
        if self.use_visibility:
            N = (v.sum()/2).data[0]
            diff = diff*v
        else:
            N = diff.numel()/2
        diff = diff.view(-1)
        return diff.dot(diff)/N
        '''

        '''
        s = h.size()
        tt = torch.zeros(s).float()
        ti = t*self.col
        #zeros = Variable(torch.zeros([14, 14]).float(), requires_grad=True).cuda()
        #zeros = Variable(torch.zeros([14, 14]).float()).cuda()
        #ones = Variable(torch.ones(s).float()).cuda()
        #pp = torch.zeros(o.size()).float()
        #tpos = t*224
        #one = np.ones(self.col).reshape(-1,1) # 縦ベクトルに変換
        #arg = (torch.arange(self.col) * self.col)
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

                    # 正規分布に近似したサンプルを得る
                    # 平均は 100 、標準偏差を 1 
                    tt[i, j, yi, xi]  = 1
                    tt[i, j] = self.min_max(fi.gaussian_filter(tt[i, j], 1))

                #x = one*(tpos[i, j, 0].cpu().data  - arg)
                #y = one*(tpos[i, j, 1].cpu().data  - arg)
                #pp[ i, j, :, :]  = x
                #pp[ i, j + self.Nj, :, :]  = y.t()
        
        #print(h[0, 0])
        '''

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


        #print(h[0, 1])
        tt = Variable(tt).cuda()
        #print(tt[0, 1])
        diff1 = h - tt
        for i in range(s[0]):
            for j in range(self.Nj):
                if int(v[i, j, 0]) == 0:
                    diff1[i, j].data[0] = diff1[i, j].data[0]*0
 
        #print(diff1[0, 1])
        #diff2 = o - Variable(pp).cuda()
        #diff2 = point - t

        N = (v.sum()/2).data[0]
        #    N = diff.numel()/2

        diff1 = diff1.view(-1)
        #diff2 = diff2.view(-1)
        d1 = diff1.dot(diff1)
        #d2 = diff2.dot(diff2)
        return (d1)/N
        #return (d1 + d2)/N
        '''


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
