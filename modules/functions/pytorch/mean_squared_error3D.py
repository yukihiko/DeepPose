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

        self.lengs = np.array([[[0,1],[5,6]], [[1,2],[6,7]], [[2,3],[7,8]],[[2,4],[7,9]], [[15,16],[19,20]], [[16,17],[20,21]], [[17,18],[21,22]], [[0,23],[5,23]], [[15,23],[19,23]] ])

        # # 3D Generate gaussian
        self.gaussianM = torch.zeros((7,7,7), dtype=torch.float64)
        gx = np.arange(0, 7, 1, np.float32)
        gy = gx[:, np.newaxis]
        x0 = y0 = z0 = 3
        self.gaussian2DM = np.exp(- ((gx - x0) ** 2 + (gy - y0) ** 2) / 2)
        
        for z in range(7):
            #print(np.exp(- ((gx - x0) ** 2 + (gy - y0) ** 2 + (z - z0) ** 2) / 2))
            self.gaussianM[z, :, :] = torch.Tensor(np.exp(- ((gx - x0) ** 2 + (gy - y0) ** 2 + (z - z0) ** 2) / 2))

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
        o2D, o3D, h, d, t2D, t3D, v = inputs
        
        #最終
        scale = 1./float(self.col)
        reshaped = h.view(-1, self.Nj, self.col*self.col)
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col

        x2D = Variable(torch.zeros(t2D.size()).float(), requires_grad=True).cuda()
        x3D = Variable(torch.zeros(t3D.size()).float(), requires_grad=True).cuda()

        s = h.size()
        tt = torch.zeros(s).float()
        v3D = v.clone()

        for i in range(s[0]):
            for j in range(self.Nj):
                #if h[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                x2D[i, j, 0] = o2D[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float() * scale
                x2D[i, j, 1] = o2D[i, j + self.Nj, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float() * scale
                
                if d[i] <= -990:
                    v3D[i, :, :] = 0
                else:
                    x3D[i, j, 0] = o3D[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float() * scale
                    x3D[i, j, 1] = o3D[i, j + self.Nj, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float() * scale
                    x3D[i, j, 2] = o3D[i, j + self.Nj*2, yCoords[i, j], xCoords[i, j]]
                
                if int(v[i, j, 0]) == 1:
                    mu_x = int(t2D[i, j, 0]*self.col + 0.5)
                    mu_y = int(t2D[i, j, 1]*self.col + 0.5)
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - self.tmp_size), int(mu_y - self.tmp_size)]
                    br = [int(mu_x + self.tmp_size + 1), int(mu_y + self.tmp_size + 1)]
                    if ul[0] >= self.col or ul[1] >= self.col or br[0] <= 0 or br[1] <= 0:
                        # If not, just return the image as is
                        v[i, j, 0] = 0
                        v[i, j, 1] = 0
                        v[i, j, 2] = 0
                        v3D[i, :, :] = 0
                        continue

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.col) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.col) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.col)
                    img_y = max(0, ul[1]), min(br[1], self.col)

                    tt[i, j][img_y[0]:img_y[1], img_x[0]:img_x[1]] = torch.Tensor(self.gaussian2DM[g_y[0]:g_y[1], g_x[0]:g_x[1]]) 

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

        vv = v[:,:,:2]
        diff2 = x2D - t2D
        diff2 = diff2*vv
        N2 = (v.sum()/3)
        diff2 = diff2.view(-1)
        d2 = diff2.dot(diff2)/N2
        #return d1 + d2

        diff3 = x3D - t3D
        diff3 = diff3*v3D
        N3 = (v3D.sum()/3)
        diff3 = diff3.view(-1)
        d3 = diff3.dot(diff3)/N3

        #return d1 + d2 + d3
        

        lengV = 0
        ll = 0

        for i in range(self.lengs.shape[0]): 
            idx00 = self.lengs[i][0][0]
            idx01 = self.lengs[i][0][1]
            idx10 = self.lengs[i][1][0]
            idx11 = self.lengs[i][1][1]
            vv = v[:, idx00] * v[: , idx01] * v[:, idx10] * v[:, idx11]
            le0 = (x3D[:, idx00] - x3D[:, idx01])*vv
            le1 = (x3D[:, idx10] - x3D[:, idx11])*vv
            le0 = le0.view(-1)
            le1 = le1.view(-1)
            le0s = torch.sqrt(le0.dot(le0))
            le1s = torch.sqrt(le1.dot(le1))
            dleng = le0s - le1s
            lengV = lengV + (vv.sum()/3)

            #ll = ll + torch.sqrt(dleng*dleng)
            ll = ll + dleng*dleng
            
            '''
            lt0 = (t3D[:, idx00] - t3D[:, idx01])*vv
            lt1 = (t3D[:, idx10] - t3D[:, idx11])*vv
            lt0 = lt0.view(-1)
            lt1 = lt1.view(-1)
            tleng0 = (le0s - torch.sqrt(lt0.dot(lt0)))
            tleng1 = (le1s - torch.sqrt(lt1.dot(lt1)))
            ll = ll + torch.sqrt(dleng*dleng) + torch.sqrt(tleng0*tleng0) + torch.sqrt(tleng1*tleng1)
            '''

        #seiyaku
        
        d4 = ll/lengV

        return d1 + d2 + d3 + d4

def mean_squared_error3D(o, o3D, h, d, t, t3D, v, use_visibility=False, col=14):
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
    return MeanSquaredError3D(use_visibility, col=col)(o, o3D, h, d, t, t3D, v)
