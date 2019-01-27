# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import scipy.ndimage.filters as fi

class MeanSquaredError3D2(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False, Nj=24, col=14):
        super(MeanSquaredError3D2, self).__init__()
        self.use_visibility = use_visibility
        self.Nj = Nj
        self.col = col
        self.tmp_size = 3.0
        self.gaussian = 1.0

        self.lengs = np.array([[[0,1],[5,6]], [[1,2],[6,7]], [[2,3],[7,8]],[[2,4],[7,9]], [[15,16],[19,20]], [[16,17],[20,21]], [[17,18],[21,22]], [[0,23],[5,23]], [[15,23],[19,23]], [[0,1],[0,5]], [[6,7],[15,19]] ])

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
        o, h, o3D, h3D, d, t, t3D, v, path = inputs
        
        #最終
        scale = 1./float(self.col)
        
        # 2D
        reshaped = h.view(-1, self.Nj, self.col*self.col)
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col
        x = Variable(torch.zeros(t.size()).float(), requires_grad=True).cuda()
        s = h.size()
        tt = torch.zeros(s).float()

        # 3D
        x3D = Variable(torch.zeros(t3D.size()).float(), requires_grad=True).cuda()
        s3D = h3D.size()
        tt3D = torch.zeros(s3D).float()

        for i in range(s3D[0]):
            for j in range(self.Nj):

                # 2D判定
                #if h[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                x[i, j, 0] = o[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float() * scale
                x[i, j, 1] = o[i, j + self.Nj, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float() * scale

                if int(v[i, j, 0]) == 1:
                    mu_x = int(t[i, j, 0]*self.col + 0.5)
                    mu_y = int(t[i, j, 1]*self.col + 0.5)
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - self.tmp_size), int(mu_y - self.tmp_size)]
                    br = [int(mu_x + self.tmp_size + 1), int(mu_y + self.tmp_size + 1)]
                    if ul[0] >= self.col or ul[1] >= self.col or br[0] <= 0 or br[1] <= 0:
                        # If not, just return the image as is
                        v[i, j, 0] = 0
                        v[i, j, 1] = 0
                        v[i, j, 2] = 0
                        continue

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.col) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.col) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.col)
                    img_y = max(0, ul[1]), min(br[1], self.col)

                    tt[i, j][img_y[0]:img_y[1], img_x[0]:img_x[1]] = torch.Tensor(self.gaussian2DM[g_y[0]:g_y[1], g_x[0]:g_x[1]]) 

                if d <= -990:
                    continue
                
                # 3D判定
                jj = j * self.col
    
                hp = h3D[i, jj:jj+self.col].squeeze()
                reshaped = hp.view(self.col*self.col*self.col)
                _, argmax = reshaped.max(-1)
                qrt = self.col*self.col
                zCoords3D = argmax/qrt
                yCoords3D = (argmax - zCoords3D*qrt)/self.col
                xCoords3D = argmax - (zCoords3D*qrt + yCoords3D*self.col)

                #if h[i, jj + zCoords, yCoords, xCoords] > 0.3:
                x3D[i, j, 0] = o3D[i, jj + zCoords3D, yCoords3D, xCoords3D] + xCoords3D.float() * scale
                x3D[i, j, 1] = o3D[i, jj + self.Nj*self.col + zCoords3D, yCoords3D, xCoords3D] + yCoords3D.float() * scale
                x3D[i, j, 2] = o3D[i, jj + self.Nj*self.col*2 + zCoords3D, yCoords3D, xCoords3D] + (zCoords3D - 7).float() * scale
                if  x3D[i,j, 0] == float('inf') or x3D[i,j, 0] == float('-inf')  :
                    x3D[i, j, 0] = 0
                if  x3D[i,j, 1] == float('inf') or x3D[i,j, 1] == float('-inf')  :
                    x3D[i, j, 1] = 0
                if  x3D[i,j, 2] == float('inf') or x3D[i,j, 2] == float('-inf')  :
                    x3D[i, j, 2] = 0

                if  t3D[i,j, 0] == float('inf') or t3D[i,j, 0] == float('-inf')  :
                    v[i, j] = 0
                    tt3D[i, j] = 0
                    continue

                if int(v[i, j, 0]) == 1 :
                    if  t3D[i,j, 2] == float('inf') or t3D[i,j, 2] == float('-inf')  :
                        t3D[i,j, 2] = 0.0
                    mu_x = int(t3D[i, j, 0]*self.col + 0.5)
                    mu_y = int(t3D[i, j, 1]*self.col + 0.5)
                    mu_z = int(t3D[i, j, 2]*self.col + 7 + 0.5)
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - self.tmp_size), int(mu_y - self.tmp_size), int(mu_z - self.tmp_size)]
                    br = [int(mu_x + self.tmp_size + 1), int(mu_y + self.tmp_size + 1), int(mu_z + self.tmp_size + 1)]
                    if ul[0] >= self.col or ul[1] >= self.col or ul[2] >= self.col or br[0] < 0 or br[1] < 0 or br[2] < 0:
                        # If not, just return the image as is
                        v[i, j, 0] = 0
                        v[i, j, 1] = 0
                        v[i, j, 2] = 0
                        continue

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.col) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.col) - ul[1]
                    g_z = max(0, -ul[2]), min(br[2], self.col) - ul[2]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.col)
                    img_y = max(0, ul[1]), min(br[1], self.col)
                    img_z = max(0, ul[2]), min(br[2], self.col)

                    #for z in range(7):
                    #    tt[i, j + mu_z + z - 3][img_y[0]:img_y[1], img_x[0]:img_x[1]] = self.gaussianM[z, :, :] 
                    tt3D[i, jj + img_z[0]:jj + img_z[1], img_y[0]:img_y[1], img_x[0]:img_x[1]] = self.gaussianM[g_z[0]:g_z[1], g_y[0]:g_y[1], g_x[0]:g_x[1]] 
                    #for z in range(14):
                    #    print(tt[i, j + z])
                

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
        d1 = torch.sqrt(diff1.dot(diff1)) / cnt

        vv = v[:,:,:2]
        diff2 = x - t
        diff2 = diff2*vv
        N2 = (v.sum()/3)
        diff2 = diff2.view(-1)
        d2 = torch.sqrt(diff2.dot(diff2))/N2

        if d <= -990:
            return d1 + d2

        #3D

        tt3D = Variable(tt3D).cuda()

        diff3 = h3D - tt3D
        cnt = 0
        for i in range(s3D[0]):
            for j in range(self.Nj):
                if int(v[i, j, 0]) == 0:
                    jj = j * self.col
                    diff3[i, jj:jj+self.col] = diff3[i, jj:jj+self.col]*0
                else:
                    cnt = cnt + 1
        diff3 = diff3.view(-1)
        d3 = torch.sqrt(diff3.dot(diff3)) / cnt

        diff4 = x3D - t3D
        diff4 = diff4*v
        N4 = (v.sum()/3)
        diff4 = diff4.view(-1)
        d4 = torch.sqrt(diff4.dot(diff4))/N4

        return d1 + d2 + d3 + d4

        le = x3D[:, 0]*v[:, 0] - x3D[:, 1]*v[:, 1]
        le = le.view(-1)
        ll = torch.sqrt(le.dot(le))

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
            dleng = (torch.sqrt(le0.dot(le0)) - torch.sqrt(le1.dot(le1)))
            ll = ll + torch.sqrt(dleng*dleng)

        return d1 + d2 + d3 + d4 + ll/self.lengs.shape[0]

        vecX = x3D[:, :23, 0] - x[:, :23, 0]
        vecY = x3D[:, :23, 1] - x[:, :23, 1]
        vecZ = x3D[:, :23, 2]
        lx = (vecZ/vecX)*(0.5 - x3D[:, :23, 0]) + x3D[:, :23, 2]
        ly = (vecZ/vecY)*(0.5 - x3D[:, :23, 1]) + x3D[:, :23, 2]
        dist = Variable(torch.ones((s3D[0],self.Nj - 1)).float(), requires_grad=True).cuda()
        d = np.array(d)
        for i in range(s3D[0]):
            dist[i, :] = dist[i, :] * d[i]
        lx = lx + dist
        ly = ly + dist
        cnt = 0
        for i in range(s3D[0]):
            for j in range(self.Nj - 1):
                if int(v[i, j, 0]) == 0:
                    lx[i, j] = lx[i, j]*0
                    ly[i, j] = ly[i, j]*0
                elif lx[i, j] == float('inf') or lx[i, j] == float('nan'):
                    lx[i, j] = lx[i, j]*0
                    ly[i, j] = ly[i, j]*0
                else:
                    cnt = cnt + 1

        lx = lx.view(-1)
        ly = ly.view(-1)
        d5 = torch.sqrt((lx.dot(lx) + ly.dot(ly))) / (cnt * 2)

        return d1 + d2 + d3 + d4 + d5
        

def mean_squared_error3D2(o, h, o3D, h3D, d, t, t3D, v, use_visibility=False, path = None,col=14):
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
    return MeanSquaredError3D2(use_visibility, col=col)(o, h, o3D, h3D, d, t, t3D, v, path)
