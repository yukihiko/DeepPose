# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch.nn as nn
import torch


class MeanSquaredErrorFC3(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=True):
        super(MeanSquaredErrorFC3, self).__init__()
        self.use_visibility = use_visibility

    def forward(self, *inputs):
        x, t, v = inputs
        op = x[:, :, 0:2]

        diff1 = (op - t)*v
        N1 = (v.sum()/2).data[0]
        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1)/N1

        return d1
                
        ov = torch.cat([x[:, :, 2:] , x[:, :, 2:]], dim=2)

        diff1 = op*torch.floor(ov + 0.5) - t*v
        N1 = (v.sum()/2).data[0]
        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1)/N1

        diff2 = ov - v
        N2 = v.sum().data[0]
        diff2 = diff2.view(-1)
        d2 = diff2.dot(diff2)/N2

        return d1 + d2


def mean_squared_error_FC3(x, t, v, use_visibility=False):
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
    return MeanSquaredErrorFC3(use_visibility)(x, t, v)
