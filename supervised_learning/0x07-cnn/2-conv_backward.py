#!/usr/bin/env python3
"""
    Convolutional Neural Networks
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding='same', stride=(1, 1)):
    """ Back prop over convolutional layer
        dZ contains partial derivative w/ respect to unactivated output
            (m, h_new, w_new, c_new)
        A_prev (m, h_prev, w_prev, c_prev) output of prev layer
        W contains kernels (kh, kw, c_prev, c_new)
            c_prev is number of channels in previous layer
            c_new is # of channels in output
        b (1, 1, 1, c_new)
        padding 'same' or 'valid'
        stride (sh, sw)
        Returns: partial der w/ respect to previous layer,
                kernels, and biases respectively
                (dAprev, dW, db)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    m, h_new, w_new, c_new = dZ.shape
    '''print(dZ.shape)
    print(A_prev.shape)'''
    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    if padding == 'same':
        """ Account for even kernels by adding 1, when k is even """
        padh = int(((h_prev - 1) * stride[0] + kh -
                   h_prev) / 2) + 1
        padw = int(((w_prev - 1) * stride[1] + kw -
                   w_prev) / 2) + 1
        A_prev_pad = np.pad(A_prev, ((0, 0), (padh, padh),
                            (padw, padw), (0, 0)), 'constant',
                            constant_values=0)
        dA_prev_pad = np.pad(dA, ((0, 0), (padh, padh),
                             (padw, padw), (0, 0)), 'constant',
                             constant_values=0)
    elif padding == 'valid':
        padh = 0
        padw = 0
        A_prev_pad = A_prev
        dA_prev_pad = dA
    for i in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for c in range(c_new):
                    ys = stride[1] * y
                    xs = stride[0] * x
                    dA_prev_pad[i, xs:xs + kh, ys:ys +
                                kw, :] += W[:, :, :, c] *\
                        dZ[i, x, y, c]
                    dW[:, :, :, c] += A_prev_pad[i, xs:xs +
                                                 kh, ys:ys + kw, :] *\
                        dZ[i, x, y, c]
                    db[:, :, :, c] += dZ[i, x, y, c]
        if padding == 'valid':
            dA[i, :, :, :] = dA_prev_pad[i, :, :, :]
        else:
            dA[i, :, :, :] = dA_prev_pad[i, padh:-padh, padw:-padw, :]
    return dA, dW, db
