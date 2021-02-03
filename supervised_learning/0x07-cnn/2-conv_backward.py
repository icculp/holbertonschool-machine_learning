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

    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    if padding == 'same':
        """ Account for even kernels by adding 1, when k is even """
        padh = int(((input_h - 1) * stride[0] + kh -
                   input_h) / 2) + int(kh % 2 == 0)
        padw = int(((input_w - 1) * stride[1] + kw -
                   input_w) / 2) + int(kh % 2 == 0)
        A_prev_pad = np.pad(A_prev, ((0, 0), (padh, padh),
                            (padw, padw), (0, 0)), 'constant',
                            constant_values=0)
        dZ_pad = np.pad(dA, ((0, 0), (padh, padh),
                        (padw, padw), (0, 0)), 'constant',
                        constant_values=0)
    elif padding == 'valid':
        padh = 0
        padw = 0
        A_prev_pad = A_prev
        dZ_pad = dA

    '''#out_h = int(np.floor(input_h - kh) / stride[0]) + int(kw % 2 == 0)
    #out_w = int(np.floor(input_w - kw) / stride[1]) + int(kh % 2 == 0)
    #output = np.zeros((m, out_h, out_w, input_c))'''
    for i in range(m):
        '''#a_prev = A_prev_pad[i]
        #dz = dZ_pad[i]'''
        for x in range(h_new):
            for y in range(w_new):
                for c in range(c_new):
                    ys = stride[0] * y
                    xs = stride[1] * x
                    '''A_prev[i, xs:xs + kh,ys:ys + kw, :]'''
                    dZ_pad[i, xs:xs + kh, ys:ys + kw, :] += W[:, :, :, c] *\
                        dZ[i, x, y, c]
                    dW[:, :, :, c] += A_prev_pad[i, xs:xs +
                                                 kh, ys:ys + kw, :] *\
                        dZ[i, x, y, c]
                    db[:, :, :, c] += dZ[i, xs, ys, c]
                    '''print(dX.shape)
                    print(W.shape)
                    print(dZ.shape)
                    dX[:, x:x + kh, y:y + kw, w] += W * dZ[:, x, y, :]
                    dW += A_prev[:, x:x + kh, y:y + kw, :] * dZ[:,x,y:]'''
        if padh != 0:
            dA[i, :, :, :] = dZ_pad[padh:-padh, padw:-padw, :]
        else:
            '''print(dZ_pad.shape)'''
            dA[i, :, :, :] = dZ_pad[i, :, :, :]
    return dA, dW, db
