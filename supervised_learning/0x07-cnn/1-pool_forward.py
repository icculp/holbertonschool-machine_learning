#!/usr/bin/env python3
"""
    Convolutional Neural Networks
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Forward prop over a pooling layer
        A_prev contains output of previous layer (m, h_prev, w_prev, c_prev)
        W contains kernels (kh, kw, c_prev, c_new)
            c_prev is number of channels in previous layer
            c_new is # of channels in output
        kernel_shape (kh, kw)
        stride (sh, sw)
        mode 'max' or 'avg'
        Returns: output of pooling layer
    """
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    input_h = A_prev.shape[1]
    input_w = A_prev.shape[2]
    input_c = A_prev.shape[3]
    m = A_prev.shape[0]

    out_h = int(np.floor(input_h - kh) / stride[0]) + int(kw % 2 == 0)
    out_w = int(np.floor(input_w - kw) / stride[1]) + int(kh % 2 == 0)
    output = np.zeros((m, out_h, out_w, input_c))
    for x in range(out_w):
        for y in range(out_h):
            ys = stride[0] * y
            xs = stride[1] * x
            if mode == 'max':
                output[:, y, x, :] = np.amax(A_prev[:, ys:ys +
                                             kh, xs:xs + kw, :],
                                             axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = np.average(A_prev[:, ys:ys +
                                                kh, xs:xs + kw, :],
                                                axis=(1, 2))
    return output
