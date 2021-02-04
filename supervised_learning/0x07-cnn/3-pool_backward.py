#!/usr/bin/env python3
"""
    Convolutional Neural Networks
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Back prop over convolutional layer
        dA contains partial derivative w/ respect to output of pooling layer
            (m, h_new, w_new, c_new)
        A_prev (m, h_prev, w_prev, c_prev) output of prev layer
        kernel_shape (kh, kw)
        padding 'same' or 'valid'
        stride (sh, sw)
        mode 'max' or 'avg'
        Returns: partial der w/ respect to previous layer (dA_prev)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dA.shape
    dA_prev = np.zeros(A_prev.shape)
    kh, kw = kernel_shape
    for i in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for c in range(c_new):
                    ys = stride[1] * y
                    xs = stride[0] * x
                    if mode == 'max':
                        A = A_prev[i, xs:xs + kh, ys:ys + kw, c]
                        mask = A == np.max(A)
                        dA_prev[i, xs:xs + kh,
                                ys:ys + kw, c] += mask * dA[i, x, y, c]
                    elif mode == 'avg':
                        average = dA[i, x, y, c] / (kh * kw)
                        average = average * np.ones(kernel_shape)
                        dA_prev[i, xs:xs + kh, ys:ys + kw, c] += average
    return dA_prev
