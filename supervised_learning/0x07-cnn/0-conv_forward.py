#!/usr/bin/env python3
"""
    Convolutional Neural Networks
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Forward prop over a convolutional layer
        A_prev contains output of previous layer (m, h_prev, w_prev, c_prev)
        W contains kernels (kh, kw, c_prev, c_new)
            c_prev is number of channels in previous layer
            c_new is # of channels in output
        b contains biases (1, 1, 1, c_new)
        activation to be applied to convolution
        padding is same or vlid
        stride contains strides
        Returns: output of convolutional layer
    """
    '''
    w = W
    a = A_prev
    b = b
    aw_ = np.matmul(w,
                    a + b)
    '''

    kh = W.shape[0]
    kw = W.shape[1]
    kc = W.shape[2]
    nc = W.shape[3]
    input_h = A_prev.shape[1]
    input_w = A_prev.shape[2]
    input_c = A_prev.shape[3]
    m = A_prev.shape[0]
    if padding == 'same':
        """ Account for even kernels by adding 1, when k is even """
        padh = int(((input_h - 1) * stride[0] + kh -
                   input_h) / 2) + 1
        '''int(kh % 2 == 0)'''
        padw = int(((input_w - 1) * stride[1] + kw -
                   input_w) / 2) + 1
        '''int(kh % 2 == 0)'''
    elif padding == 'valid':
        padh = 0
        padw = 0
        image_padded = A_prev
    elif type(padding) == tuple:
        padh, padw = padding
    image_padded = np.pad(A_prev, ((0, 0), (padh, padh),
                          (padw, padw), (0, 0)), 'constant',
                          constant_values=0)
    out_h = int(np.floor(A_prev.shape[1] + 2 * padh - kh) / stride[0]) + 1
    out_w = int(np.floor(A_prev.shape[2] + 2 * padw - kw) / stride[1]) + 1
    output = np.zeros((m, out_h, out_w, nc))
    for w in range(nc):
        for x in range(out_w):
            for y in range(out_h):
                ys = stride[0] * y
                xs = stride[1] * x
                output[:, y, x, w] = np.sum(np.multiply(W[:, :, :, w],
                                            image_padded[:, ys:ys +
                                            kh, xs:xs + kw, :]),
                                            axis=(1, 2, 3))

    act = activation(output + b)
    return act
