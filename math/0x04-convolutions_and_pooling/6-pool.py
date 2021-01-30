#!/usr/bin/env python3
"""
    Convolutions project
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ Performs custom padding convolution on grayscale
        images is ndarray (m, h, w) containing multiple images
        m # images, h height in pixels, w width in pixels
        kernel ndarray (kh, kw) kernel for convolution
        two for loops allowed
        padding is a tuple of (ph, pw)
        Returns: ndarray containing convoluted images
    """
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    input_h = images.shape[1]
    input_w = images.shape[2]
    input_c = images.shape[3]
    m = images.shape[0]
    out_h = int(np.floor(images.shape[1] - kh) / stride[0]) + 1
    out_w = int(np.floor(images.shape[2] - kw) / stride[1]) + 1
    output = np.zeros((m, out_h, out_w, input_c))
    for x in range(out_w):
        for y in range(out_h):
            ys = stride[0] * y
            xs = stride[1] * x
            if mode == 'max':
                output[:, y, x, :] = np.amax(images[:, ys:ys +
                                             kh, xs:xs + kw, :], axis=(1, 2))
            elif mode == 'avg':
                output[:, y, x, :] = np.average(images[:, ys:ys +
                                                kh, xs:xs + kw, :],
                                                axis=(1, 2))
    return output
