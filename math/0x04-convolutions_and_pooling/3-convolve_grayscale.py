#!/usr/bin/env python3
"""
    Convolutions project
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ Performs custom padding convolution on grayscale
        images is ndarray (m, h, w) containing multiple images
        m # images, h height in pixels, w width in pixels
        kernel ndarray (kh, kw) kernel for convolution
        two for loops allowed
        padding is a tuple of (ph, pw)
        Returns: ndarray containing convoluted images
    """
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    input_h = images.shape[1]
    input_w = images.shape[2]

    m = images.shape[0]
    if padding == 'same':
        """ Account for even kernels by adding 1, when k is even """
        padh = int(((input_h - 1) * stride[0] + kh -
                   input_h) / 2) + int(kh % 2 == 0)
        padw = int(((input_w - 1) * stride[1] + kw -
                   input_w) / 2) + int(kh % 2 == 0)
        image_padded = np.pad(images, ((0, 0), (padh, padh),
                              (padw, padw)), 'constant',
                              constant_values=0)
    elif padding == 'valid':
        padh = 0
        padw = 0
        image_padded = images
    elif type(padding) == tuple:
        padh, padw = padding
        image_padded = np.pad(images, ((0, 0), (padh, padh),
                              (padw, padw)), 'constant',
                              constant_values=0)
    out_h = int(np.floor(images.shape[1] + 2 * padh - kh) / stride[0]) + 1
    out_w = int(np.floor(images.shape[2] + 2 * padw - kw) / stride[1]) + 1
    output = np.zeros((m, out_h, out_w))
    for x in range(out_w):
        for y in range(out_h):
            ys = stride[0] * y
            xs = stride[1] * x
            output[:, y, x] = (np.
                               tensordot(image_padded[:, ys:ys + kh,
                                         xs:xs + kw], kernel))
    return output
