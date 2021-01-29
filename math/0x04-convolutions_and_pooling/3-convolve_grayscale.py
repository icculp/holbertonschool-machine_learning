#!/usr/bin/env python3
"""
    Convolutions project
"""
import numpy as np
from math import ceil, floor


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
        out_h = int(ceil(float(input_h) / float(stride[0])))
        out_w = int(ceil(float(input_w) / float(stride[1])))
        padh = int(kh / 2)
        padw = int(kw / 2)
        image_padded = np.pad(images, ((0, 0), (padh, padh),
                             (padw, padw)), 'constant',
                             constant_values=0)
    elif padding == 'valid':
        out_h = int(ceil(float(input_h - kh + 1) / float(stride[0])))
        out_w = int(ceil(float(input_w - kw + 1) / float(stride[1])))
        image_padded = images
    output = np.zeros((m, out_h, out_w))
    for x in range(out_w):
        for y in range(out_h):
            output[:, y, x] = (np.
                               tensordot(image_padded[:, y:y + kh,
                                         x:x + kw], kernel))
    return output
