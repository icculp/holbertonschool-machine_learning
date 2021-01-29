#!/usr/bin/env python3
"""
    Convolutions project
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Performs SAME convolution on grayscale
        images is ndarray (m, h, w) containing multiple images
        m # images, h height in pixels, w width in pixels
        kernel ndarray (kh, kw) kernel for convolution
        two for loops allowed
        Returns: ndarray containing convoluted images
    """
    input_h = images.shape[1]
    input_w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    out_h = images.shape[1]
    out_w = images.shape[2]
    m = images.shape[0]

    """ Same = with padding, so have to create padded array """
    pad_along_height = max((out_h - 1) + kh - input_h, 0)
    pad_along_width = max((out_w - 1) + kw - input_w, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    output = np.zeros((m, out_h, out_w))
    image_padded = np.zeros((m, input_h + pad_along_height,
                            input_w + pad_along_width))
    image_padded[:, pad_top:-pad_bottom, pad_left:-pad_right] = images
    image_padded = np.pad(images, ((0, 0), (pad_top, pad_bottom),
                          (pad_left, pad_right)), 'constant')
    for x in range(out_w):
        for y in range(out_h):
            output[:, y, x] = (np.
                               tensordot(image_padded[:, y:y + kh,
                                         x:x + kw], kernel))
    return output
