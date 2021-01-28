#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ Performs valid convolution on grayscale
        images is ndarray (m, h, w) containing multiple images
        m # images, h height in pixels, w width in pixels
        kernel ndarray (kh, kw) kernel for convolution
        two for loops allowed
        Returns: ndarray containing convoluted images
    """
    out_h = images.shape[1] - kernel.shape[0] + 1
    out_w = images.shape[2] - kernel.shape[1] + 1
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    m = images.shape[0]
    output = np.zeros((m, out_h, out_w))
    for x in range(out_w):
        for y in range(out_h):
            output[y, x] = np.sum(kernel * images[:, y:y + kh,
                                  x:x + kw])
    return output
