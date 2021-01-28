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
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    out_h = images.shape[1] - kh + 1
    out_w = images.shape[2] - kw + 1
    m = images.shape[0]
    output = np.zeros((m, out_h, out_w))
    for x in range(out_w):
        for y in range(out_h):
            output[slice(0, m, 1), y, x] = (kernel *
                                            images[slice(0, m, 1), y:y + kh,
                                                   x:x + kw]).sum()
    return output
