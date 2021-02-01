#!/usr/bin/env python3
"""
    Convolutional Neural Networks
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Forward prop over a convolutional layer
        A_prev contains output of previous layer (m, h_prev, w_prev, c_prev)
        W contains kernels (kh, kw, c_prev, c_new)
        b contains biases (1, 1, 1, c_new)
        activation to be applied to convolution
        padding is same or vlid
        stride contains strides
        Returns: output of convolutional layer
    """
    
