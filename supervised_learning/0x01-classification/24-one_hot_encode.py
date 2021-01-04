#!/usr/bin/env python3
"""
    One hot encoder
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ Defines a deep neural network
        performing binary classification """
    shape = (Y.size, Y.max() + 1)
    hot = np.zeros(shape)
    r = np.arange(Y.size)
    hot[r, Y] = 1
    return hot
