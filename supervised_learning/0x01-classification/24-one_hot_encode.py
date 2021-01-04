#!/usr/bin/env python3
"""
    One hot encoder
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ Converts numeric label vector into one-hot matrix """
    shape = (Y.size, classes)
    hot = np.zeros(shape)
    r = np.arange(Y.size)
    hot[Y, r] = 1
    return hot
