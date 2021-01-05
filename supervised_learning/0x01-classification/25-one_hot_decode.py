#!/usr/bin/env python3
"""
    One hot encoder
"""
import numpy as np


def one_hot_decode(one_hot):
    """ Converts a one-hot matrix into a vector of labels """
    if type(one_hot) is not np.ndarray:
        return None
    try:
        hot = np.argmax(one_hot, axis=0)
        return hot
    except Exception:
        return None
