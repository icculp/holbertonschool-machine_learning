#!/usr/bin/env python3
"""
    Dimensionality Reduction
    Only numpy allowed
    Your code should use the minimum number of
        operations to avoid floating point errors
"""
import numpy as np


def grads(Y, P):
    """ doc
    """
    dY = np.ndarray(Y.shape)
    Q = np.ndarray(P.shape)
    return dY, Q
