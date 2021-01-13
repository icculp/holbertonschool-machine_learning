#!/usr/bin/env python3
"""
    Optimization project
"""


def normalize(X, m, s):
    """ Normalizes/standardizes a matrix
        X is ndarray with shape (d, nx)
            d number of data points
            nx number of features
        m is ndarray shape (nx,) contains mean of all features of X
        s is ndarray shape (nx,) contands std of all features of X
        Returns: Normalized X matrix
    """
    return (X - m) / s
