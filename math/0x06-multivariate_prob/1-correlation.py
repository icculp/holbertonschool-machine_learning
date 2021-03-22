#!/usr/bin/env python3
"""
    Multivariate Probability

    only numpy allowed
"""
import numpy as np


def correlation(C):
    """ Calculates the correlation matrix
        C covariance matrix
            ndarray (d, d) d # dimensions
        Returns: correlation matrix (d, d)
    """
    if type(C) is not np.ndarray or len(C.shape) != 2:
        raise TypeError("C must be a numpy.ndarray")
    _, d = C.shape
    if C.shape != (d, d):
        raise ValueError("C must be a 2D square matrix")
    diag = np.sqrt(np.diag(C))
    return C / np.outer(diag, diag)
    return (diag ** -1) * C * (diag ** -1)
