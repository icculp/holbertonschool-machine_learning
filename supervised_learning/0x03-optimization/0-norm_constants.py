#!/usr/bin/env python3
"""
    Optimization project
"""
import numpy as np


def normalization_constants(X):
    """ Calculates the normalization constants of a matrix
        X is ndarray with shape (m, nx)
        m number of data points
        nx number of features

        Returns: mean and sd for each feature
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
