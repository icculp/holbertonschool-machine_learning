#!/usr/bin/env python3
"""
    Optimization project
"""
import numpy as np


def shuffle_data(X, Y):
    """ Shuffles data points in two matrices the same way
        X is ndarray with shape (m nx)
            m number of data points
            nx number of features
        Y is ndarray with shape (m nx)
            m number of data points
            nx number of features
        Returns: shuffled X, Y matrices
    """
    permute = np.random.permutation(np.arange(X.shape[0]))
    return X[permute], Y[permute]
