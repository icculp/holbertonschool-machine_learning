#!/usr/bin/env python3
"""
    Multivariate Probability

    only numpy allowed
"""
import numpy as np


def mean_cov(X):
    """ Calculates the mean and covariance of a dataset
        X ndarray (n, d) n # data points, d # dimensions
        Returns: mean, cov
            (1, d), (d, d)

        NOT ALLOWED numpy.cov
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    ones = np.ones((n, 1))
    ''' method 0
    x = (X.T - mean.T) @ (X - mean) / (n - 1)
    '''

    ''' method 1 '''
    x = X - ones * ones.T @ X * ((ones.T @ ones) ** -1)
    x = (x.T @ x) * (1 / n)


    ''' method 2
    x = X - (ones @ ones.T @ X) * (1 / n)
    x = x.T @ x
    x = x * (1 / n)
    '''
    return mean, x
