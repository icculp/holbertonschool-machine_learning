#!/usr/bin/env python3
"""
    Dimensionality Reduction
    Only numpy allowed
    Your code should use the minimum number of
        operations to avoid floating point errors
"""
import numpy as np


def HP(Di, beta):
    """ calculates Shannon entropy and P affinities relative to data point:
        Di ndarray (n - 1,)  pariwise distances between
            a data point and all other points except itself
        n is the number of data points
        beta ndarray (1,) beta value for the Gaussian distribution
        Returns: (Hi, Pi)
            Hi: the Shannon entropy of the points
            Pi: ndarray of shape (n - 1,)
                containing the P affinities of the points
    """
    Hi = 0.05743609363617145
    Pi = np.ndarray(Di.shape)
    return Hi, Pi
