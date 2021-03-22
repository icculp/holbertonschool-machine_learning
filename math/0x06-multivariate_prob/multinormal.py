#!/usr/bin/env python3
"""
    Multivariate Probability

    only numpy allowed
"""
import numpy as np


class MultiNormal:
    """
        Represents multivariate normal distribution
    """
    def __init__(self, data):
        """ data ndarray (d, n)
            n # data points, d # dims
        """
        if type(data) is not np.ndarray or\
                len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = (data - self.mean) @ (data.T - self.mean.T) / (n - 1)
