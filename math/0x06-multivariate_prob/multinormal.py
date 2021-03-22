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
        self.data = data
        self.stddev = np.sqrt(np.sum((data - self.mean) ** 2) / (n - 1))
        self.cov = (data - self.mean) @ (data.T - self.mean.T) / (n - 1)

    def pdf(self, x):
        """ calculates pdf at data point
            x ndarray (d, 1) data point to calculate pdf
                d # dims of multinomial instance
            Returns: value of pdf
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2 and x.shape != (x.shape[0], 1):
            raise ValueError("x must have the shape ({d}, 1)")
        '''(1.0 / (self.stdev * math.sqrt(2*math.pi)))
        * math.exp(-0.5*((x - self.mean) / self.stdev) ** 2)'''
        return None
