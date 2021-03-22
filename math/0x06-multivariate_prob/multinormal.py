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
        self.n = n
        self.data = data
        self.stdev = np.std(data, axis=1)
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
        '''#print("x", x)
        pdf = (1.0 / (self.stdev * np.sqrt(2 * np.pi)))\
        * np.exp(-0.5*((x - self.mean) / self.stdev) ** 2)
        var = self.stdev ** 2
        denom = (2* np.pi * var) ** .5
        num = np.exp(-(x - self.mean)**2/(2*var))'''
        e = 2.7182818285
        pi = 3.1415926536
        mean = self.mean
        std = self.stdev
        n = len(self.mean)
        det = np.linalg.det(self.cov)
        xm = x - self.mean
        norm = 1.0 / (((2 * np.pi) ** float(n / 2)) * np.sqrt(det))
        inv = np.linalg.inv(self.cov)
        res = np.exp(-0.5 * (xm.T @ inv @ xm))
        return (norm * res)[0][0]
        '''xp = (-.5 * xm * (1 / self.cov) * xm.T)
        return (1 / ((2 * pi) ** (n / 2)))\
            * (np.sqrt(E) ** -1)\
            * np.exp(xp)
        return (1 / (std * ((2 * pi) ** (.5)))) *\
               (e ** (-(1/2) * (((x - mean) / std) ** 2)))

        return num/denom
        return pdf'''
