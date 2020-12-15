#!/usr/bin/env python3
"""
    Tasks 6-9, building Normal class
"""


class Normal:
    """ A class that represents a normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ constructor for Normal class """
        self.data = data
        if data is None:
            self.mean = mean
            self.stddev = stddev
        else:
            self.mean = sum(self.data) / len(self.data)
            difs = [(self.data[i] - self.mean) for i in range(len(self.data))]
            sifs = [difs[j] * difs[j] for j in range(len(difs))]
            self.stddev = (sum(sifs) / len(self.data)) ** (1/2)

    @property
    def data(self):
        """ data getter """
        return self.__data

    @data.setter
    def data(self, value):
        """ data setter """
        if value is None:
            self.__data = None
            return
        if type(value) is not list:
            raise TypeError("data must be a list")
        if len(value) < 2:
            raise ValueError("data must contain multiple values")
        self.__data = value

    @property
    def mean(self):
        """ mean getter """
        return self.__mean

    @mean.setter
    def mean(self, value):
        """ mean setter """
        if value <= 0:
            raise ValueError("mean must be a positive value")
        self.__mean = float(value)

    @property
    def stddev(self):
        """ lambtha getter """
        return self.__stddev

    @stddev.setter
    def stddev(self, value):
        """ stdev setter """
        if value <= 0:
            raise ValueError("stddev must be a positive value")
        self.__stddev = float(value)

    def pmf(self, k):
        """ probability mass function """
        e = 2.7182818285
        '''if self.data is None:
            return 0'''
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        '''if k < 0 or k > len(self.data):
            return 0'''
        num = (e ** (self.lambtha * -1)) * (self.lambtha ** k)
        den = 1
        for i in range(1, k + 1):
            den = den * i
        return num / den

    def cdf(self, k):
        """ cumultive density function """
        e = 2.7182818285
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cumulative = []
        for i in range(k + 1):
            num = (e ** (self.lambtha * -1)) * (self.lambtha ** i)
            den = 1
            for i in range(0, i + 1):
                if i == 0:
                    pass
                else:
                    den = den * i
            cumulative.append(num / den)
        return sum(cumulative)
