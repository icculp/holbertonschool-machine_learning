#!/usr/bin/env python3
"""
    Tasks 10-12, building Binomial class
"""


class Binomial:
    """ A class that represents a binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ constructor for binomial class """
        self.data = data
        if data is None:
            self.n = n
            self.p = p
        if data is not None:
            mean = sum(self.data) / len(self.data)
            vals = [(self.data[i] - mean) ** 2 for i in range(len(self.data))]
            var = sum(vals) / (len(self.data) - 1)
            q = var / mean
            p = 1 - q
            self.n = mean / p
            self.p = mean / self.n

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
    def n(self):
        """ n getter """
        return self.__n

    @n.setter
    def n(self, value):
        """ n setter """
        if value <= 0:
            raise ValueError("n must be a positive value")
        self.__n = int(value)

    @property
    def p(self):
        """ p getter """
        return self.__p

    @p.setter
    def p(self, value):
        """ p setter """
        if value <= 0 or value >= 1:
            raise ValueError("p must be greater than 0 and less than 1")
        self.__p = float(value)

    def pmf(self, k):
        """ probability mass function """
        e = 2.7182818285
        pi = 3.1415926536

    def cdf(self, k):
        """ cumultive density function """
        e = 2.7182818285
        pi = 3.1415926536
