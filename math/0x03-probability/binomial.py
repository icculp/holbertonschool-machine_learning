#!/usr/bin/env python3
"""
    Tasks 10-12, building Binomial class
"""


class Binomial:
    """ A class that represents a binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ constructor for Binomial class """
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
        if k < 0:
            return 0
        k = int(k)
        q = 1 - self.p
        n_f, k_f, nk_f = 1, 1, 1
        nk_dif = self.n - k
        for i in range(1, self.n + 1):
            n_f = n_f * i
        for i in range(1, k + 1):
            k_f = k_f * i
        for i in range(1, nk_dif + 1):
            nk_f = nk_f * i
        nk = (n_f / (k_f * nk_f))
        return nk * (self.p ** k) * (1 - self.p) ** (self.n - k)

    def cdf(self, k):
        """ cumultive density function """
        if k < 0:
            return 0
        k = int(k)
        p = []
        for i in range(k + 1):
            p.append(self.pmf(i))
        return sum(p)
