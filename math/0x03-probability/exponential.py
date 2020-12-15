#!/usr/bin/env python3
"""
    Tasks 3-5, building exponential class
"""


class Exponential:
    """ A class that represents an exponential distribution """

    def __init__(self, data=None, lambtha=1.):
        """ constructor for Poisson class """
        self.data = data
        if data is None:
            self.lambtha = float(lambtha)
        else:
            self.lambtha = 1 / (float(sum(self.data) / len(self.data)))

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
    def lambtha(self):
        """ lambtha getter """
        return self.__lambtha

    @lambtha.setter
    def lambtha(self, value):
        """ lambtha setter """
        if value <= 0:
            raise ValueError("lambtha must be a positive value")
        self.__lambtha = float(value)

    def pdf(self, k):
        """ probability density function """
        e = 2.7182818285
        '''if self.data is None:
            return 0'''
        if type(k) is not int:
            '''k = int(k)'''
            return 0
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
        mean = 1 / self.lambtha
        p = 1 - (e ** ((k * -1) / mean))
        '''
        for i in range(k + 1):
            num = (e ** (self.lambtha * -1)) * (self.lambtha ** i)
            den = 1
            for i in range(0, i + 1):
                if i == 0:
                    pass
                else:
                    den = den * i
            cumulative.append(num / den)
        '''
        return p
