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
            calculate n and p from data
            round n to int
            p then n then redo p

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
        if n <= 0:
            raise ValueError("n must be a positive value")
        self.__n = int(value)

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

    def z_score(self, x):
        """ calculates the z-score given x """
        return (x - self.mean) / self.stddev

    def x_value(self, x):
        """ calculates the x value given a z-score  """
        return (x * self.stddev) + self.mean

    def pdf(self, k):
        """ probability density function """
        e = 2.7182818285
        pi = 3.1415926536
        return (1 / (self.stddev * ((2 * pi) ** (.5)))) *\
               (e ** (-(1/2) * (((k - self.mean) / self.stddev) ** 2)))

    def cdf(self, k):
        """ cumultive density function """
        e = 2.7182818285
        pi = 3.1415926536
        x = (k - self.mean) / (self.stddev * (2 ** .5))
        erf = (2 / (pi ** .5)) *\
              (x - ((x ** 3) / 3) +
                   ((x ** 5) / 10) -
                   ((x ** 7) / 42) +
                   ((x ** 9) / 216))
        return (1 + erf) / 2
