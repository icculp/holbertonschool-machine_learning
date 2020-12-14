#!/usr/bin/env python3
"""
    Task 0
"""


class Poisson:
    """ A class that represents a poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """ constructor for Poisson class """
        self.data = data
        if data is None:
            self.lambtha = float(lambtha)
        else:
            self.lambtha = float(sum(self.data) / len(self.data))

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
        if value < 0:
            raise ValueError("lambtha must be a positive value")
        self.__lambtha = float(value)