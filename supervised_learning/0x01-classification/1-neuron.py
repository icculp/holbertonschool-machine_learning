#!/usr/bin/env python3
"""
    Build a neuron
"""
import numpy as np


class Neuron():
    """ Defines a single neuron performing binary classification """

    def __init__(self, nx):
        """ nx is number of input features
            must be an integer and >= 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ W getter """
        return self.__W

    @property
    def b(self):
        """ W getter """
        return self.__b

    @property
    def A(self):
        """ W getter """
        return self.__A
