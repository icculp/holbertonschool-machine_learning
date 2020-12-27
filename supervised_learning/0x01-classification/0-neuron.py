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
        self.nx = nx
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0

    @property
    def nx(self):
        """ nx getter """
        return self.__nx

    @nx.setter
    def nx(self, value):
        """ nx setter """
        if type(value) is not int:
            raise TypeError("nx must be an integer")
        if value < 1:
            raise ValueError("nx must be a positive integer")
