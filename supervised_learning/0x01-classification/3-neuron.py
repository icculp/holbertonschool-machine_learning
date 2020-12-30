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

    def forward_prop(self, X):
        """ Forward propogation of the neuron """
        aw = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-aw))
        return self.__A

    def cost(self, Y, A):
        """ calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = -(1 / m) * loss
        return cost
