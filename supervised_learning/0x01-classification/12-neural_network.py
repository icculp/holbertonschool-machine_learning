#!/usr/bin/env python3
"""
    Build a neural network
"""
import numpy as np


class NeuralNetwork:
    """ Defines a neural network with one hidden layer
        performing binary classification """

    def __init__(self, nx, nodes):
        """ nx is number of input features
            must be an integer and >= 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ W1 getter """
        return self.__W1

    @property
    def b1(self):
        """ b1 getter """
        return self.__b1

    @property
    def A1(self):
        """ A1 getter """
        return self.__A1

    @property
    def W2(self):
        """ W2 getter """
        return self.__W2

    @property
    def b2(self):
        """ b2 getter """
        return self.__b2

    @property
    def A2(self):
        """ A2 getter """
        return self.__A2

    def forward_prop(self, X):
        """ Forward propogation of the network """
        aw1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-aw1))
        aw2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-aw2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = -(1 / m) * loss
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the predictions of a neural network """
        prop1, prop2 = self.forward_prop(X)
        cost = self.cost(Y, prop2)
        x = np.where(prop2 >= 0.5, 1, 0)
        return x, cost
