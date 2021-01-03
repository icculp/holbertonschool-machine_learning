#!/usr/bin/env python3
"""
    Build a deep neural network
"""
import numpy as np


class DeepNeuralNetwork:
    """ Defines a deep neural network
        performing binary classification """

    def __init__(self, nx, layers):
        """ nx is number of input features
            must be an integer and >= 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)

        def checklist(li, i, L):
            """ checks if every element > 1 """
            if i == L:
                return
            if li[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            checklist(li, i + 1, L)

        checklist(layers, 0, self.__L)
        self.__cache = dict()
        self.__weights = dict()
        for i in range(1, self.__L + 1):
            if i == 1:
                l2 = nx
            else:
                l2 = layers[i - 2]
            l1 = layers[i - 1]
            w = 'W' + str(i)
            b = 'b' + str(i)
            self.__weights.update({w: np.random.randn(l1, l2)
                                  * np.sqrt(2 / l2)})
            self.__weights.update({b: np.zeros((l1, 1))})

    @property
    def L(self):
        """ L getter """
        return self.__L

    @property
    def cache(self):
        """ cache getter """
        return self.__cache

    @property
    def weights(self):
        """ weights getter """
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation of the network """
        self.__cache['A0'] = X

        def sig_act(aw):
            """ sigmoid activation function """
            return 1 / (1 + np.exp(-aw))

        for i in range(self.L):
            w = 'W' + str(i + 1)
            a = 'A' + str(i)
            b = 'b' + str(i + 1)
            aw_ = np.matmul(self.__weights[w],
                            self.__cache[a]) + self.__weights[b]
            A = 'A' + str(i + 1)
            self.__cache[A] = sig_act(aw_)
        return self.__cache[A], self.__cache

    def cost(self, Y, A):
        """ calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = -(1 / m) * loss
        return cost
