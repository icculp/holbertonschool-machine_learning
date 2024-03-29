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
        self.L = len(layers)

        def checklist(li, i, L):
            """ checks if every element > 1 """
            if i == L:
                return
            if li[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            checklist(li, i + 1, L)

        checklist(layers, 0, self.L)
        self.cache = dict()
        self.weights = dict()
        for i in range(1, self.L + 1):
            if i == 1:
                l2 = nx
            else:
                l2 = layers[i - 2]
            l1 = layers[i - 1]
            w = 'W' + str(i)
            b = 'b' + str(i)
            self.weights.update({w: np.random.randn(l1, l2) * np.sqrt(2 / l2)})
            self.weights.update({b: np.zeros((l1, 1))})
