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
            raise ValueError("nodes must be a positve integer")
        self.W1 = np.random.randn(nodes, nx)
        '''self.b1 = np.array([np.array([0.]) for i in range(nodes)])'''
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
