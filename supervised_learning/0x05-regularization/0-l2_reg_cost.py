#!/usr/bin/env python3
"""
    Regularization project
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculates the cost of a neural network w/ L2 regularization
        cost is cost of NN w/o L2 regularization
        lambtha is regularization parameter
        weights is dict of weights and biases (ndarrays)
        L is number of layers
        m is number of data points
        Returns: cost of NN accounting for L2 regularization
    """
    '''print(weights.keys())'''
    w2_norm = 0
    for i in range(1, L + 1):
        w = "W" + str(i)
        w2_norm += np.sqrt(np.sum(weights[w] ** 2))
    '''print(weights)'''
    l2_cost = cost + (lambtha / (2 * m)) * w2_norm
    return l2_cost
