#!/usr/bin/env python3
"""
    Regularization project
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Updates weights and biases of NN using grad desc w/ dropout
        Y is one-hot ndarray (classes, m)
        weights is dict of weights and biases (ndarrays)
        cache is dict of outputs of each layer
        alpha is learning rate
        lambtha is regularization parameter
        keep_prob is the probability that a node will be kept
        L is number of layers
        tanh activations on each except last softmax
    """
    '''print(weights.keys())'''
    '''print(cache.keys())'''
    m = Y.shape[1]
    wold = weights.copy()
    for i in range(L, 0, -1):
        A_i = cache['A' + str(i)]
        if i != L:
            d = 'D' + str(i)
            A_i = cache[d] * (1 / (keep_prob))
            '''#/ (keep_prob)#np.multiply(A_i, cache[d]) / (1 - keep_prob)'''
        A_iless1 = cache['A' + str(i - 1)]
        if i == L:
            dz = np.subtract(A_i, Y)
        else:
            """ take derivative based on tanh activation """
            der = 1 - (A_i ** 2)
            dz = np.matmul(wold['W' + str(i + 1)].T, dz2) * der
        dz2 = dz
        dw = np.matmul(dz, A_iless1.T) / m
        wupdate = np.subtract(wold['W' + str(i)], np.multiply(alpha, dw))
        bupdate = np.subtract(weights['b' + str(i)],
                              np.multiply(alpha, np.sum(dz,
                                          axis=1, keepdims=True) / m))
        weights['W' + str(i)] = wupdate
        weights['b' + str(i)] = bupdate
