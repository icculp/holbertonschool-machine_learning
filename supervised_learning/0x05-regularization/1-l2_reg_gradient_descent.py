#!/usr/bin/env python3
"""
    Regularization project
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Updates weights and biases of NN using grad desc w/ l2 reg
        Y is one-hot ndarray (classes, m)
        weights is dict of weights and biases (ndarrays)
        cache is dict of outputs of each layer
        alpha is learning rate
        lambtha is regularization parameter
        L is number of layers
        tanh activations on each except last softmax
    """
    '''print(weights.keys())'''
    '''print(cache.keys())'''
    m = Y.shape[1]
    wold = weights.copy()
    for i in range(L, 0, -1):
        A_i = cache['A' + str(i)]
        A_iless1 = cache['A' + str(i - 1)]
        if i == L:
            dz = np.subtract(A_i, Y)
        else:
            """ take derivative based on activation """
            der = 1 - (A_i ** 2)
            dz = np.matmul(wold['W' + str(i + 1)].T, dz2) * der
        dz2 = dz
        dw = np.matmul(dz, A_iless1.T) / m +\
            ((lambtha / m) * wold['W' + str(i)])
        wupdate = np.subtract(wold['W' + str(i)], np.multiply(alpha, dw))
        bupdate = np.subtract(weights['b' + str(i)],
                              np.multiply(alpha, np.sum(dz,
                                          axis=1, keepdims=True) / m))
        weights['W' + str(i)] = wupdate
        weights['b' + str(i)] = bupdate
    '''
    w2_norm = 0
    m = Y.shape[1]
    dw[l] = bpret + lambtha / m
    for i in range(L):
        w = "W" + str(i)
        w2_norm += np.sqrt(np.sum(weights[w] ** 2))
    '''
    '''same as above* w2_norm += np.linalg.norm(weights[w], 'fro')'''
    '''print(weights)'''
