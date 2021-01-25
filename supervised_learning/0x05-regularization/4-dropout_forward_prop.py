#!/usr/bin/env python3
"""
    Regularization project
"""
import numpy as np
import tensorflow as tf


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Forward prop w/ dropout from scratch
        X is ndarray (nx, m) nx input features, m data points
        weights is dict of weights and biases
        L is num of layers
        keep_prob is probability node will be kept
        All layers except last use tanh, then softmax for last
        Returns: dict containing outputs of each layer and
        the dropout mask of each layer
    """
    def tanh_act(aw):
        """ tanh activation function """
        return np.tanh(aw)

    def soft_act(aw):
        """ softmax activation function """
        return np.exp(aw) / np.sum(np.exp(aw), axis=0)
    cache = dict()
    cache['A0'] = X
    for i in range(L):
        w = "W" + str(i + 1)
        a = "A" + str(i)
        b = "b" + str(i + 1)
        d = "D" + str(i + 1)

        aw_ = np.matmul(weights[w],
                        cache[a]) + weights[b]
        A = "A" + str(i + 1)
        if i == (L - 1):
            act = tanh_act(aw_)
            act = soft_act(aw_)
            cache[A] = act
        else:
            act = tanh_act(aw_)
            cache[d] = np.random.rand(act.shape[0], act.shape[1])
            cache[d] = np.where(cache[d] < keep_prob, 1, 0)
            cache[A] = (1 / keep_prob) * np.multiply(act, cache[d])
    return cache
