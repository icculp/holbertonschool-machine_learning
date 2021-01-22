#!/usr/bin/env python3
"""
    Regularization project
"""
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
    return
