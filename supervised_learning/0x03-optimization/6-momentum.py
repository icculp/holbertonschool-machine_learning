#!/usr/bin/env python3
"""
    Optimization project
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ Creates tf training op for NN
        using gradient descent w/ momentum optimizer

        loss is loss of NN
        alpha is learning rate
        beta1 is momentum weight

        Returns: momentum optimization operation
    """
    opt = tf.train.MomentumOptimizer(alpha, beta1)
    return opt.minimize(loss)
