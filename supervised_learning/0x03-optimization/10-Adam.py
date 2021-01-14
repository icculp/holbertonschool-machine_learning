#!/usr/bin/env python3
"""
    Optimization project
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ Creates Adam optimizer training operation in tf
        (loss, 0.001, 0.9, 0.99, 1e-8)

        loss is the loss of the NN
        alpha is the learning rate
        beta1 is first moment weight
        beta2 is second moment weight
        epsilon is small number to avoid division by zero

        Returns: Adam optimiization operation
    """
    opt = tf.train.AdamOptimizer(alpha, beta1=beta1,
                                 beta2=beta2, epsilon=epsilon)
    return opt.minimize(loss)
