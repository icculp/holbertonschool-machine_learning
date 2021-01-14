#!/usr/bin/env python3
"""
    Optimization project
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ Updates a variable using RMSProp opmtimizer
        (alpha, beta, eps, W, dW, dW_prev)

        loss is the loss of the network
        alpha is the learning rate
        beta2 is RMSProp weight
        epsilon is small number to avoid division by zero

        Returns: RMSProp optimiization operation
    """
    opt = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    return opt.minimize(loss)
