#!/usr/bin/env python3
"""
    Regularization project
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Creates tensorflow layer including l2 regularization
        prev is tensor containing output of previous layer
        n is number of nodes new layer contains
        activation is activation function of new layer
        lambtha l2 regularization parameter
        Returns: output of new layer
    """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(scale=lambtha)
    l2_layer = tf.layers.Dense(n, activation=activation,
                               kernel_initializer=weights,
                               kernel_regularizer=reg)

    return l2_layer(prev)
