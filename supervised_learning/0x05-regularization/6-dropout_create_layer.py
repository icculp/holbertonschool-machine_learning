#!/usr/bin/env python3
"""
    Regularization project
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Creates tensorflow layer using dropout
        prev is tensor containing output of previous layer
        n is number of nodes new layer contains
        activation is activation function of new layer
        keep_prob probability node will be kept
        Returns: output of new layer
    """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop = tf.layers.Dropout(rate=keep_prob)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=weights,
                            kernel_regularizer=drop)

    return layer(prev)
