#!/usr/bin/env python3
"""
    Optimization project
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ Creates a batch normalization layer
        prev is activated output of previous layer
        n is number of nodes
        activation is activation function to be used

        kernal initializer
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        gamma and beta initialized as vectors of 1, 0 respectively
        epsilon of 1e-8

        Returns: normlized Z matrix
    """
    k = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.Dense(n, kernel_initializer=k)
    mean, var = tf.nn.moments(base(prev), axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(base(prev), mean, var,
                                           beta, gamma, epsilon)
    return activation(batch_norm)
