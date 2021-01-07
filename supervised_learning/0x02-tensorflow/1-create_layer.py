#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """ Returns tensor output of the layer """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=w, name='layer')
    '''r = tf.keras.layers.Add(n, activation=activation, name='layer')(prev)'''
    return layer(prev)
