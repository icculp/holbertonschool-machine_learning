#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ Calculates softmax cross-entropy loss of a prediction
        y is placeholder for labels of input data
        y_pred is tensor containing networks predictions
    """
    '''loss = tf.metrics.accuracy(
        y,
        y_pred, name="Meanie")
    print(loss)
    m = tf.convert_to_tensor(loss[0], name="Mean")'''
    m = tf.losses.softmax_cross_entropy(
        y,
        y_pred)
    return m
