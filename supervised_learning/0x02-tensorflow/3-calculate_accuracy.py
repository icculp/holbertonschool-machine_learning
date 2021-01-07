#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ Calculates accuracy (cost) of a prediction """
    '''loss = tf.losses.softmax_cross_entropy('''
    loss = tf.metrics.accuracy(
        y,
        y_pred, name="Meanie")
    m = tf.convert_to_tensor(loss[0], name="Mean")
    return tf.reduce_mean(m, name="Mean")
