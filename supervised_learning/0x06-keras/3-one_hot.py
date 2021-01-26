#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Converts a label vector into one-hot matrix
        The last dimension of one-hot matrix must be number of classes
        Returns: One-hot matrix
    """
    return K.utils.to_categorical(labels, classes)
