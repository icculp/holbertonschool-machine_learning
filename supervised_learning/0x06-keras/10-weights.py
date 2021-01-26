#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ Saves the entire model
        Returns: None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename, ):
    """ Loads weights to a model """
    network.load_weights(filename)
