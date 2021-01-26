#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def save_model(network, filename):
    """ Saves the entire model
        Returns: None
    """
    network.save(filename)


def load_model(filename):
    """ Loads a model """
    return K.models.load_model(filename)
