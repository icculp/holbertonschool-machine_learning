#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def predict(network, data, verbose=True):
    """ Saves the entire model
        Returns: None
    """
    return network.predict(data, verbose=verbose)
