#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ Saves the entire model
        Returns: None
    """
    return network.evaluate(data, labels, verbose=verbose)
