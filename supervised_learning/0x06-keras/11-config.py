#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def save_config(network, filename):
    """ Saves the entire model
        Returns: None
    """
    with open(filename, "w") as json_file:
        json_file.write(network.to_json())


def load_config(filename):
    """ Loads weights to a model """
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
