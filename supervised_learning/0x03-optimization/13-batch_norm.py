#!/usr/bin/env python3
"""
    Optimization project
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ Normalizes an unactivated output using batch normalization

        Z is ndarray shape (m, n)
            m data points
            n features in Z
        gamma ndarray shape (1, n) containing scales used
        beta ndarray (1, n) containing offsets
        epsilon small number to avoid div/0

        Returns: normlized Z matrix
    """
    norm = (Z - np.mean(Z, axis=0)) / (np.std(Z + epsilon, axis=0))
    return norm * gamma + beta
