#!/usr/bin/env python3
"""
    Transfer Learning
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """ Preprocesses data for model
        X ndarray (m, 32, 32, 3) ontaining CIFAR 10 data
        Y ndarray (m, containing CIFAR 10 labels
        Returns X_p, Y_p
            X_p ndarray containing preprocessed X
            Y_p ndarray containing preprocessed X
        model must exceed 87% validation accuracy
        Save trained model as ./cifar10.h5
        
    """

