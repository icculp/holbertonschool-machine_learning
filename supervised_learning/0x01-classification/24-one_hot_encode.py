#!/usr/bin/env python3
"""
    One hot encoder
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ Converts numeric label vector into one-hot matrix """
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    '''if classes < 1:
        return None'''
    if len(Y.shape) != 1:
        return None
    '''if Y.shape[0] < 1:
        return None'''
    '''if classes != Y.max() + 1:
        return None'''
    try:
        shape = (classes, Y.shape[0])
        hot = np.eye(classes)[Y]
        '''hot = np.zeros(shape)'''
        '''r = np.arange(Y.shape[0])'''
        '''hot[Y, r] = 1'''
        return hot.T
    except Exception:
        return None
