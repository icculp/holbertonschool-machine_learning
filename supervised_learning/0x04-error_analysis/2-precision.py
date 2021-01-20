#!/usr/bin/env python3
"""
    Error Analysis project
"""
import numpy as np


def precision(confusion):
    """ calculates the precision for each class in confusion matrix
        confusion is ndarray (classes, classes), rows correct coluns predicted
        Returns: ndarray (classes,) containing sensitivity for each class
    """
    '''totals = sum(sum(label) for label in confusion)'''
    '''print("totals", totals)'''
    classes = len(confusion)
    positives = np.sum(confusion, axis=0)
    return np.array([confusion[i][i] / positives[i] for i in range(classes)])
