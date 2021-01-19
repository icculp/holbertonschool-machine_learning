#!/usr/bin/env python3
"""
    Error Analysis project
"""
import numpy as np


def sensitivity(confusion):
    """ calculates the sensitivity for each class in confusion matrix
        confusion is ndarray (classes, classes), rows correct coluns predicted
        Returns: ndarray (classes,) containing sensitivity for each class
    """
    '''totals = sum(sum(label) for label in confusion)
    print("totals", totals)'''
    classes = len(confusion)
    return [confusion[i][i] / sum(confusion[i]) for i in range(classes)]
