#!/usr/bin/env python3
"""
    Task 2
"""


def matrix_shape(matrix):
    """ Return shape as list of integers """
    shape = []
    if type(matrix) == int:
        pass
    else: 
        shape.append(len(matrix))
        shape += matrix_shape(matrix[0])
    return shape
    """
    try:
        shape.append(len(matrix[0][0]))
    except Exception:
        pass
    try:
        shape.append(len(matrix[0][0][0]))
    except Exception:
        pass
    return shape
    """
