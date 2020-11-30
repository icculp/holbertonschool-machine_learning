#!/usr/bin/env python3
"""
    Task 2
"""


def matrix_shape(matrix):
    """ Return shape as list of integers """
    shape = [len(matrix), len(matrix[0])]
    try:
        shape.append(len(matrix[0][0]))
    except Exception:
        pass
    try:
        shape.append(len(matrix[0][0][0]))
    except Exception:
        pass
    return shape
