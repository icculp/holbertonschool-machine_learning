#!/usr/bin/env python3
"""
    Task 16 advanced
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


def add_matrices(mat1, mat2):
    """ Adds two matrices """
    import numpy as np
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    return mat1 + mat2
