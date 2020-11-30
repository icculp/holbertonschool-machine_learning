#!/usr/bin/env python3
"""
    Task 5
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


def add_matrices2D(mat1, mat2):
    """ Adds two arrays of same size, else returns None """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    new = mat1.copy()
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            new[i][j] += mat2[i][j]
    return new
