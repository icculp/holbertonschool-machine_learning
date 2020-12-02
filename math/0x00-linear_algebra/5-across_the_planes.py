#!/usr/bin/env python3
"""
    Task 5
"""


def matrix_shape(matrix):
    """ Return shape as list of integers """
    shape = []
    if type(matrix) != list:
        pass
    else:
        shape.append(len(matrix))
        shape += matrix_shape(matrix[0])
    return shape


def add_matrices2D(mat1, mat2):
    """ Adds two arrays of same size, else returns None """
    new = list(map(list, mat1))
    new2 = list(map(list, mat2))
    if matrix_shape(new) != matrix_shape(new2):
        return None
    for i in range(len(mat1)):
        for j in range(len(new[0])):
            new[i][j] += new2[i][j]
    return new
