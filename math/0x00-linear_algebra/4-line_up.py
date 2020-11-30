#!/usr/bin/env python3
"""
    Task 4
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


def add_arrays(arr1, arr2):
    """ Adds two arrays of same size, else returns None """
    if matrix_shape(arr1) != matrix_shape(arr2):
        return None
    new = []
    for i in range(len(arr1)):
        new.append(arr1[i] + arr2[i])
    return new
