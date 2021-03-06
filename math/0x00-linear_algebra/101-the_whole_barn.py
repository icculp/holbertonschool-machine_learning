#!/usr/bin/env python3
"""
    Task 16 advanced
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


def add_matrices(mat1, mat2):
    """ adds two without list comprehension """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if type(mat1[0]) == int or type(mat1[0]) == float:
        mat3 = []
        for i in range(len(mat1)):
            mat3.append(mat1[i] + mat2[i])
        return mat3
    else:
        mat3 = []
        for i in range(len(mat1)):
            mat3.append(add_matrices(mat1[i], mat2[i]))
        return mat3


'''
def add_matrices(mat1, mat2):
    """ Adds two matrices """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    return [[w + u for w in v] for u, v in zip(mat1, mat2)] '''
'''if type(mat1[0]) != int:
        return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
    else:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
'''
