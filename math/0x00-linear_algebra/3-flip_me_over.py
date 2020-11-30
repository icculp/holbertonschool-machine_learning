#!/usr/bin/env python3
"""
    Task 3
"""


def matrix_transpose(matrix):
    """ Transpose (flip) a matrix """
    rows = len(matrix)
    cols = len(matrix[0])
    tranny = []
    for i in range(cols):
        row = []
        for j in range(rows):
            row.append(matrix[j][i])
        tranny.append(row.copy())
    return tranny
