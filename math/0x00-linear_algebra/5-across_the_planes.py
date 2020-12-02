#!/usr/bin/env python3
"""
    Task 5
"""


def add_matrices2D(mat1, mat2):
    """ Adds two arrays of same size, else returns None """
    new = list(map(list, mat1))
    new2 = list(map(list, mat2))

    try:
        if len(mat1) != len(mat2):
            return None
        if len(mat1[0]) != len(mat2[0]):
            return None
    except Exception:
        return None

    for i in range(len(mat1)):
        for j in range(len(new[0])):
            new[i][j] += new2[i][j]
    return new
