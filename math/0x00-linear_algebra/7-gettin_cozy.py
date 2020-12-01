#!/usr/bin/env python3
"""
    Task 7
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices """
    new = list(map(list, mat1))
    new2 = list(map(list, mat2))
    if axis == 0:
        new.extend(new2.copy())
    else:
        for li in range(len(new)):
            new[li].extend(new2[li])
    return new.copy()
