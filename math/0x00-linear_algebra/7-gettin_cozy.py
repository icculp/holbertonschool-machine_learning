#!/usr/bin/env python3
"""
    Task 7
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices """
    new = list(map(list, mat1))
    new2 = list(map(list, mat2))
    if axis == 0:
        try:
            if len(new[0]) != len(new2[0]):
                return None
        except Exception:
            return None
        new.extend(new2.copy())
    elif axis == 1:
        if len(new) != len(new2):
            return None
        for li in range(len(new)):
            new[li].extend(new2[li])
    else:
        return None
    return new.copy()
