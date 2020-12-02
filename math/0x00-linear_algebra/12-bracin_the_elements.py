#!/usr/bin/env python3
"""
    Task 12
"""


def np_elementwise(mat1, mat2):
    """ transposes np array """
    sum = mat1 + mat2
    dif = mat1 - mat2
    prod = mat1 * mat2
    quot = mat1 / mat2
    return (sum, dif, prod, quot)
