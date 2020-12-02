#!/usr/bin/env python3
"""
    Task 12
"""
import numpy as np


def np_elementwise(mat1, mat2):
    """ transposes np array """
    print(dir(mat1))
    sum = np.add(mat1, mat2)
    dif = np.subtract(mat1, mat2)
    prod = np.multiply(mat1, mat2)
    quot = np.divide(mat1, mat2)
    return (sum, dif, prod, quot)
