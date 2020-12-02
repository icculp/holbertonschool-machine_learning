#!/usr/bin/env python3
"""
    Task 8
"""


def mat_mul(mat1, mat2):
    """ Matrix multiplication (dot) """
    new = list(map(list, mat1))
    new2 = list(map(list, mat2))
    if len(new[0]) != len(new2):
        return None
    res = [[0] * len(new2[0]) for i in new]
    for i in range(len(new)):
        for j in range(len(new2[0])):
            for k in range(len(new2)):
                res[i][j] += new[i][k] * new2[k][j]
    return res
