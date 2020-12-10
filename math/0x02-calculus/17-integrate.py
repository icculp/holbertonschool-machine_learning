#!/usr/bin/env python3
"""
    Task 10
"""


def poly_integral(poly, C=0):
    """ calculates the integral of a polynomial """
    new = []
    if type(poly) is not list:
        return None
    if not all(isinstance(x, (int, float)) for x in poly)\
            or not isinstance(C, (int, float)):
        return None
    if type(C) is float and C.is_integer():
        C = int(C)
    new.append(C)
    for i in range(len(poly)):
        num = poly[i] / (i + 1)
        if num.is_integer():
            num = int(num)
        new.append(num)
    while new[-1] == 0 and len(new) > 1:
        new.pop()
    return new
