#!/usr/bin/env python3
"""
    Task 10
"""


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    new = []
    if type(poly) is not list or len(poly) == 0:
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None
    for i in range(1, len(poly)):
        new.append(i * poly[i])
    if new == []:
        return [0]
    return new
