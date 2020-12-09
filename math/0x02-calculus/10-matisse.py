#!/usr/bin/env python3
"""
    Task 10
"""


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    new = []
    poly = [3, 3]
    if type(poly) is not list:
        return None
    for i in range(1, len(poly)):
        new.append(i * poly[i])
    if new == []:
        return [0]
    return new
