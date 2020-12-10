#!/usr/bin/env python3
"""
    Task 9
"""


def summation_i_squared(n):
    """ calculates the sum of i squared from 1 to n """
    sum = 0
    if type(n) is not int or n < 1:
        return None
    if n == 1:
        return 1
    else:
        sum += summation_i_squared(n - 1)
        sum += n * n
    return sum
