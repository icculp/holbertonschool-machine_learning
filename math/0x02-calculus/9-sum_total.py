#!/usr/bin/env python3
"""
    Task 9
"""


def summation_i_squared(n):
    """ calculates the sum of i squared from 1 to n """
    sum = 0
    for i in range(1, n + 1):
        sum += (i * i)
    return sum
