#!/usr/bin/env python3
"""
    Task 9
"""


def summation_i_squared(n):
    """ calculates the sum of i squared from 1 to n """
    ''' Recursive solution reaching max depth
    sum = 0
    if type(n) is not int or n < 1:
        return None
    if n == 1:
        return 1
    else:
        sum += int(summation_i_squared(n - 1))
        sum += int(n * n)
    '''
    sum = (n * (n + 1) * ((2 * n) + 1)) / 6
    return int(sum)
