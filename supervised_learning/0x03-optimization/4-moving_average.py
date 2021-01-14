#!/usr/bin/env python3
"""
    Optimization project
"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def moving_average(data, beta):
    """ Calculates the exponential weighted moving average of dataset
        data is list of data to calculate ma
        beta is weight used for moving average
        bias correction is used
        Returns: list containing ma's of data
    """
    days = [x for x in range(1, len(data) + 1)]
    vt = 0
    wma = []
    for day in days:
        vt = (vt * beta) + ((1 - beta) * data[day - 1])
        bias_correction = 1 - (beta ** day)
        wma.append(vt / bias_correction)
    return wma
