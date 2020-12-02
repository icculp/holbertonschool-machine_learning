#!/usr/bin/env python3
"""
    Task 15 advanced
"""
import numpy as np


def np_slice(matrix, axes={}):
    """ slices matrices """
    m = matrix.copy()
    n = np.array([])
    for k in axes.keys():
        val = axes[k]
        if len(val) == 3:
            start = val[0]
            stop = val[1]
            step = val[2]
        elif len(val) == 2:
            start = val[0]
            stop = val[1]
            step = None
        elif len(val) == 1:
            start = val[0]
            stop = None
            step = None
        else:
            start = None
            stop = None
            step = None
        s = slice(start, stop, step)
        i = range(1, m.shape[k])
        t = np.take(m,indices=i,axis=k)
        '''n = np.append(n, t[s])'''
        '''print(t[s])'''
    return t[s]
