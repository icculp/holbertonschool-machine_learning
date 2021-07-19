#!/usr/bin/env python3
"""
    Pandas project
"""
import pandas as pd


def from_numpy(array):
    """ creates a pd.DataFrame from a np.ndarray
        array is np.ndarray to convert
        columns to be labeled in alphabetical order
        not more than 26 columns
        returns pd.df
    """
    col_len = array.shape[1]
    alph = 65
    cols = [chr(alph + i) for i in range(col_len)]
    return pd.DataFrame(array, columns=cols)
