#!/usr/bin/env python3
"""
    Pandas project
"""
import pandas as pd


def from_file(filename, delimeter):
    """  loads data from a file as a pd.DataFrame:

        filename is the file to load from
        delimiter is the column separator
        Returns: the loaded pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimeter)
