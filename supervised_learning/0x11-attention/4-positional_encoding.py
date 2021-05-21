#!/usr/bin/env python3
"""
    Attention
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """ calculates the positional encoding for a transformer
        max_seq_len is an integer representing the maximum sequence length
        dm is the model depth
        Returns: a numpy.ndarray of shape (max_seq_len, dm)
            containing the positional encoding vectors
    """
    pe = np.zeros((max_seq_len, dm))

    pe = np.array([[pos / np.power(10000, 2 * (j // 2) / dm)
                    for j in range(dm)]
                  for pos in range(max_seq_len)])
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    '''
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            pe[pos, i] = (np.sin(pos / (10000 ** ((2 * i) / dm))))
            pe[pos, i + 1] = (np.cos(pos / (10000 ** ((2 * (i + 1)) / dm))))'''
    return pe
