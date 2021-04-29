#!/usr/bin/env python3
"""
    Recurrent Neural Networks
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ performs forward propagation for a deep RNN
        rnn_cells is a list of instances of length l for forward propagation
        X is the data to be used, ndarray (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state, ndarray (l, m, h)
            h is the dimensionality of the hidden state
        Returns: H, Y
            H ndarray containing all of the hidden states
            Y ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    H[-1] = np.zeros(h)
    Y = np.zeros((t, i))
    Y = []
    h_next = h_0
    for ti in range(t):
        H[ti + 1], y = rnn_cell.forward(h_next, X[ti])
        h_next = H[ti]
        Y.append(y)
    return H, np.array(Y)
