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
    print("xxxx", X.shape)
    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    # H[-1] = np.zeros(h)
    Y = np.zeros((t, i))
    Y = []
    h_next = h_0
    for ti in range(t):
        # print('ti', ti)
        for l in range(len(rnn_cells)):
            # print('l', l)
            x = X[ti] if not l else H[ti + 1, l - 1]
            H[ti + 1, l], y = rnn_cells[l].forward(h_next[l], x)
            # print(H[ti + 1])
            # print('yyy', y.shape)
        Y.append(y)
        h_next = H[ti + 1]
    # print(Y)
    # print('yshape', np.array(Y).shape)
    # print('hshape', H.shape)
    return H, np.array(Y)
