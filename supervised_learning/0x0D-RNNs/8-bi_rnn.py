#!/usr/bin/env python3
"""
    Recurrent Neural Networks
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ performs forward prop for a bidirectional RNN

        bi_cell instance of BidirectinalCell that used for forward propagation
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state in forward direction, ndarray  (m, h)
            h is the dimensionality of the hidden state
        h_t is the initial hidden state in backward direction, ndarray (m, h)
        Returns: H, Y
            H is a ndarray containing all of the concatenated hidden states
            Y is a ndarray containing all of the outputs
    """
    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t, m, h * 2))
    F = np.zeros((t, m, h))
    B = np.zeros((t, m, h))
    Y = []
    h_prev = h_0
    h_next = h_t
    for ti in range(t):
        F[ti] = bi_cell.forward(h_prev, X[ti])
        B[t - ti - 1] = bi_cell.backward(h_next, X[t - ti - 1])
        # H[ti] = np.concatenate((f, b), axis=-1)
        # y = bi_cell.output)
        # Y.append(y)
        h_next = B[t - ti - 1]
        h_prev = F[ti]
    H = np.concatenate((F, B), axis=-1)
    Y = bi_cell.output(H)
    return H, np.array(Y)
