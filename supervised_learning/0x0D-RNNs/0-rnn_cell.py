#!/usr/bin/env python3
"""
    Recurrent Neural Networks
"""
import numpy as np


class RNNCell:
    """ Represents a cell of a simple RNN """

    def __init__(self, i, h, o):
        """
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs

            Wh and bh are for the concatenated hidden state and input data
            Wy and by are for the output
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(z):
        ''' softmax activation function '''
        t = np.exp(z)
        a = np.exp(z) / np.sum(t, axis=1).reshape(-1, 1)
        return a

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step
            x_t is a ndarray  (m, i) that contains the data input for the cell
                m is the batch size for the data, i is dims
            h_prev is a ndarray (m, h) containing the previous hidden state
            The output of the cell should use a softmax activation function
            Returns: h_next, y
                h_next is the next hidden state
                y is the output of the cell
        """
        # softmax(arr, axis=0)
        m, i = x_t.shape
        Wi = self.Wh[:i]
        Wh = self.Wh[i:]
        cat = np.concatenate((h_prev, x_t), axis=1)
        # print('meow', cat.shape)
        h_next = np.tanh(cat @ self.Wh + self.bh)
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y
