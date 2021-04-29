#!/usr/bin/env python3
"""
    Recurrent Neural Networks
"""
import numpy as np


class GRUCell:
    """ represents a gated recurrent unit """

    def __init__(self, i, h, o):
        """
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs

            Creates public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
                Wz and bz are for the update gate
                Wr and br are for the reset gate
                Wh and bh are for the intermediate hidden state
                Wy and by are for the output
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        ''' sigmoid function '''
        return 1 / (1 + np.exp(-x))

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
        m, i = x_t.shape
        cat = np.concatenate((h_prev, x_t), axis=1)
        cc = self.Wh[:i]
        # print('meow', cat.shape)
        z = self.sigmoid(cat @ self.Wz + self.bz)
        r = self.sigmoid(cat @ self.Wr + self.br)
        # print('rrrr', r)
        cat2 = np.concatenate(((h_prev * r), x_t), axis=1)  # * r
        h = np.tanh(((cat2) @ self.Wh) + self.bh)
        # print(h)
        h_next = (np.ones_like(z) - z) * h_prev + z * h
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y
