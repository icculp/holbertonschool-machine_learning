#!/usr/bin/env python3
"""
    Recurrent Neural Networks
"""
import numpy as np


class BidirectionalCell:
    """ represents a bidiretional cell of a RNN """

    def __init__(self, i, h, o):
        """
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs

            Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
                Whf and bhf are for the hidden states in the forward direction
                Whb and bhb are for the hidden states in the backward direction
                Wy and by are for the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h * 2, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
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
            Returns: h_next, the next hidden state
        """
        m, i = x_t.shape
        cat = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(cat @ self.Whf + self.bhf)
        h_next = np.tanh(cat @ self.Whf + self.bhf)
        return h_next
