#!/usr/bin/env python3
"""
    Recurrent Neural Networks
"""
import numpy as np


class LSTMCell:
    """ represents a LTSM unit"""

    def __init__(self, i, h, o):
        """
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs

            Creates the public instance attributes
                    Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
                Wfand bf are for the forget gate
                Wuand bu are for the update gate
                Wcand bc are for the intermediate cell state
                Woand bo are for the output gate
                Wyand by are for the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """ performs forward propagation for one time step
            x_t is a ndarray  (m, i) that contains the data input for the cell
                m is the batch size for the data, i is dims
            h_prev is a ndarray (m, h) containing the previous hidden state
            c_prev is a ndarray (m, h) containing the previous cell state
            The output of the cell should use a softmax activation function
            Returns: h_next, c_next, y
                h_next is the next hidden state
                c_next is the next cell state
                y is the output of the cell
        """
        m, i = x_t.shape
        cat = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(cat @ self.Wf + self.bf)
        i = self.sigmoid(cat @ self.Wu + self.bu)
        c_hat = np.tanh(cat @ self.Wc + self.bc)
        c_next = f * c_prev + i * c_hat
        o = self.sigmoid(cat @ self.Wo + self.bo)
        h_next = o * np.tanh(c_next)
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, c_next, y
