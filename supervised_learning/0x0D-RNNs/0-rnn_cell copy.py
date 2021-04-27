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
        a = np.exp(z) / np.sum(t, axis=1).reshape(-1,1)
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
        from scipy.special import softmax
        # softmax(arr, axis=0)
        m, i = x_t.shape
        Wi = self.Wh[:i]
        Wh = self.Wh[i:]
        cat = np.concatenate((h_prev, x_t), axis=1)
        # print('meow', cat.shape)
        h_next = np.tanh(cat @ self.Wh + self.bh)
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y











        m, i = x_t.shape
        U = self.Wh[:i]
        W = self.Wh[i:]
        x = x_t
        T = len(x_t)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, len(self.Wh[:self.Wh.shape[1]]) ))
        s[-1] = np.zeros(self.Wh.shape[1])
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, len(self.Wh[:self.Wh.shape[1]])))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            #s[t] = np.tanh(U[:, x_t[]] + W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return s, o
        
        m, i = x_t.shape
        Wi = self.Wh[:i]
        Wh = self.Wh[i:]
        print("wi", Wi.shape, "wh", Wh.shape)
        print("wh", self.Wh.shape, "wy", self.Wy.shape)
        print("bh", self.bh.shape, "by", self.by.shape)
        print("xtshape", x_t.shape, "hprev", h_prev.shape)
        print("one", self.Wh[:i].shape)
        one = self.Wy.dot(x_t)# np.dot(x_t, Wh)  # x_t.dot(self.Wh[:i])
        two = h_prev @ Wh  # h_prev.dot(self.Wh[i:])
        sum = one + two
        h_next = np.tanh(sum + self.bh)
        soft = h_next @ self.Wy
        y = self.softmax(soft) # + self.by)
        return h_next, y
