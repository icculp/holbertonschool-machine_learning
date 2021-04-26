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
        """
        self.Wh = 
        self.Wy = 
        self.bh = 
        self.by = 

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step
            x_t is a ndarray  (m, i) that contains the data input for the cell
                m is the batche size for the data, i dims
            h_prev is a ndarray (m, h) containing the previous hidden state
            The output of the cell should use a softmax activation function
            Returns: h_next, y
                h_next is the next hidden state
                y is the output of the cell
        """
        return h_next, y
