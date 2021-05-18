#!/usr/bin/env python3
"""
    Attention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ calculate the attention for machine translation  """

    def __init__(self, units):
        """
            units is an integer representing the number of hidden
                units in the alignment model
            Sets the following public instance attributes:
            W - a Dense layer with units units, to be applied to
                the previous decoder hidden state
            U - a Dense layer with units units, to be applied to
                the encoder hidden states
            V - a Dense layer with 1 units, to be applied to the
                tanh of the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = 0
        self.U = 0
        self.V = 0

    def call(self, s_prev, hidden_states):
        """
            s_prev is a tensor of shape (batch, units)
                containing the previous decoder hidden state
            hidden_states is a tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder
            Returns: context, weights
            context is a tensor of shape (batch, units) that contains
                the context vector for the decoder
            weights is a tensor of shape (batch, input_seq_len, 1)
                that contains the attention weights
        """
        return context, weights
