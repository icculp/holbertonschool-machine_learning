#!/usr/bin/env python3
"""
    Attention
"""
import tensorflow as tf


# print('rnn 8')
class RNNEncoder(tf.keras.layers.Layer):
    """ RNN encoder for machine translation """

    def __init__(self, vocab, embedding, units, batch):
        """
            vocab: int representing the size of the input vocabulary
            embedding: int representing the dimensionality of embedding vector
            units: int representing the number of hidden units in the RNN cell
            batch: int representing the batch size

            Sets the following public instance attributes:
            batch - the batch size
            units - the number of hidden units in the RNN cell
            embedding - a keras Embedding layer that converts words from the
                vocabulary into an embedding vector
            gru - a keras GRU layer with units units
            Should return both the full sequence of outputs as well
                as the last hidden state
            Recurrent weights should be initialized with glorot_uniform
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        # print('27')
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        # print('29')
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # print('32')

    def initialize_hidden_state(self):
        """ Initializes the hidden states for the RNN cell to a tensor of zeros
            Returns: a tensor of shape (batch, units)
                containing the initialized hidden states
        """
        # print('39')
        t = tf.zeros((self.batch, self.units))
        # print('41')
        return t

    def call(self, x, initial):
        """
            x is a tensor of shape (batch, input_seq_len) containing
                input to encoder layer as word indices within the vocabulary
            initial is a tensor of shape (batch, units)
                containing the initial hidden state
            Returns: outputs, hidden
                outputs is a tensor (batch, input_seq_len, units)
                    containing the outputs of the encoder
                hidden is a tensor (batch, units)
                    containing the last hidden state of the encoder
        """
        # print('did we get called?')
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        # print(hidden.shape)
        # hidden = self.gru(x)
        return outputs, hidden
