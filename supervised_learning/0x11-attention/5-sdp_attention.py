#!/usr/bin/env python3
"""
    Attention
"""
import numpy as np
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
            containing the query matrix
        K is a tensor with its last two dimensions as (..., seq_len_v, dk)
            containing the key matrix
        V is a tensor with its last two dimensions as (..., seq_len_v, dv)
            containing the value matrix
        mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v)
            containing the optional mask, or defaulted to None
        if mask is not None, multiply -1e9 to the mask and add it to the
            scaled matrix multiplication
        The preceding dimensions of Q, K, and V are the same
        Returns: output, weights
            output a tensor with its last two dimensions as
                (..., seq_len_q, dv) containing scaled dot product attention
            weights a tensor with its last two dimensions as
                (..., seq_len_q, seq_len_v) containing the attention weights
    """
    def softmax(x):
        ''' softmax '''
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)

    attn = tf.matmul(Q, K, transpose_b=True)

    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)

    # attn = self.dropout(F.softmax(attn, dim=-1))
    output = tf.matmul(attn, V)

    return output, tf.nn.softmax(attn)
    # return output, weights
