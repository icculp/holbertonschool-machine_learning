#!/usr/bin/env python3
"""
    Transformer Applications
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def create_masks(inputs, target):
    ''' creates all masks for training/validation
        inputs is a tf.Tensor of shape (batch_size, seq_len_in)
            that contains the input sentence
        target is a tf.Tensor of shape (batch_size, seq_len_out)
            that contains the target sentence
        This function should only use tensorflow operations in order to
            properly function in the training step
        Returns: encoder_mask, combined_mask, decoder_mask
            encoder_mask is the tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) to be applied in the encoder
            combined_mask is the tf.Tensor of shape
                (batch_size, 1, seq_len_out, seq_len_out) used in the 1st
                attention block in the decoder to pad and mask future tokens
                in the input received by the decoder. It takes the maximum
                between a lookaheadmask and the decoder target padding mask.
            decoder_mask is the tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) used in the 2nd attention block
                in the decoder.
    '''
    batch_size, seq_len_in = tf.shape(inputs).numpy()
    _, seq_len_out = tf.shape(target).numpy()

    def create_look_ahead_mask(size):
        ''' lookahead mask '''
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def create_padding_mask(seq):
        ''' padding mask '''
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]

    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)
    look_ahead = create_look_ahead_mask(seq_len_out)
    decoder_target = create_padding_mask(target)
    combined_mask = tf.maximum(look_ahead, decoder_target)
    return encoder_mask, combined_mask, decoder_mask
