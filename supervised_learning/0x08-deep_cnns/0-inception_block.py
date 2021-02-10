#!/usr/bin/env python3
"""
    Deep Convolutional Architectures
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ Builds an inception block for deep CNN
        A_prev is output from previous layer
        filters tuple or list containing:
            F1 # of filters in 1x1 conv
            F3R # of filters in 1x1 before 3x3
            F3 # filters in 3x3
            F5R # filters in 1x1 before 5x5
            F5 # filters in 5x5
            FPP # filters in 1x1 after max pooling
        ReLu use on all inside inception block
        Returns: concatenated output of inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    one_conv3 = K.layers.Conv2D(F5R, kernel_size=(1, 1),
                                activation="relu", padding='same')(A_prev)
    one_conv2 = K.layers.Conv2D(F3R, kernel_size=(1, 1),
                                activation="relu", padding='same')(A_prev)
    three_conv = K.layers.Conv2D(F3, kernel_size=(3, 3),
                                 activation="relu", padding='same')(one_conv2)
    three_max_pool = K.layers.MaxPool2D(pool_size=(3, 3),
                                        strides=(1, 1), padding='same')(A_prev)
    five_conv = K.layers.Conv2D(F5, kernel_size=(5, 5),
                                activation="relu", padding='same')(one_conv3)
    one_conv4 = K.layers.Conv2D(FPP, kernel_size=(1, 1),
                                activation="relu",
                                padding='same')(three_max_pool)
    one_conv1 = K.layers.Conv2D(F1, kernel_size=(1, 1),
                                activation="relu", padding='same')(A_prev)
    out = K.layers.Concatenate()([one_conv1, three_conv, five_conv, one_conv4])
    return out
