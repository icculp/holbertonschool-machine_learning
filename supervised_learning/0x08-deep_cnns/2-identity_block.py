#!/usr/bin/env python3
"""
    Deep Convolutional Architectures
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ Builds an identity block for resnet
        A_prev is output from previous layer
        filters is tuple/list (F11, F3, F12)
            F11 first 1x1 conv
            F3 3x3 conv
            F12 second 1x1 conv
        Batch norm after each convolution,
            along channels axis and ReLu
        Returns: activated output of block
    """
    init = K.initializers.he_normal()
    conv = K.layers.Conv2D(64, kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='valid',
                           kernel_initializer=init)(A_prev)
    bnorm = K.layers.BatchNormalization()(conv)
    relu = K.layers.Activation('relu')(bnorm)

    conv = K.layers.Conv2D(64, kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='same',
                           kernel_initializer=init)(relu)
    bnorm = K.layers.BatchNormalization()(conv)
    relu = K.layers.Activation('relu')(bnorm)

    conv = K.layers.Conv2D(256, kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='valid',
                           kernel_initializer=init)(relu)
    bnorm = K.layers.BatchNormalization()(conv)

    shortcut = K.layers.Add()([bnorm, A_prev])
    out = K.layers.Activation('relu')(shortcut)

    return out
