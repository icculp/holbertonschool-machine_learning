#!/usr/bin/env python3
"""
    Deep Convolutional Architectures
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ Builds a dense block
        X is output from previous layer
        nb_filters is int of filters in X,
        growth_rate gr for dense block
        layers # layers in block
        Use bottleneck layers used for DenseNet-B
        he normal weight initialization
        batchnorm and ReLu for all convs
        Returns: concatenated output of each layer within
            dense block and # of filters within the outputs
    """

    for layer in range(layers):
        init = K.initializers.he_normal()
        bnorm = K.layers.BatchNormalization()(X)
        relu = K.layers.Activation('relu')(bnorm)
        conv = K.layers.Conv2D(growth_rate * 4, kernel_size=(1, 1),
                               strides=1,
                               padding='same',
                               kernel_initializer=init)(relu)
        bnorm = K.layers.BatchNormalization()(conv)
        relu = K.layers.Activation('relu')(bnorm)
        conv = K.layers.Conv2D(growth_rate, kernel_size=(3, 3),
                               strides=1,
                               padding='same',
                               kernel_initializer=init)(relu)
        nb_filters += growth_rate
        X = K.layers.Concatenate(axis=-1)([X, conv])

    return X, nb_filters
