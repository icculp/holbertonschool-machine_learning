#!/usr/bin/env python3
"""
    Deep Convolutional Architectures
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ Builds a transition layer
        X is output from previous layer
        nb_filters is int of filters in X,
        compression is compression factor
        Returns: output of transition layer and
            # of filters within the outputs
    """
    init = K.initializers.he_normal()
    bnorm = K.layers.BatchNormalization()(X)
    relu = K.layers.Activation('relu')(bnorm)
    nb = int(nb_filters * compression)
    conv = K.layers.Conv2D(nb, kernel_size=(1, 1),
                           strides=1,
                           padding='same',
                           kernel_initializer=init)(relu)
    avg = K.layers.AveragePooling2D(pool_size=(2, 2),
                                    strides=(2, 2))(conv)

    return avg, nb
