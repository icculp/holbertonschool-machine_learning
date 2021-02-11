#!/usr/bin/env python3
"""
    Deep Convolutional Architectures
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ Builds a resnet50
        Input will have shape (224, 224, 3)
        Batch norm after each convolution,
            along channels axis and then ReLu
        Returns: keras model
    """
    init = K.initializers.he_normal()
    F11, F3, F12 = filters
    conv = K.layers.Conv2D(F11, kernel_size=(1, 1),
                           strides=s,
                           padding='valid',
                           kernel_initializer=init)(A_prev)
    bnorm = K.layers.BatchNormalization()(conv)
    relu = K.layers.Activation('relu')(bnorm)

    conv = K.layers.Conv2D(F3, kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='same',
                           kernel_initializer=init)(relu)
    bnorm = K.layers.BatchNormalization()(conv)
    relu = K.layers.Activation('relu')(bnorm)

    conv = K.layers.Conv2D(F12, kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='valid',
                           kernel_initializer=init)(relu)
    last = K.layers.BatchNormalization()(conv)

    shortcut = K.layers.Conv2D(F12, kernel_size=(1, 1),
                               strides=s,
                               padding='valid',
                               kernel_initializer=init)(A_prev)
    shortcut = K.layers.BatchNormalization()(shortcut)

    relu = K.layers.Add()([last, shortcut])
    out = K.layers.Activation('relu')(relu)

    return out
