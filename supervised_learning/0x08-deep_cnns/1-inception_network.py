#!/usr/bin/env python3
"""
    Deep Convolutional Architectures
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ Builds an inception network model
        Returns: keras model
    """
    x = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(64, kernel_size=(7, 7),
                                strides=(2, 2),
                                activation="relu", padding='same')(x)
    mpool = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2), padding='same')(one_conv1)
    conv = K.layers.Conv2D(F5R, kernel_size=(3, 3),
                                strides=(1, 1),
                                activation="relu", padding='same')(x)
    
    one = inception_block(x, [64, 96, 128, 16, 32, 32])
    two = inception_block(one, [64, 96, 128, 16, 32, 32])
    three = inception_block(two, [64, 96, 128, 16, 32, 32])
    model = K.Model(inputs=x, outputs=three)
    return model
