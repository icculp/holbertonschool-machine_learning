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
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()
    conv = K.layers.Conv2D(64, kernel_size=(7, 7),
                           strides=2,
                           padding='same',
                           kernel_initializer=init)(X)
    bnorm = K.layers.BatchNormalization()(conv)
    relu = K.layers.Activation('relu')(bnorm)

    three_max_pool = K.layers.MaxPool2D(pool_size=(3, 3),
                                        strides=(2, 2),
                                        padding='same')(relu)
    filters = (64, 64, 256)
    ident = projection_block(three_max_pool, filters, s=1)
    ident = identity_block(ident, filters)
    ident = identity_block(ident, filters)

    filters = (128, 128, 512)
    ident = projection_block(ident, filters)
    ident = identity_block(ident, filters)
    ident = identity_block(ident, filters)
    ident = identity_block(ident, filters)

    filters = (256, 256, 1024)
    ident = projection_block(ident, filters)
    ident = identity_block(ident, filters)
    ident = identity_block(ident, filters)
    ident = identity_block(ident, filters)
    ident = identity_block(ident, filters)
    ident = identity_block(ident, filters)

    filters = (512, 512, 2048)
    ident = projection_block(ident, filters)
    ident = identity_block(ident, filters)
    ident = identity_block(ident, filters)

    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=(1, 1),
                                        padding='same')(ident)
    dense = K.layers.Dense(1000, activation='softmax')(avgpool)
    model = K.Model(inputs=X, outputs=dense)
    return model
