#!/usr/bin/env python3
"""
    Deep Convolutional Architectures
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ assume input (224, 224, 3)
        Returns: output of transition layer and
            # of filters within the outputs
    """
    init = K.initializers.he_normal()
    x = K.Input(shape=(224, 224, 3))
    bnorm = K.layers.BatchNormalization()(x)
    relu = K.layers.Activation('relu')(bnorm)
    conv = K.layers.Conv2D(64, kernel_size=(7, 7),
                           strides=2,
                           padding='same',
                           kernel_initializer=init)(relu)
    maxy = K.layers.MaxPool2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(conv)
    X, nb_filters = dense_block(maxy, 64, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)
    avg = K.layers.AveragePooling2D(pool_size=(7, 7),
                                    strides=(1, 1))(X)
    dense = K.layers.Dense(1000, activation='softmax')(avg)
    model = K.Model(inputs=x, outputs=dense)
    return model
