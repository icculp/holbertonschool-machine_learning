#!/usr/bin/env python3
"""
    Convolutional Neural Networks
"""
import tensorflow.keras as K


def lenet5(X):
    """ Builds modified LeNet-5 using tensorflow (with keras)
        X K.Input (m, 28, 28, 1) w/ input images
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes

        Returns:
            a tensor for the softmax activated output
            a training operation that utilizes Adam optimization
            a tensor for the loss of the netowrk
            a tensor for the accuracy of the network
    """
    init = K.initializers.he_normal(seed=none)
    layer = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                            padding='same', activation='relu',
                            kernel_initializer=init,
                            )(X)
    layer = K.layers.MaxPool2D((2, 2), strides=(2, 2))(layer)
    layer = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                            padding='valid', activation='relu',
                            kernel_initializer=init)(layer)
    layer = K.layers.MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2))(layer)
    layer = K.layers.Flatten()(layer)
    layer = K.layers.Dense(120, activation='relu',
                           kernel_initializer=init)(layer)
    layer = K.layers.Dense(84, activation='relu',
                           kernel_initializer=init)(layer)
    layer = K.layers.Dense(10, activation='softmax',
                           kernel_initializer=init)(layer)
    model = K.Model(inputs=X, outputs=layer)
    adam = K.optimizers.Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
