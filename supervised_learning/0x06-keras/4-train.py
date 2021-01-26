#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow as tf
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """ Trains a model using mini-bath gradient descent
        now using keras api in tensorflow
        network is model to train
        data is ndarray (m, nx)
        labels is one-hot ndarray (m, classes), labels of data
        batch_size is size of batches
        epochs is numbber of passes of gradient descent on every example
        verbose is bool for printing during training
        shuffle is bool whether to shuffle at each epoch
        Returns: History object generated after training
    """
    network.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
    network.fit(data, labels, epochs=epochs, batch_size=batch_size)
    return
