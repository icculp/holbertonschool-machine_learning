#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Sets up Adam optimization for keras model with categorical
        crossentropy loss and accuracy metrics
        network is model to optimize
        alpha is learning rate
        beta1 is first Adam optimization parameter
        beta2 is second adam optmization parameter
        Returns: None
    """
    A = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=A,
                    metrics=['accuracy'])
