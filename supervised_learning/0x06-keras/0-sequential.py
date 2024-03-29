#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Buils a NN with Keras
        nx is number of input features
        layers is list containing the # of nodes in each layer
        activations is list containing activations used for each layer
        labmtha is L2 regularization parameter
        keep_prob is probability that node will be kept
        Returns: keras model

        not allowed to use Input class
    """
    weights = K.initializers.VarianceScaling(mode="fan_avg")
    reg = K.regularizers.l2(lambtha)
    model = K.Sequential()
    model.add(K.layers.Dense(layers[0],
                             activation=activations[0],
                             kernel_initializer=weights,
                             kernel_regularizer=reg,
                             input_shape=(nx,)))
    for i in range(1, len(layers)):
        reg = K.regularizers.l2(lambtha)
        model.add(K.layers.Dropout(rate=(1 - keep_prob)))
        model.add(
            K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_initializer=weights,
                           kernel_regularizer=reg)
        )
    return model
