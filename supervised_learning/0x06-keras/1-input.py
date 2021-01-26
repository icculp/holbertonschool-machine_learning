#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow as tf
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
    x = tf.placeholder("float", shape=(None, nx), name='x')
    weights = K.initializers.VarianceScaling(mode="fan_avg")
    '''drop = K.layers.Dropout(rate=keep_prob)'''
    reg = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    '''model = K.Sequential()'''
    '''model.add(tf.keras.Input(shape=())'''
    layer1 = K.layers.Dense(layers[0],
                            activation=activations[0],
                            kernel_initializer=weights,
                            kernel_regularizer=reg)
    drop1 = K.layers.Dropout(rate=(1 - keep_prob))
    layer2 = K.layers.Dense(layers[1],
                            activation=activations[1],
                            kernel_initializer=weights,
                            kernel_regularizer=reg)
    drop2 = K.layers.Dropout(rate=(1 - keep_prob))
    layer3 = K.layers.Dense(layers[2],
                            activation=activations[2],
                            kernel_initializer=weights,
                            kernel_regularizer=reg)
    '''#if i != len(layers) - 1:
        #model.add(drop)'''
    '''for la in model.layers:
        print(dir(la))
        break
    print(len(layers))
    print(len(activations))'''
    out = layer3(drop2(layer2(drop1(layer1(inputs)))))
    return K.Model(inputs=inputs, outputs=out)
