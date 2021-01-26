#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Converts a label vector into one-hot matrix
        The last dimension of one-hot matrix must be number of classes
        Returns: One-hot matrix
    """
    '''tf.compat.v1.enable_eager_execution()
    print(type(labels))'''
    '''oh = K.backend.one_hot(labels, labels.shape[-1])'''
    '''if classes is None:
        classes = labels.shape[-1]
    oh = K.backend.one_hot(labels, classes)
    return K.backend.get_session().run(oh)'''
    return K.utils.to_categorical(labels, classes)
