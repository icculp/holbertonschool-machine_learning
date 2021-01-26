#!/usr/bin/env python3
"""
    Keras project (finally)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
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
    early = K.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=patience,
                                      mode='min')
    cb = []
    if early_stopping is True and validation_data is not None:
        cb.append(early)
    history = network.fit(data, labels, epochs=epochs,
                          batch_size=batch_size,
                          callbacks=cb,
                          verbose=verbose, shuffle=shuffle)
    return history
