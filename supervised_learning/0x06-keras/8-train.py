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
                save_best=False, filepath=None,
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
    cb = []
    if validation_data is not None:
        if early_stopping is True:
            early = K.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience,
                                              mode='min')
            cb.append(early)
        if learning_rate_decay is True:
            def decayed_lr(step):
                """ Inverse time decay schedule function, step == epoch"""
                return alpha / (1 + decay_rate * (step))
            learn = K.callbacks.LearningRateScheduler(schedule=decayed_lr,
                                                      verbose=True)
            cb.append(learn)
        if save_best is True:
            saver = K.callbacks.ModelCheckpoint(filepath, save_best_only=True,
                                                mode='min')
            cb.append(saver)
    history = network.fit(data, labels, epochs=epochs,
                          batch_size=batch_size,
                          callbacks=cb,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle)
    if save_best is True:
        network.save(filepath)
    return history
