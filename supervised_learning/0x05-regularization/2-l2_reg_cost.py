#!/usr/bin/env python3
"""
    Regularization project
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """ Calculates cost of NN w/ L2 regularization
        Returns: cost as a tensor
    """
    '''with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.nn.l2_normalize)
    t = tf.losses.get_regularization_losses()
    c = np.argmax(cost, axis=0)
    t = tf.contrib.layers.l2_regularizer(scale=1.0)
    tf.add_to_collection('t', t)
    '''
    '''reg_losses = tf.get_collection(tf.GraphKeys.LOSSES)
    l = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
    reg_constant = 0.01  # Choose an appropriate one.'''
    '''loss = cost + reg_constant * reg_losses'''
    '''ense = tf.layers.Dense(10,  kernel_regularizer=
                              tf.contrib.layers.l2_regularizer(.001))'''
    '''print(dir(cost))'''
    t = tf.losses.get_regularization_losses()
    '''print(t)'''
    '''t = tf.contrib.layers.l2_regularizer(np.random.uniform(0.01))'''
    return t + cost
