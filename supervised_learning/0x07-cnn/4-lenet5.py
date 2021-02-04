#!/usr/bin/env python3
"""
    Convolutional Neural Networks
"""
import tensorflow as tf


def lenet5(x, y):
    """ Builds modified LeNet-5 using tensorflow (without keras)
        x tf.placeholder (m, 28, 28, 1) w/ input images
        y tf.placeholder (m, 10) w/ one-hot labels

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
    init = tf.contrib.layers.variance_scaling_initializer()
    layer1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                              padding='same', activation='relu',
                              kernel_initializer=init)(x)
    layer2 = tf.layers.MaxPooling2D((2, 2), strides=(2, 2))(layer1)
    layer3 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                              padding='valid', activation='relu',
                              kernel_initializer=init)(layer2)
    layer4 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer3)
    layer5 = tf.layers.Dense(120, activation='relu',
                             kernel_initializer=init)(layer4)
    layer6 = tf.layers.Dense(84, activation='relu',
                             kernel_initializer=init)(layer5)
    flat = tf.layers.Flatten()(layer5)
    y_pred = tf.layers.Dense(10, activation='softmax',
                             kernel_initializer=init)(flat)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    grady = tf.train.AdamOptimizer(.001)
    op = grady.minimize(loss)
    acc = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(acc, tf.float32), name="Mean")
    return y_pred, op, loss, accuracy
