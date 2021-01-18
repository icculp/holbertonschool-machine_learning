#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh
# set variables
np.random.seed(15)
s1, s2 = np.random.randint(15, 50, 2)
b1, b2 = np.random.randint(100, 125, 2)
b = np.random.randint(4, 10)
e1 = s1 + (b1 * (2 ** b))
e2 = s2 + (b2 * (2 ** b))
c = 10
lib= np.load('../data/MNIST.npz')
X_train = lib['X_train'][s1:e1].reshape((e1 - s1, -1))
Y_train = one_hot(lib['Y_train'][s1:e1], c)
X_valid = lib['X_valid'][s2:e2].reshape((e2 - s2, -1))
Y_valid = one_hot(lib['Y_valid'][s2:e2], c)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./test.chpt.meta')
    saver.restore(sess, './test.chpt')
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    y_pred = tf.get_collection('y_pred')[0]
    accuracy = tf.get_collection('accuracy')[0]
    loss = tf.get_collection('loss')[0]
    Tacc, Tcost = sess.run((accuracy, loss), feed_dict={x:X_train, y:Y_train})
    Vacc, Vcost = sess.run((accuracy, loss), feed_dict={x:X_valid, y:Y_valid})
    print("\tTraining Cost: {}".format(Tcost))
    print("\tTraining Accuracy: {}".format(Tacc))
    print("\tValidation Cost: {}".format(Vcost))
    print("\tValidation Accuracy: {}".format(Vacc))
