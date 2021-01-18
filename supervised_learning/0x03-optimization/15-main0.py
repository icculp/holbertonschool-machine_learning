#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
model = __import__('15-model').model
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
n1, n2, n3 = np.random.randint(50, 100, 3)
c = 10
layers = [n1, n2, n3, c]
activations = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None]
a, b1, b2 = np.random.uniform(0.01, size=3)
lib= np.load('../data/MNIST.npz')
X_train = lib['X_train'][s1:e1].reshape((e1 - s1, -1))
Y_train = one_hot(lib['Y_train'][s1:e1], c)
X_valid = lib['X_valid'][s2:e2].reshape((e2 - s2, -1))
Y_valid = one_hot(lib['Y_valid'][s2:e2], c)
tf.set_random_seed(0)
model((X_train, Y_train), (X_valid, Y_valid), layers=layers, activations=activations,
      alpha=a, beta1=b1, beta2=b2, batch_size=(2 ** b), epochs=1, save_path='./test.chpt')
