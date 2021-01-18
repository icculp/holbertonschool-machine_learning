#!/usr/bin/env python3

import numpy as np
Deep = __import__('28-deep_neural_network').DeepNeuralNetwork

def one_hot(Y, classes):
    """convert an array to a one hot encoding"""
    oh = np.zeros((classes, Y.shape[0]))
    oh[Y, np.arange(Y.shape[0])] = 1
    return oh

np.random.seed(6)
nx, m = np.random.randint(100, 200, 2).tolist()
classes = np.random.randint(5, 20)
X = np.random.randn(nx, m)
Y = one_hot(np.random.randint(0, classes, m), classes)

deep = Deep(nx, [100, 50, classes], activation='sig')
A, cost = deep.train(X, Y, iterations=10, graph=False, verbose=False)
np.set_printoptions(threshold=np.inf)
A = A.astype(float)
print(A)
print(cost)
print(deep.L)
for k, v in sorted(deep.cache.items()):
    print(k, v)
for k, v in sorted(deep.weights.items()):
    print(k, v)
