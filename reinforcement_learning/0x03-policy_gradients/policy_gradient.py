#!/usr/bin/env python3
"""
    Policy Gradients
"""
import numpy as np


def softmax(z):
    # z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z).T, axis=0)).T
    return sm


def policy(matrix, weight):
    """ computes to policy with a weight matrix """
    return softmax(matrix.dot(weight))
    '''policy = []
    for i in range(len(weight)):
        policy.append(weight[i] * matrix.T[i])
        print('m', matrix.T[i])
        print('w', weight[i])
    return np.array(policy).sum(axis=0)'''
