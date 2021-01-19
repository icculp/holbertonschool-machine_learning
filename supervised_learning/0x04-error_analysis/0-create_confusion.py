#!/usr/bin/env python3
"""
    Error Analysis project
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix
        labels is a one-hot ndarray with shape (m, classes)
        containing the correct labels for each data point
        m number of data points
        classes number of classes
        logists is one-hot ndarray (m, classes) containing
        predicted labels
        Returns: confusion matrix ndarray (classes, classes)
        rows representing correct
        columns representing predicted
    """
    '''
    print("labels", labels, "classes",
          labels.shape[1], "data", labels.shape[0])
    print("logits", logits, "classes",
          logits.shape[1], "data", logits.shape[0])
    print("size", labels.size)
    label_to_ind = {y: x for x, y in enumerate(np.asarray(labels))}
    print(type(label_to_ind))
    print(label_to_ind)
    '''
    hot1 = np.argmax(labels, axis=0)
    hot2 = np.argmax(logits, axis=0)
    result = np.zeros((len(hot1), len(hot2)))
    for a, p in zip(labels, logits):
        '''print(labels[i].astype(int))'''
        '''print("a", a, "p", p)'''
        result[np.argmax(a, axis=0)][np.argmax(p, axis=0)] += 1
    return result
