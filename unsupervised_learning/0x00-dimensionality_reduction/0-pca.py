#!/usr/bin/env python3
"""
    Dimensionality Reduction

    Only numpy allowed
    Your code should use the minimum number of
        operations to avoid floating point errors
"""
import numpy as np


def pca(X, var=0.95):
    """ performs PCA on a dataset
        X ndarray (n, d)
            n # data points
            d # of dims in each point
            all dims have mean of 0 across all points
        var is fraction of variance that PCA transform should maintain
        Returns: weights matrix, W ndarray (d, nd)
            nd new dims that transformed X
    """
    W = np.ndarray(X.shape)
    return W
