#!/usr/bin/env python3
"""
    Dimensionality Reduction
    Only numpy allowed
    Your code should use the minimum number of
        operations to avoid floating point errors
"""
import numpy as np


def pca(X, ndim):
    """ performs PCA on a dataset
        X ndarray (n, d)
            n # data points
            d # of dims in each point
            all dims have mean of 0 across all points
        var is fraction of variance that PCA transform should maintain
        Returns: weights matrix, W ndarray (d, nd)
            nd new dims that transformed X
    """
    X = (X - np.mean(X, axis=0))  # / np.std(X)
    n = X.shape[0]
    # V = np.cov(X.T)
    U, values, vectors = np.linalg.svd(X)
    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors.T
    vectors = vectors[:, idx]
    # return X.dot(vectors.T)
    return X @ vectors[:, :ndim]
