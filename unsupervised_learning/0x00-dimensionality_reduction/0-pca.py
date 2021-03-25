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
    # X = X / np.std(X)
    n = X.shape[0]
    # V = np.cov(X.T)
    _, values, vectors = np.linalg.svd(X)
    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors.T
    vectors = vectors[:, idx]
    variance_explained = []
    for i in values:
        variance_explained.append((i / sum(values)))
    cum = np.cumsum(variance_explained)
    for c in range(len(cum)):
        # print(cum[c], var)
        if cum[c] > var:
            break
    return vectors[:, :c + 1]
