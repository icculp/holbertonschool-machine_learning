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
    '''#print("xshape", X.shape)
    #mean = np.mean(X.T)
    #print("mean", mean)'''
    X = X / np.std(X)
    '''#mean = np.expand_dims(mean, axis=0)
    #print("S", s)'''
    n = X.shape[0]
    # X = X / (n - 1)
    '''#X = standardize_data(X)
    print("XXXXXXXXXXXXXXX", X)
    #x = standardize_data(X)
    #cov = np.cov(x.T)
    #eigen_values, eigen_vectors = np.linalg.eig(cov)
    #M = np.mean(X.T, axis=1)
    #print("mshape", M.shape)
    #C = X - M
    #print("cshape", C.shape)
    #V = X.T * X * (1 / (n - 1))'''
    V = np.cov(X.T) / (n)
    '''#V = np.matmul(X.T, X) / n
    #V = np.dot(X.T,
    #		 X) / (X.shape[0] - 1)'''
    '''print("Vshape", V.shape)'''
    values, vectors = np.linalg.eig(V)
    '''print("values", values.shape)
    print('vectors', vectors.shape)'''
    '''#P = vectors.T.dot(C.T)'''
    '''#pca_u, pca_s, pca_v = np.linalg.svd(V)
    #values, values, vectors = np.linalg.svd(V)
    #print("pca_s", pca_u)'''
    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    '''#values = pca_s[idx]
    #vectors = pca_u[idx]'''
    '''print("VECTORSHAPE", vectors.shape)'''
    variance_explained = []
    for i in values:
        variance_explained.append((i / sum(values)))
    '''print("variance_explained", variance_explained)'''
    cum = np.cumsum(variance_explained)
    for c in range(len(cum)):
        if cum[c] > var:
            break
    c += 1
    return -vectors[:, :c + 1]