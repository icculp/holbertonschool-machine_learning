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
    """
    '''#print("xshape", X.shape)
    #mean = np.mean(X.T)
    #print("mean", mean)'''
    #X = X / np.std(X)
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
    '''#pca_u, pca_s, pca_v = np.linalg.svd(V)'''
    valuees, values, vectors = np.linalg.svd(V)
    # print("pca_s", pca_u)
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
    '''#projection_matrix = (vectors.T[:][:c + 1]).T
    #projection_matrix = pca_v'''
    '''print("cum", cum)'''
    '''#W = x * x.T
    #T = X @ W
    #print(T)
    #print(len(W))
    #print(W.shape)
    #pca = X.dot(projection_matrix)
    #print(pca.shape)'''
    """
    """def svd_flip(u, v, u_based_decision=True):
            if u_based_decision:
                # columns of u, rows of v
                max_abs_cols = np.argmax(np.abs(u), axis=0)
                signs = np.sign(u[max_abs_cols, range(u.shape[1])])
                u *= signs
                v *= signs[:, np.newaxis]
            else:
                # rows of v, columns of u
                max_abs_rows = np.argmax(np.abs(v), axis=1)
                signs = np.sign(v[range(v.shape[0]), max_abs_rows])
                u *= signs
                v *= signs[:, np.newaxis]
            return u, v
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    components_ = Vt

        # Get variance explained by singular values
    explained_variance_ = (S ** 2) / (n - 1)
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = explained_variance_ / total_var
    singular_values_ = S.copy()  # Store the singular values.

    variance_explained = []
    for i in values:
        variance_explained.append((i / sum(values)))
    '''print("variance_explained", variance_explained)'''
    cum = np.cumsum(variance_explained)
    for c in range(len(cum)):
        if cum[c] > var:
            break
    c += 1

    return Vt#S# U, Vt"""
    cov = np.dot(X)
    U, S, V = np.linalg.svd(X)
    #values = U @ V.T
    # inspect shapes of the matrices
    #print(U.shape, S.shape, V.shape)
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
    assert X == (U @ np.diag(S) @ V)
    return -vectors[:, :c + 1]
