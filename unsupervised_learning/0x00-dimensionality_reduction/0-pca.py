#!/usr/bin/env python3
"""
    Dimensionality Reduction

    Only numpy allowed
    Your code should use the minimum number of
        operations to avoid floating point errors
"""
import numpy as np


def standardize_data(X):
         
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = X.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(X[:,column])
        std = np.std(X[:,column])
        tempArray = np.empty(0)
        
        for element in X[:,column]:
            
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray


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
    #print("xshape", X.shape)
    #mean = np.mean(X.T)
    #print("mean", mean)
    s = np.std(X)
    #mean = np.expand_dims(mean, axis=0)
    #print("S", s)
    n = X.shape[0]
    #X = X - mean
    X = X / s
    print("XXXXXXXXXXXXXXX", X)
    #x = standardize_data(X)
    #cov = np.cov(x.T)
    #eigen_values, eigen_vectors = np.linalg.eig(cov)
    #M = np.mean(X.T, axis=1)
    #print("mshape", M.shape)
    #C = X - M
    #print("cshape", C.shape)
    V = np.cov(X.T)# / (n - 1)
    #V = X.T * X * (1 / (n - 1))
    print("Vshape", V.shape)
    values, vectors = np.linalg.eig(V)
    print("values", values.shape)
    print('vectors', vectors.shape)
    #P = vectors.T.dot(C.T)
    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    variance_explained = []
    for i in values:
        variance_explained.append((i / sum(values)))
    print("variance_explained", variance_explained)
    cum = np.cumsum(variance_explained)
    for c in range(len(cum)):
        if cum[c] > var:
            break
    c += 1
    projection_matrix = (vectors.T[:][:c + 1]).T
    print("cum", cum)
    #W = x * x.T
    #T = X @ W
    #print(T)
    #print(len(W))
    #print(W.shape)
    #pca = X.dot(projection_matrix)
    #print(pca.shape)
    return projection_matrix * -1.0
