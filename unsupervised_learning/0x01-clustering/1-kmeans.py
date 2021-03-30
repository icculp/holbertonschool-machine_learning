#!/usr/bin/env python3
"""
    Clustering
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """ Performs k-means on dataset
        X ndarray (n, d) dataset to cluster
            n # data points, d # dimensions
        k positive int # clusters
        iterations positive int max iterations
        If no change in centroids between iterations
            function should return
        Initialized with multivariate uniform distribution
            along each dimension in d
        If a cluster contains no data points during the update
            step, reinitialize its centroid
        Min values for distribution min of X along each dim
        You should use numpy.random.uniform exactly twice
        2 loops max
        Returns: (C, clss), or (None, None) on failure
            C ndarray (k, d)  centroid means for each cluster
            clss ndarray (n,) index of the cluster in C that
                each data point belongs to
    """
    n, d = X.shape
    centroids = np.random.uniform(low=X.min(axis=0),
                             high=X.max(axis=0),
                             size=(k, d))
    
    clss = np.random.randint(low=0, high=k, size=n)

    for i in range(iterations):
        C = np.array(
            [np.linalg.norm(X - c, axis=1) for c in centroids])
        new_labels = np.argmin(C, axis=0)

        if (clss == new_labels).all():
            clss = new_labels
            break
        else:
            difference = np.mean(clss != new_labels)
            clss = new_labels
            for c in range(k):
                centroids[c] = np.mean(X[clss == c], axis=0)
    return C, clss
