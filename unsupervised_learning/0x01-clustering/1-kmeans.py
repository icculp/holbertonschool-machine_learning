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
    centroids = np.random.uniform(low=X.min(),
                                  high=X.max(),
                                  size=(k, d))

    clss = np.random.uniform(low=0, high=k, size=n)
    # randint
    for i in range(iterations):
        distances = np.array(
            [np.linalg.norm(X - c, axis=1) for c in centroids])
        # print(C)
        new_labels = np.argmin(distances, axis=0)

        if (clss == new_labels).all():
            clss = new_labels
            break
        else:
            # difference = np.mean(clss != new_labels)
            # print(difference)
            clss = new_labels
            for c in range(k):
                centroids[c] = np.mean(X[c == clss], axis=0)
                # print(centroids[c])
    return centroids, clss
