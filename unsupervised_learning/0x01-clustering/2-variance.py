#!/usr/bin/env python3
"""
    Clustering
"""
import numpy as np


def variance(X, C):
    """ Calculates the total intra-cluster variance
        X ndarray (n, d) dataset to cluster
            n # data points, d # dimensions
        C ndarray (k, d) centroid means for each cluster
            k # data points, d # dimensions
        Returns: var, or None on failure
            var is the total variance
    """
    n, d = X.shape
    centroids = np.random.uniform(low=X.min(),
                                  high=X.max(),
                                  size=(k, d))

    clss = np.random.uniform(low=0, high=(k), size=n)
    # randint
    for i in range(iterations):
        distances = np.array(
            [np.linalg.norm(X - c, axis=1) for c in centroids])
        # print(C)
        new_labels = np.argmin(distances)#, axis=1)

        if (clss == new_labels).all():
            # clss = new_labels
            break
        else:
            # difference = np.mean(clss != new_labels)
            # print(difference)
            clss = new_labels
            for c in range(k):
                centroids[c] = np.mean(X[c == clss], axis=0)
                # print(centroids[c])
    return centroids, clss
