#!/usr/bin/env python3
"""
    Clustering
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ expectation step in the EM algorithm for a GMM
        X ndarray (n, d) dataset
            n # data points to evaluate pdf, d # dimensions
        pi ndarray (k,) priors for each cluster
        m ndarray (k, d) centroid means
        S ndarray (k, d, d) covariance of distribution
        1 loop max
        Returns: g, l, or None, None on failure
            g ndarray (k, n) containing posterior probabilities
            l total log likelihood
    """
    if type(X) is not np.ndarray or X.ndim != 2\
            or type(pi) is not np.ndarray or pi.ndim != 1\
            or type(m) is not np.ndarray or m.ndim != 2\
            or type(S) is not np.ndarray or S.ndim != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    try:

        return None
    except Exception:
        return None
