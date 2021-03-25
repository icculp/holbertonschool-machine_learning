#!/usr/bin/env python3
"""
    Dimensionality Reduction
    Only numpy allowed
    Your code should use the minimum number of
        operations to avoid floating point errors
"""
import numpy as np


def P_init(X, perplexity):
    """ performs PCA on a dataset
        X ndarray (n, d) dataset to be transformed by t-SNE
            n is the number of data points
            d is the number of dimensions in each point
        perplexity all Gaussian distributions should have
        Returns: (D, P, betas, H)
            D: ndarray (n, n) calculates squared pairwise
                distance between two data points
            The diagonal of D should be 0s
            P: ndarray (n, n) initialized to all 0‘s
                that will contain the P affinities
            betas: ndarray (n, 1) initialized to all 1’s
                that will contain all of the beta values
                \beta_{i} = \frac{1}{2{\\sigma_{i}}^{2} }
            H is the Shannon entropy for perplexity perplexity with a base of 2
    """
    n, d = X.shape
    D = np.ndarray((n, n))
    P = np.ndarray((n, n))
    betas = np.ndarray((n, 1))
    H = 2
    return D, P, betas, H
