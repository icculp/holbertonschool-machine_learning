#!/usr/bin/env python3
"""
    Hidden Markov Models project
"""
import numpy as np


def regular(P):
    """ determines the steady state probabilities of a regular markov chain
        P square 2D ndarray (n, n) transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
        Returns: ndarray (1, n) containing the steady state
            probabilities, or None on failure
    """
    if type(P) is not np.ndarray or P.shape[0] != P.shape[1]:
        return None
    if not (P > 0).all():
        return None
    dim = P.shape[0]
    q = (P - np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q, ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ, bQT)
