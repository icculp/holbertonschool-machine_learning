#!/usr/bin/env python3
"""
    Bayesian Probability

    only numpy allowed
"""
import numpy as np


def likelihood(x, n, P):
    """
        You are conducting a study on a revolutionary cancer drug
            and are looking to find the probability that a patient
            who takes this drug will develop severe side effects.
            During your trials, n patients take the drug and x patients
            develop severe side effects. You can assume that x follows
            a binomial distribution.
        x # of patients that develop severe side effects
        n total number of patients observed
        P 1D ndarray various hypothetical probabilities of
            developing severe side effects
        Returns: 1D ndarray containing the likelihood of obtaining the data
            x and n, for each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater " +
                         "than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for value in P:
        if value > 1 or value < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
    '''if not np.all(P >= 0) and np.all(P <= 1):
        raise ValueError("All values in P must be in the range [0, 1]")'''

    def factorial(m):
        """ calculates factorial of n """
        # print('m', m)
        # r = range(1, m + 1)
        # print('r', r)
        return np.math.factorial(n)

    likelihood = np.ndarray(P.shape)
    pos = ((x / n))

    for p in range(len(P)):
        # print('n', n)
        fact_n = factorial(n)
        # print('fact_n', fact_n)
        likelihood[p] = ((factorial(n) /
                         (factorial(x) * factorial(n - x))) *
                         (np.power(P[p], x)) *
                         (np.power((1 - P[p]), (n - x))))
        # ((P[p]) * (1 - pB)) / (P[p])
    return P  # likelihood
