#!/usr/bin/env python3
"""
    Bayesian Probability

    You are conducting a study on a revolutionary cancer drug
    and are looking to find the probability that a patient
    who takes this drug will develop severe side effects.
    During your trials, n patients take the drug and x patients
    develop severe side effects. You can assume that x follows
    a binomial distribution.

    only numpy allowed
"""
import numpy as np


def posterior(x, n, P, Pr):
    """ calculates posterior probability for the various hypothetical
            probabilities of developing severe side effects given the data
        x # of patients that develop severe side effects
        n is the total number of patients observed
        P 1D ndarray various hypothetical probabilities
            of developing severe side effects
        Pr 1D numpy.ndarray containing the prior beliefs of P
        Returns: 1D ndarray containing the intersection of obtaining
            x and n with each probability in P, respectively
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
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not (np.all(P >= 0) and np.all(P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not (np.all(Pr >= 0) and np.all(Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(1, np.sum(Pr)):
        raise ValueError("Pr must sum to 1")

    def factorial(m):
        """ calculates factorial of n """
        return np.math.factorial(m)

    likelihood = np.ndarray(P.shape)
    for p in range(len(P)):
        fact_n = factorial(n)
        pA = (factorial(n) /
              (factorial(x) * factorial(n - x)))
        pB = (np.power(P[p], x)) *\
             (np.power((1 - P[p]), (n - x)))
        likelihood[p] = pA * pB
    intersection = likelihood * Pr
    marginal = np.sum(intersection)
    return intersection / (marginal)
