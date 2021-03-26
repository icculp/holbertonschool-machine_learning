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
from scipy import special
import numpy as np


def posterior(x, n, p1, p2):
    """ calculates posterior probability for the various hypothetical
            probabilities of developing severe side effects given the data
        x # of patients that develop severe side effects
        n is the total number of patients observed
        p1 is the lower bound on the range
        p2 is the upper bound on the range
        You can assume the prior beliefs of p follow a uniform distribution
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
    if not isinstance(p1, float) or\
            not (np.all(p1 >= 0) and np.all(p1 <= 1)):
        raise TypeError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or\
            not (np.all(p2 >= 0) and np.all(p2 <= 1)):
        raise TypeError("p1 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    def factorial(m):
        """ calculates factorial of n """
        return np.math.factorial(m)
    #p = np.uniform.random(range(1, 11))
    #print("p", p)
    beta = special.beta(p1, p2)
    print("beta", beta)
    P = np.linspace(0, 1, 11)
    print("P", P)
    likelihood = np.ndarray(P.shape)
    for p in range(len(P)):
        fact_n = factorial(n)
        pA = (factorial(n) /
              (factorial(x) * factorial(n - x)))
        pB = (np.power(P[p], x)) *\
             (np.power((1 - P[p]), (n - x)))
        likelihood[p] = pA * pB
    intersection = likelihood * beta
    marginal = np.sum(intersection)
    return (intersection / (marginal))
