#!/usr/bin/env python3
"""
    Advanced Linear Algebra

    not allowed to im-port any module
    must be done by hand!
"""
import numpy as np


def definiteness(matrix):
    """ Calculates definiteness of a matrix
        matrix is a ndarray (n, n) whose definiteness should be calculated
        Return: the string: Positive definite,
                            Positive semi-definite,
                            Negative semi-definite,
                            Negative definite, or
                            Indefinite
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix) >= 1 and np.array_equal(matrix, matrix.T):
        eigvals = np.linalg.eigvals(matrix)
        '''print(matrix)
        print("eigvals", eigvals)'''
        if np.all(eigvals > 0):
            return "Positive definite"
        elif np.all(eigvals >= 0):
            return "Positive semi-definite"
        elif np.all(eigvals < 0):
            return "Negative definite"
        elif np.all(eigvals <= 0):
            return "Negative semi-definite"
        else:
            return "Indefinite"
    else:
        return None
